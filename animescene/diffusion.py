"""SDXL + ControlNet-Union sampler with regional (per-character) prompting.

Built directly on diffusers *components* (UNet, ControlNetUnion, VAE, CLIPs)
with our own denoising loop, because the stock pipelines can't do what the
stage gives us:

* **Regional cross-attention**: every character's prompt only speaks to the
  latent pixels its 3D silhouette owns (plus a shared scene prompt mixed in
  everywhere), so four fighters keep four distinct designs and palettes.
* **Multi-condition ControlNet-Union**: OpenPose + depth (+ optional normal)
  rendered from the same 3D scene, so the pose and the occlusion order agree.
* **txt2img, img2img and masked inpaint** in the same loop, used for the
  base pass, the hires pass and the per-character detail passes.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CONTROL_MODES = {"openpose": 0, "depth": 1, "hed": 2, "scribble": 2, "canny": 3, "lineart": 3,
                 "normal": 4, "segment": 5, "tile": 6, "repaint": 7}
DEFAULT_MODELS = os.environ.get("ANIMESCENE_MODELS", "/home/user/models")


# ------------------------------------------------------------------ regional attention
@dataclass
class RegionState:
    active: bool = False
    contexts: list = field(default_factory=list)  # per region (1, L, 2048) cond embeddings
    masks: torch.Tensor | None = None  # (R, h, w) latent-resolution soft masks
    base_weight: float = 0.3
    cache: dict = field(default_factory=dict)

    def masks_for(self, n_tokens: int, dtype) -> torch.Tensor:
        """(R, N, 1) masks resampled to an attention layer's token grid."""
        if n_tokens in self.cache:
            return self.cache[n_tokens]
        R, h, w = self.masks.shape
        for k in range(6):
            hh, ww = math.ceil(h / 2 ** k), math.ceil(w / 2 ** k)
            if hh * ww == n_tokens:
                break
        else:
            raise RuntimeError(f"cannot map {n_tokens} tokens to latent {h}x{w}")
        m = F.interpolate(self.masks[None].float(), size=(hh, ww), mode="area")[0]
        m = m.reshape(R, -1, 1).to(dtype)
        self.cache[n_tokens] = m
        return m


def _attend(attn, hidden, ctx, bias=None):
    q = attn.to_q(hidden)
    k = attn.to_k(ctx)
    val = attn.to_v(ctx)
    B, N, _ = q.shape
    hd = k.shape[-1] // attn.heads
    q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
    k = k.view(k.shape[0], -1, attn.heads, hd).transpose(1, 2)
    val = val.view(val.shape[0], -1, attn.heads, hd).transpose(1, 2)
    if k.shape[0] != B:
        k, val = k.expand(B, -1, -1, -1), val.expand(B, -1, -1, -1)
    o = F.scaled_dot_product_attention(q, k, val, attn_mask=bias)
    o = o.transpose(1, 2).reshape(B, N, attn.heads * hd).to(q.dtype)
    o = attn.to_out[0](o)
    return attn.to_out[1](o)


class RegionalCrossAttn:
    """Cross-attention processor: batch is [uncond, cond]; the cond half is a mask-weighted
    mixture of the base prompt and one prompt per region (the last region = background)."""

    def __init__(self, state: RegionState):
        self.state = state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 **kwargs):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        out = _attend(attn, hidden_states, encoder_hidden_states)
        st = self.state
        if not st.active or not st.contexts:
            return out
        B = hidden_states.shape[0]
        half = B // 2
        hc = hidden_states[half:]
        masks = st.masks_for(hc.shape[1], hc.dtype)  # (R, N, 1)
        b = st.base_weight
        leftover = (1 - masks.sum(0)).clamp(min=0)[None]  # pixels no region owns -> base prompt
        mix = out[half:] * (b + (1 - b) * leftover)
        for r, ctx in enumerate(st.contexts):
            mix = mix + (1 - b) * masks[r][None] * _attend(attn, hc, ctx.to(hc.dtype))
        return torch.cat([out[:half], mix], 0)


# ------------------------------------------------------------------ reference attention
@dataclass
class Reference:
    """A character identity anchor: its sheet image, its prompt, and where it applies."""
    image: Image.Image
    prompt: str
    mask: np.ndarray | None = None  # HxW in output pixels; None = whole image
    weight: float = 1.0


@dataclass
class RefState:
    mode: str = "off"  # off | write | read
    banks: list = field(default_factory=list)  # per reference: {layer: (1, N_r, C)}
    current: int = 0
    masks: torch.Tensor | None = None  # (R, h, w) latent-resolution; rows of ones = everywhere
    weights: list = field(default_factory=list)
    style_fidelity: float = 0.5
    cache: dict = field(default_factory=dict)

    def bias(self, n_tokens: int, ref_lens: list, dtype) -> torch.Tensor:
        """Additive attention bias (N, N + sum(ref_lens)): each pixel may attend to its own
        character's reference tokens only (log-weighted), never to another character's."""
        key = (n_tokens, tuple(ref_lens))
        if key in self.cache:
            return self.cache[key]
        R, h, w = self.masks.shape
        for k in range(6):
            hh, ww = math.ceil(h / 2 ** k), math.ceil(w / 2 ** k)
            if hh * ww == n_tokens:
                break
        else:
            raise RuntimeError(f"cannot map {n_tokens} tokens to latent {h}x{w}")
        m = F.interpolate(self.masks[None].float(), size=(hh, ww), mode="area")[0].reshape(R, -1)
        cols = [torch.zeros(n_tokens, n_tokens)]
        for r, L in enumerate(ref_lens):
            wgt = torch.clamp(m[r] * self.weights[r], min=1e-6).log()
            wgt[m[r] < 0.05] = float("-inf")
            cols.append(wgt[:, None].expand(n_tokens, L))
        b = torch.cat(cols, 1).to(dtype)
        self.cache[key] = b
        return b


class RefSelfAttn:
    """Self-attention that, in "write" mode, records a reference image's features and, in "read"
    mode, lets each pixel also attend to the features of its own character's reference sheet."""

    def __init__(self, state: RefState, name: str):
        self.state, self.name = state, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        st = self.state
        if st.mode == "write":
            st.banks[st.current][self.name] = hidden_states.detach()
            return _attend(attn, hidden_states, hidden_states)
        if st.mode != "read" or not st.banks:
            return _attend(attn, hidden_states, hidden_states)
        B, N, _ = hidden_states.shape
        refs = [b[self.name].to(hidden_states.dtype) for b in st.banks]
        ctx = torch.cat([hidden_states] + [r.expand(B, -1, -1) for r in refs], 1)
        bias = st.bias(N, [r.shape[1] for r in refs], hidden_states.dtype)
        out = _attend(attn, hidden_states, ctx, bias)
        if B >= 2 and st.style_fidelity > 0:  # uncond half: partly plain, keeps the CFG contrast
            half = B // 2
            plain = _attend(attn, hidden_states[:half], hidden_states[:half])
            out = torch.cat([st.style_fidelity * plain + (1 - st.style_fidelity) * out[:half], out[half:]])
        return out


def _fit_area(size, area):
    """Scale (w, h) to about `area` pixels, multiples of 64."""
    w, h = size
    k = math.sqrt(area / (w * h))
    return max(64, int(round(w * k / 64)) * 64), max(64, int(round(h * k / 64)) * 64)


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, models: str = DEFAULT_MODELS, dtype=torch.bfloat16, threads: int | None = None):
        from diffusers import AutoencoderKL, ControlNetUnionModel, DPMSolverMultistepScheduler, \
            UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        torch.set_num_threads(threads or os.cpu_count() or 4)
        t0 = time.time()
        base = os.path.join(models, "sdxl")
        self.dtype = dtype
        self.tok1 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
        self.te1 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).eval()
        self.te2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2",
                                                               torch_dtype=dtype).eval()
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype).eval()
        vae_dir = os.path.join(models, "vae_fix")
        self.vae = AutoencoderKL.from_pretrained(vae_dir if os.path.isdir(vae_dir) else base,
                                                 **({} if os.path.isdir(vae_dir) else {"subfolder": "vae"}),
                                                 torch_dtype=torch.float32).eval()
        self.vae.enable_tiling()
        cn_dir = os.path.join(models, "controlnet_union")
        self.controlnet = ControlNetUnionModel.from_pretrained(cn_dir, torch_dtype=dtype).eval() \
            if os.path.isdir(cn_dir) else None
        self.sched_config = DPMSolverMultistepScheduler.load_config(base, subfolder="scheduler")
        self.region = RegionState()
        self.ref = RefState()
        procs = {}
        for name, proc in self.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                procs[name] = RegionalCrossAttn(self.region)
            elif name.endswith("attn1.processor"):
                procs[name] = RefSelfAttn(self.ref, name)
            else:
                procs[name] = proc
        self.unet.set_attn_processor(procs)
        self.loaded_loras: dict = {}
        print(f"[engine] loaded in {time.time() - t0:.0f}s", flush=True)

    # -------------------------------------------------------------- prompts
    @torch.no_grad()
    def encode(self, prompt: str, chunks: int | None = None):
        """Long-prompt SDXL encoding (A1111-style 75-token chunks). Returns (emb, pooled, n_chunks)."""
        ids1 = self.tok1(prompt, add_special_tokens=False).input_ids
        ids2 = self.tok2(prompt, add_special_tokens=False).input_ids
        n = max(1, math.ceil(max(len(ids1), len(ids2)) / 75))
        if chunks is not None:
            if n > chunks:
                ids1, ids2, n = ids1[:75 * chunks], ids2[:75 * chunks], chunks
            n = max(n, chunks)
        embs, pooled = [], None
        for i in range(n):
            parts = []
            for tok, te, ids in ((self.tok1, self.te1, ids1), (self.tok2, self.te2, ids2)):
                c = ids[i * 75:(i + 1) * 75]
                pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                row = [tok.bos_token_id] + c + [tok.eos_token_id]
                row = row + [pad] * (77 - len(row))
                o = te(torch.tensor([row]), output_hidden_states=True)
                parts.append(o.hidden_states[-2])
                if te is self.te2 and i == 0:
                    pooled = o.text_embeds
            embs.append(torch.cat(parts, -1))
        return torch.cat(embs, 1), pooled, n

    # -------------------------------------------------------------- loras
    def set_loras(self, loras: list[dict]):
        """Activate LoRA files ({"path", "scale"}); others are disabled."""
        names = []
        for l in loras:
            name = os.path.splitext(os.path.basename(l["path"]))[0].replace(".", "_")
            if name not in self.loaded_loras:
                from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin as M
                sd, alphas = M.lora_state_dict(l["path"])
                M.load_lora_into_unet(sd, alphas, self.unet, adapter_name=name)
                self.loaded_loras[name] = l["path"]
            names.append((name, l.get("scale", 0.8)))
        if names:
            self.unet.set_adapters([n for n, _ in names], [s for _, s in names])
        elif self.loaded_loras:
            self.unet.disable_lora()

    # -------------------------------------------------------------- vae
    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> Image.Image:
        x = self.vae.decode(lat.float() / self.vae.config.scaling_factor).sample
        x = (x[0].clamp(-1, 1) + 1) / 2
        return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(img.convert("RGB"))).float().permute(2, 0, 1)[None] / 127.5 - 1
        return self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, prompt: str, negative: str, width: int, height: int, steps: int = 26, cfg: float = 6.0,
               seed: int = 0, controls: dict | None = None, control_end: float = 0.75,
               regions: list[tuple[str, np.ndarray]] | None = None, base_weight: float = 0.3,
               init_image: Image.Image | None = None, strength: float = 1.0,
               inpaint_mask: np.ndarray | None = None, references: list | None = None,
               style_fidelity: float = 0.5, progress: str = "") -> Image.Image:
        """controls: {"openpose": (PIL, scale), "depth": (PIL, scale), ...}
        regions: [(prompt, mask HxW float)] - masks need not sum to 1; leftover -> base prompt only.
        inpaint_mask: HxW float, 1 = regenerate (only with init_image).
        references: [Reference] identity anchors (character sheets), each restricted to its mask."""
        from diffusers import DPMSolverMultistepScheduler

        sched = DPMSolverMultistepScheduler.from_config(self.sched_config, use_karras_sigmas=True,
                                                        algorithm_type="dpmsolver++", solver_order=2,
                                                        final_sigmas_type="zero", timestep_spacing="trailing")
        sched.set_timesteps(steps)
        timesteps = sched.timesteps
        if init_image is not None and strength < 1.0:
            start = int(round(steps * (1 - strength)))
            timesteps = timesteps[start:]
            if hasattr(sched, "set_begin_index"):
                sched.set_begin_index(start)
        g = torch.Generator().manual_seed(seed)
        h8, w8 = height // 8, width // 8

        # prompts (all padded to the same number of 77-token chunks)
        lens = [self.encode(p)[2] for p in [prompt, negative] + [r[0] for r in (regions or [])]]
        n = max(lens)
        pos, pooled, _ = self.encode(prompt, n)
        neg, npooled, _ = self.encode(negative, n)
        ctx = torch.cat([neg, pos]).to(self.dtype)
        add_text = torch.cat([npooled, pooled]).to(self.dtype)
        tids = torch.tensor([[height, width, 0, 0, height, width]] * 2, dtype=self.dtype)
        added = {"text_embeds": add_text, "time_ids": tids}

        st = self.region
        st.cache.clear()
        if regions:
            st.contexts = [self.encode(p, n)[0].to(self.dtype) for p, _ in regions]
            ms = np.stack([np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w8, h8), Image.BILINEAR),
                                      np.float32) / 255 for _, m in regions])
            st.masks = torch.from_numpy(ms)
            st.base_weight = base_weight
            st.active = True
        else:
            st.active, st.contexts = False, []

        # controls
        cn_in = None
        if controls and self.controlnet is not None:
            imgs, modes, scales = [], [], []
            for kind, (im, scale) in controls.items():
                im = im.convert("RGB").resize((width, height), Image.BILINEAR)
                t = torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None] / 255
                imgs.append(torch.cat([t, t]).to(self.dtype))
                modes.append(CONTROL_MODES[kind])
                scales.append(float(scale))
            ctype = torch.zeros(self.controlnet.config.num_control_type)
            ctype[modes] = 1
            cn_in = dict(controlnet_cond=imgs, control_type=ctype[None].repeat(2, 1).to(self.dtype),
                         control_type_idx=modes, conditioning_scale=scales)

        # references: encode once; every step they are re-noised to t and run through the UNet
        refs = []
        rs = self.ref
        rs.cache.clear()
        rs.banks, rs.mode = [], "off"
        if references:
            ms = []
            for ref in references:
                rw, rh = _fit_area(ref.image.size, 0.62e6)
                rlat = self.encode_image(ref.image.convert("RGB").resize((rw, rh), Image.LANCZOS))
                remb, rpool, _ = self.encode(ref.prompt)
                rids = torch.tensor([[rh, rw, 0, 0, rh, rw]], dtype=self.dtype)
                refs.append((rlat, torch.randn(rlat.shape, generator=g), remb.to(self.dtype),
                             {"text_embeds": rpool.to(self.dtype), "time_ids": rids}))
                m = np.ones((height, width), np.float32) if ref.mask is None else ref.mask.astype(np.float32)
                ms.append(np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
                                     .resize((w8, h8), Image.BILINEAR), np.float32) / 255)
                rs.banks.append({})
            rs.masks = torch.from_numpy(np.stack(ms))
            rs.weights = [r.weight for r in references]
            rs.style_fidelity = style_fidelity

        # latents
        noise = torch.randn((1, 4, h8, w8), generator=g)
        if init_image is not None:
            init = self.encode_image(init_image.resize((width, height), Image.LANCZOS))
            lat = sched.add_noise(init, noise, timesteps[:1]) if strength < 1.0 else noise * sched.init_noise_sigma
        else:
            init = None
            lat = noise * sched.init_noise_sigma
        keep = None
        if inpaint_mask is not None and init is not None:
            mk = Image.fromarray((np.clip(inpaint_mask, 0, 1) * 255).astype(np.uint8)).resize((w8, h8), Image.BILINEAR)
            keep = 1 - torch.from_numpy(np.asarray(mk, np.float32) / 255)[None, None]

        t0 = time.time()
        total = len(timesteps)
        for i, t in enumerate(timesteps):
            if refs:
                st.active, was = False, st.active
                rs.mode = "write"
                for ri, (rlat, rnoise, remb, radd) in enumerate(refs):
                    rs.current = ri
                    rx = sched.add_noise(rlat, rnoise, t.reshape(1))
                    rx = sched.scale_model_input(rx, t).to(self.dtype)
                    self.unet(rx, t, encoder_hidden_states=remb, added_cond_kwargs=radd, return_dict=False)
                rs.mode = "read"
                st.active = was
            x = torch.cat([lat, lat])
            x = sched.scale_model_input(x, t).to(self.dtype)
            kw = {}
            if cn_in is not None and i < control_end * total:
                down, mid = self.controlnet(x, t, encoder_hidden_states=ctx, added_cond_kwargs=added,
                                            return_dict=False, **cn_in)
                kw = dict(down_block_additional_residuals=down, mid_block_additional_residual=mid)
            eps = self.unet(x, t, encoder_hidden_states=ctx, added_cond_kwargs=added, return_dict=False, **kw)[0]
            eu, ec = eps.float().chunk(2)
            e = eu + cfg * (ec - eu)
            # CFG rescale (keeps saturated anime colours from blowing out at higher cfg)
            std_c, std_e = ec.std(), e.std()
            e = 0.3 * (e * std_c / std_e) + 0.7 * e
            lat = sched.step(e, t, lat, return_dict=False)[0]
            if keep is not None:
                if i + 1 < total:
                    ref = sched.add_noise(init, noise, timesteps[i + 1:i + 2])
                else:
                    ref = init
                lat = lat * (1 - keep) + ref * keep
            el = time.time() - t0
            print(f"\r[{progress}] step {i + 1}/{total}  {el / (i + 1):.1f}s/it  eta {el / (i + 1) * (total - i - 1):.0f}s",
                  end="", flush=True)
        print(flush=True)
        st.active = False
        rs.mode, rs.banks = "off", []
        return self.decode(lat)
