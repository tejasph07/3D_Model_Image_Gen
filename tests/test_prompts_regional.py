"""Prompt assembly and regional cross-attention maths (no model weights needed)."""
import numpy as np
import pytest

from animescene.characters import Cast
from animescene.prompts import build, count_tags
from animescene.scene import stage_scene

torch = pytest.importorskip("torch")


def test_prompts_describe_every_character_and_the_scene():
    st = stage_scene("scenes/shrine_clash.json", Cast(), size=(304, 208))
    pr = build(st)
    assert "2girls" in pr["global"] and "2boys" in pr["global"]
    assert "from below" in pr["global"]  # low_angle camera
    assert len(pr["regions"]) == 4
    by = {a.name: r for a, r in zip(st.scene.actors, pr["regions"])}
    assert "kicking" in by["Mira"] and "1girl" in by["Mira"]
    assert "knocked back" in by["Gorou"] and "midair" in by["Gorou"]
    assert "serious" not in by["Gorou"]  # a character in pain doesn't keep its neutral expression
    assert "blocking" in by["Ren"]
    for c, r in zip(st.chars, pr["regions"]):
        assert c.hair_color in r


def test_count_tags():
    from animescene.characters import design
    girls = [design(n, seed=i, gender="female") for i, n in enumerate("ABC")]
    assert count_tags(girls) == "3girls, multiple girls"


class FakeAttn(torch.nn.Module):
    def __init__(self, d=8, heads=2):
        super().__init__()
        self.heads = heads
        self.to_q = torch.nn.Linear(d, d, bias=False)
        self.to_k = torch.nn.Linear(d, d, bias=False)
        self.to_v = torch.nn.Linear(d, d, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(d, d), torch.nn.Identity()])


def test_regional_attention_routes_each_pixel_to_its_prompt():
    from animescene.diffusion import RegionalCrossAttn, RegionState, _attend
    torch.manual_seed(0)
    attn = FakeAttn()
    h, w = 4, 6
    hidden = torch.randn(2, h * w, 8)
    base, ctx_a, ctx_b = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    masks = torch.zeros(2, h, w)
    masks[0, :, :3] = 1  # left half -> region A
    masks[1, :, 3:] = 1  # right half -> region B
    st = RegionState(active=True, contexts=[ctx_a, ctx_b], masks=masks, base_weight=0.0)
    out = RegionalCrossAttn(st)(attn, hidden, torch.cat([torch.randn(1, 5, 8), base]))
    cond = hidden[1:]
    oa, ob = _attend(attn, cond, ctx_a), _attend(attn, cond, ctx_b)
    grid = out[1].reshape(h, w, 8)
    assert torch.allclose(grid[:, :3], oa[0].reshape(h, w, 8)[:, :3], atol=1e-5)
    assert torch.allclose(grid[:, 3:], ob[0].reshape(h, w, 8)[:, 3:], atol=1e-5)


def test_unowned_pixels_fall_back_to_the_base_prompt():
    from animescene.diffusion import RegionalCrossAttn, RegionState, _attend
    torch.manual_seed(1)
    attn = FakeAttn()
    hidden = torch.randn(2, 12, 8)
    base, ctx_a = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    masks = torch.zeros(1, 3, 4)  # region owns nothing
    st = RegionState(active=True, contexts=[ctx_a], masks=masks, base_weight=0.3)
    out = RegionalCrossAttn(st)(attn, hidden, torch.cat([torch.randn(1, 5, 8), base]))
    assert torch.allclose(out[1], _attend(attn, hidden[1:], base)[0], atol=1e-5)


def test_masks_are_resampled_to_every_attention_resolution():
    from animescene.diffusion import RegionState
    st = RegionState(masks=torch.rand(3, 104, 152))
    for k in range(3):
        hh, ww = int(np.ceil(104 / 2 ** k)), int(np.ceil(152 / 2 ** k))
        assert st.masks_for(hh * ww, torch.float32).shape == (3, hh * ww, 1)


def test_color_blocks_paint_each_character_in_its_own_colours():
    from animescene.characters import PALETTES
    from animescene.controls import _rgb
    from animescene.raster import PART_NAMES
    st = stage_scene("scenes/shrine_clash.json", Cast(), size=(608, 416))
    img = np.asarray(st.colors).astype(float)
    for i, c in enumerate(st.chars):
        torso = (st.frame.ids == i) & (st.frame.parts == PART_NAMES.index("torso"))
        mean = img[torso].mean(0)
        want = _rgb(PALETTES[c.palette][0])
        # shading scales brightness; hue (normalised colour) must match the palette
        cos = mean @ want / (np.linalg.norm(mean) * np.linalg.norm(want))
        assert cos > 0.95, (c.name, c.palette, mean, want)
