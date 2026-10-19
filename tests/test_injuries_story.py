"""Persistent injuries, story panels and page layout (no image model needed)."""
import json

import numpy as np
from PIL import Image

from animescene.characters import Cast
from animescene.injuries import anchor_from_point, draw, make_injury, surface_point
from animescene.scene import stage_scene
from animescene.story import panel_spec

STORY = "stories/alley_rin.json"


def staged_panel(story, pid, cast):
    panel = next(p for p in story["panels"] if p["id"] == pid)
    return stage_scene(panel_spec(story, panel), cast, size=(608, 416))


def load_cast(story, tmp_path):
    cast = Cast(tmp_path / "cast.json")
    used = []
    for cs in story["cast"]:
        used.append(cast.resolve(cs, used).palette)
    return cast


def test_anchor_round_trip_lands_on_the_same_surface_point():
    st = stage_scene({"characters": [{"name": "A", "seed": 1, "action": "stance"}]}, Cast(), size=(64, 64))
    a = st.scene.actor("A")
    for bone, t, ang in [("stomach", 0.4, 0), ("l_forearm", 0.5, 30), ("r_thigh", 0.3, -60)]:
        p, n = surface_point(a, bone, t, ang)
        b2, t2, a2 = anchor_from_point(a, p + n * 0.01)
        p2, _ = surface_point(a, b2, t2, a2)
        assert np.linalg.norm(p - p2) < 0.02, (bone, b2)


def test_injury_follows_the_body_into_any_pose():
    st1 = stage_scene({"characters": [{"name": "A", "seed": 1, "action": "stand"}]}, Cast(), size=(64, 64))
    st2 = stage_scene({"characters": [{"name": "A", "seed": 1, "action": "clutch_stomach"}]}, Cast(), size=(64, 64))
    a1, a2 = st1.scene.actor("A"), st2.scene.actor("A")
    for spot in ("l_cheek", "stomach", "r_forearm"):
        inj = make_injury("cut", spot=spot)
        p1, _ = surface_point(a1, inj["bone"], inj["t"], inj["angle"])
        p2, _ = surface_point(a2, inj["bone"], inj["t"], inj["angle"])
        # same place relative to the bone it is attached to
        bone = {"l_cheek": "head", "stomach": "spine", "r_forearm": "r_elbow"}[spot]
        d1 = np.linalg.norm(p1 - a1.joints[bone])
        d2 = np.linalg.norm(p2 - a2.joints[bone])
        assert abs(d1 - d2) < 0.03


def test_story_injuries_persist_across_panels(tmp_path):
    story = json.load(open(STORY))
    cast = load_cast(story, tmp_path)
    staged_panel(story, "p2", cast)  # Goro takes the palm strike -> stomach bruise
    staged_panel(story, "p3", cast)  # Rin takes the punch -> cut where the fist landed
    rin = Cast(tmp_path / "cast.json").get("Rin")
    goro = Cast(tmp_path / "cast.json").get("Goro")
    assert len(rin.injuries) == 1 and rin.injuries[0]["bone"] == "head"
    assert len(goro.injuries) == 1 and goro.injuries[0]["bone"] in ("stomach", "chest")
    # staging the same panel again does not duplicate the wound
    staged_panel(story, "p3", cast)
    assert len(Cast(tmp_path / "cast.json").get("Rin").injuries) == 1
    # and a later panel carries it
    st8 = staged_panel(story, "p8", Cast(tmp_path / "cast.json"))
    assert len(st8.scene.actor("Rin").char.injuries) == 1


def test_injury_drawing_is_deterministic_and_occlusion_aware(tmp_path):
    story = json.load(open(STORY))
    cast = load_cast(story, tmp_path)
    staged_panel(story, "p2", cast)
    st = staged_panel(story, "p8", Cast(tmp_path / "cast.json"))
    i = [a.name for a in st.scene.actors].index("Goro")
    goro = st.scene.actors[i]
    base = Image.new("RGB", st.size, (200, 180, 170))
    im1, d1 = draw(base, goro, st.cam, st.frame.depth, st.frame.ids, i)
    im2, _ = draw(base, goro, st.cam, st.frame.depth, st.frame.ids, i)
    assert np.array_equal(np.asarray(im1), np.asarray(im2))
    changed = np.abs(np.asarray(im1).astype(int) - np.asarray(base).astype(int)).sum(-1) > 0
    if d1:  # drawn only where Goro's own body is
        ys, xs = np.nonzero(changed)
        assert (st.frame.ids[ys, xs] == i).mean() > 0.6
    # if the camera can't see the spot, nothing is drawn
    other = Image.new("RGB", st.size, (0, 0, 0))
    ids = st.frame.ids.copy()
    ids[ids == i] = -1
    im3, d3 = draw(other, goro, st.cam, st.frame.depth, ids, i)
    assert d3 == []


def test_manga_page_layout(tmp_path):
    from animescene.manga import compose_pages
    story = json.load(open(STORY))
    panels = []
    for p in story["panels"][:4]:
        d = tmp_path / p["id"]
        d.mkdir()
        Image.new("RGB", (1216, 832), (90, 90, 120)).save(d / "final.png")
        (d / "meta.json").write_text(json.dumps({"size": [1216, 832], "subjects": [300, 100, 900, 800],
                                                  "actors": {c["name"]: {"head": [600, 300], "head_top": [600, 250]}
                                                             for c in p["characters"]},
                                                  "impacts": [[620, 420]]}))
        panels.append((p, d))
    pages = compose_pages(story, panels, tmp_path)
    assert pages and Image.open(pages[0]).width == 1600
