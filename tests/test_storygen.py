import json
from pathlib import Path

from animescene import storygen

ROOT = Path(__file__).resolve().parents[1]


def test_handwritten_story_is_valid():
    spec = json.loads((ROOT / "stories" / "alley_rin_fight.json").read_text())
    assert storygen.validate(spec, storygen.vocabulary()) == []


def test_errors_go_back_and_get_fixed(monkeypatch):
    good = json.loads((ROOT / "stories" / "alley_rin_fight.json").read_text())
    bad = json.loads(json.dumps(good))
    bad["beats"][1]["move"] = "flying_kick"
    bad["beats"][2]["panels"][0]["key"] = "smash"
    bad["beats"][3]["others"].pop("Jun")
    answers = iter(["```json\n" + json.dumps(bad) + "\n```", json.dumps(good)])
    prompts = []

    def fake(text, temperature=0.6):
        prompts.append(text)
        return next(answers)
    monkeypatch.setitem(storygen.BACKENDS, "fake", fake)
    spec = storygen.generate("Rin beats three thugs.", backend="fake", log=lambda s: None)
    assert spec == good
    assert "flying_kick" in prompts[1] and "smash" in prompts[1] and "missing from 'others'" in prompts[1]
