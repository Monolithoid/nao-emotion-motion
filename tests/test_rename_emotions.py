"""Migration of the former French emotion ids to the English ids."""
import json

from nao_motion.cli import main
from nao_motion.data.rename import rename_emotions


def _take(root, sub, take_id, emotion, **extra):
    d = root / sub / take_id
    d.mkdir(parents=True)
    meta = dict({"take_id": take_id, "emotion": emotion, "intensity": 0.5}, **extra)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "samples.npz").write_bytes(b"untouched")
    return d


def _meta(d):
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def test_rename_takes_trash_and_generated_mix(tmp_path):
    a = _take(tmp_path, "takes", "arms-1", "joie")
    b = _take(tmp_path, "takes", "head-1", "surprise",
              arm_generation={"emotion": {"colere": 0.6, "peur": 0.4}, "arm_checkpoint_digest": "x"})
    c = _take(tmp_path, "trash/takes", "arms-2", "calme")
    r = rename_emotions(tmp_path, known=["joy", "sadness", "anger", "fear", "surprise", "calm"])
    assert r["takes_seen"] == 3 and r["takes_changed"] == 3 and not r["unknown"]
    assert r["renamed"] == {"joie -> joy": 1, "calme -> calm": 1}
    assert _meta(a)["emotion"] == "joy"
    assert _meta(b)["emotion"] == "surprise"
    assert _meta(b)["arm_generation"]["emotion"] == {"anger": 0.6, "fear": 0.4}
    assert _meta(c)["emotion"] == "calm"
    assert (a / "samples.npz").read_bytes() == b"untouched"
    again = rename_emotions(tmp_path)
    assert again["takes_changed"] == 0


def test_dry_run_changes_nothing_and_unknown_is_reported(tmp_path):
    a = _take(tmp_path, "takes", "arms-1", "tristesse")
    _take(tmp_path, "takes", "arms-2", "nostalgie")
    r = rename_emotions(tmp_path, known=["sadness"], dry_run=True)
    assert r["takes_changed"] == 1 and _meta(a)["emotion"] == "tristesse"
    assert r["unknown"] == {"nostalgie": ["arms-2"]}


def test_cli(tmp_path, capsys):
    a = _take(tmp_path, "takes", "arms-1", "peur")
    assert main(["--data", str(tmp_path), "rename-emotions"]) == 0
    assert _meta(a)["emotion"] == "fear"
    assert '"takes_changed": 1' in capsys.readouterr().out
