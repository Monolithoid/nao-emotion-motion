import json

import numpy as np
import pytest

from nao_motion.calibration import Calibration
from nao_motion.data.dataset import DatasetError, export_dataset, import_dataset
from nao_motion.data.store import TakeStore
from tests.test_data import arm_take


@pytest.fixture
def setup(tmp_path, cfg):
    """Takes: 2 real accepted, 1 rejected, 1 simulated, 1 incomplete; one calibration; one model."""
    st = TakeStore(tmp_path / "data", cfg)
    ids = {}
    for name, kw, status in [("ok1", {}, "accepted"), ("ok2", {"emotion": "fear"}, "accepted"),
                             ("rej", {}, "rejected"), ("sim", {"synthetic": True}, "accepted"),
                             ("inc", {"annotations": {"notes": "coupée", "tags": ["incomplete"]}}, "accepted")]:
        meta, tr = arm_take(cfg, **kw)
        meta.status = status
        if name == "ok1":
            meta.calibration_version = "nao-20261104-100000"
        st.save(meta, tr)
        ids[name] = meta.take_id
    conf = tmp_path / "configs"
    (conf / "calibration").mkdir(parents=True)
    for f in ("emotions.yaml", "joints.yaml"):
        (conf / f).write_text((cfg.root / f).read_text(encoding="utf-8"), encoding="utf-8")
    Calibration(version="nao-20261104-100000", robot={"name": "nao"}, limits={"HeadYaw": (-1.0, 1.0)},
                measured={}).save(conf / "calibration" / "nao-20261104-100000.yaml")
    runs = tmp_path / "runs"
    (runs / "arms-1").mkdir(parents=True)
    (runs / "arms-1" / "model.pt").write_bytes(b"weights")
    (runs / "arms-1" / "metrics.json").write_text("{}")
    (runs / "latest_arms.txt").write_text("arms-1")
    return {"data": tmp_path / "data", "conf": conf, "runs": runs, "ids": ids, "store": st}


def test_export_selects_real_accepted_complete_takes(tmp_path, setup):
    out = tmp_path / "export"
    r = export_dataset(setup["data"], out, setup["conf"], setup["runs"], author="Test")
    ids = setup["ids"]
    exported = sorted(p.name for p in (out / "takes").iterdir())
    assert exported == sorted([ids["ok1"], ids["ok2"]])
    assert r["takes"] == 2 and r["added"] == 2
    # exact copies of the raw files
    src = setup["data"] / "takes" / ids["ok1"]
    assert (out / "takes" / ids["ok1"] / "samples.npz").read_bytes() == (src / "samples.npz").read_bytes()
    assert (out / "calibration" / "nao-20261104-100000.yaml").exists()
    assert (out / "configs" / "emotions.yaml").exists()
    assert (out / "models" / "arms" / "model.pt").read_bytes() == b"weights"
    man = json.loads((out / "dataset.json").read_text())
    assert man["license"] == "CC-BY-4.0" and man["stats"]["by_kind"] == {"arms": 2}
    assert "takes/" + ids["ok1"] + "/samples.npz" in man["files"]
    assert len((out / "index.csv").read_text().strip().splitlines()) == 3
    assert "CC-BY-4.0" in (out / "LICENSE").read_text() and "license: cc-by-4.0" in (out / "README.md").read_text()


def test_reexport_updates_and_keeps_git(tmp_path, setup):
    out = tmp_path / "export"
    (out / ".git").mkdir(parents=True)
    (out / ".git" / "HEAD").write_text("ref")
    export_dataset(setup["data"], out, setup["conf"], setup["runs"])
    ids, st = setup["ids"], setup["store"]
    st.update(ids["ok2"], status="rejected")
    st.update(ids["ok1"], notes="revue")
    r = export_dataset(setup["data"], out, setup["conf"], setup["runs"])
    assert r["removed"] == 1 and r["updated"] == 1
    assert not (out / "takes" / ids["ok2"]).exists()
    assert (out / ".git" / "HEAD").read_text() == "ref"


def test_export_refuses_foreign_folder_and_empty_selection(tmp_path, setup):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "x.txt").write_text("x")
    with pytest.raises(DatasetError):
        export_dataset(setup["data"], foreign, setup["conf"])
    with pytest.raises(DatasetError):
        export_dataset(setup["data"], tmp_path / "e", setup["conf"], statuses=("pending",))


def test_strip_notes(tmp_path, setup):
    setup["store"].update(setup["ids"]["ok1"], notes="privé")
    out = tmp_path / "export"
    export_dataset(setup["data"], out, setup["conf"], strip_notes=True)
    meta = json.loads((out / "takes" / setup["ids"]["ok1"] / "meta.json").read_text())
    assert meta["annotations"]["notes"] == ""


def test_import_roundtrip_never_overwrites(tmp_path, setup, cfg):
    out = tmp_path / "export"
    export_dataset(setup["data"], out, setup["conf"], setup["runs"])
    new_conf = tmp_path / "pc2" / "configs"
    (new_conf / "calibration").mkdir(parents=True)
    r = import_dataset(out, tmp_path / "pc2" / "data", new_conf, tmp_path / "pc2" / "runs")
    assert r["takes_added"] == 2 and r["calibrations_added"] == ["nao-20261104-100000.yaml"]
    assert r["models_added"] == ["arms"]
    assert (tmp_path / "pc2" / "runs" / "latest_arms.txt").read_text() == "arms-1"
    assert not (new_conf / "calibration" / "ACTIVE").exists()  # nothing is activated
    st2 = TakeStore(tmp_path / "pc2" / "data", cfg)
    tk = st2.load(setup["ids"]["ok1"])
    np.testing.assert_array_equal(tk.traj.q, setup["store"].load(setup["ids"]["ok1"]).traj.q)
    r2 = import_dataset(out, tmp_path / "pc2" / "data", new_conf, tmp_path / "pc2" / "runs")
    assert r2["takes_added"] == 0 and r2["takes_skipped"] == 2 and r2["calibrations_added"] == []


def test_import_refuses_corrupted_export(tmp_path, setup):
    out = tmp_path / "export"
    export_dataset(setup["data"], out, setup["conf"])
    p = next((out / "takes").glob("*/samples.npz"))
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(DatasetError):
        import_dataset(out, tmp_path / "pc2" / "data", tmp_path / "pc2" / "configs")
    assert not (tmp_path / "pc2" / "data" / "takes").exists()


def test_hand_edited_readme_neither_blocks_import_nor_is_overwritten(tmp_path, setup):
    out = tmp_path / "export"
    export_dataset(setup["data"], out, setup["conf"])
    (out / "README.md").write_text("# Mon jeu de données\n", encoding="utf-8")
    r = import_dataset(out, tmp_path / "pc2" / "data", tmp_path / "pc2" / "configs")
    assert r["takes_added"] == 2
    r2 = export_dataset(setup["data"], out, setup["conf"])
    assert r2.get("readme_kept") and (out / "README.md").read_text(encoding="utf-8") == "# Mon jeu de données\n"
    assert "## Content" in (out / "README.generated.md").read_text(encoding="utf-8")


def test_placeholder_calibration_is_not_reported_missing(tmp_path, setup):
    """Takes recorded before the first calibration cite joints.yaml's placeholder name, not a file."""
    out = tmp_path / "export"
    r = export_dataset(setup["data"], out, setup["conf"])
    import yaml
    ph = yaml.safe_load((setup["conf"] / "joints.yaml").read_text(encoding="utf-8"))["calibration"]["version"]
    assert ph not in r["missing_calibrations"]
    ok2 = json.loads((out / "takes" / setup["ids"]["ok2"] / "meta.json").read_text())
    assert ok2["calibration_version"] == ph  # test config: no active calibration
    assert r["uncalibrated_takes"] == 1 and "before the first calibration" in (out / "README.md").read_text()


def test_generator_skipped_without_blocking_the_export(tmp_path, setup):
    """No usable configuration / calibration / numpy model: dataset exported, generator left out."""
    out = tmp_path / "export"
    r = export_dataset(setup["data"], out, setup["conf"], setup["runs"])
    assert r["takes"] == 2 and r["generator"]["error"]
    assert not (out / "generator").exists()
    assert "Ready-to-use" not in (out / "README.md").read_text()


def test_generator_included_and_import_still_works(tmp_path, setup, cfg):
    import shutil
    from nao_motion.calibration import activate
    from nao_motion.config import load_config
    from nao_motion.demo import simulate_calibration
    from tests.conftest import make_wb
    from tests.test_gesture_export import _fake_npz, _load_lib

    conf = tmp_path / "full_configs"
    shutil.copytree(cfg.root, conf)
    (conf / "calibration" / "ACTIVE").unlink(missing_ok=True)
    path = simulate_calibration(make_wb(tmp_path / "w", load_config(conf))).save(conf / "calibration" / "gen.yaml")
    activate(conf, path, allow_simulated=True)
    c = load_config(conf)
    runs = tmp_path / "runs2"
    for k, comp in enumerate(("arms", "head")):
        _fake_npz(runs / f"{comp}-1" / "model_numpy.npz", c, comp, k)
        (runs / f"{comp}-1" / "model.pt").write_bytes(b"weights")
        (runs / f"latest_{comp}.txt").write_text(f"{comp}-1")

    out = tmp_path / "export"
    r = export_dataset(setup["data"], out, conf, runs)
    gen = r["generator"]
    assert "error" not in gen, gen
    assert gen["models"] == {"arms": "arms-1", "head": "head-1"} and gen["missing"] == ["hands"]
    assert gen["calibration_version"] == c.calibration_version
    lib = _load_lib(out / "generator" / "nao_gesture.py")
    g = lib.GestureGenerator.load(str(out / "generator" / "nao_gesture.json")).generate("joy", 0.7)
    assert g.joints
    man = json.loads((out / "dataset.json").read_text())
    assert "generator/nao_gesture.json" in man["files"] and man["generator"]["models"]["arms"] == "arms-1"
    card = (out / "README.md").read_text()
    assert "## Ready-to-use gesture generator" in card and "GestureGenerator.load('generator/nao_gesture.json')" in card

    # re-export with the same models: generator files untouched (no diff in the dataset repository)
    before = (out / "generator" / "nao_gesture.json").read_bytes()
    export_dataset(setup["data"], out, conf, runs)
    assert (out / "generator" / "nao_gesture.json").read_bytes() == before

    # import ignores generator/ and still checks the data files
    r2 = import_dataset(out, tmp_path / "pc2" / "data", tmp_path / "pc2" / "configs", tmp_path / "pc2" / "runs")
    assert r2["takes_added"] == 2 and sorted(r2["models_added"]) == ["arms", "head"]

    # --no-generator removes a previous generator/
    r3 = export_dataset(setup["data"], out, conf, runs, generator=False)
    assert r3["generator"] is None and not (out / "generator").exists()
