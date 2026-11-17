"""Trash of takes and training runs: reversible move, export and lists ignore it, active model."""
import json

import pytest

from nao_motion.data import trash
from nao_motion.data.dataset import export_dataset, select_takes


def _take(root, take_id, kind="arms", status="accepted", **extra):
    d = root / "takes" / take_id
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"take_id": take_id, "kind": kind, "status": status, **extra}))
    (d / "samples.npz").write_bytes(b"x" * 2048)
    return d


def _run(runs, name, latest=False):
    d = runs / name
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"m" * 100)
    (d / "metrics.json").write_text(json.dumps({"metrics": {}}))
    if latest:
        (runs / f"latest_{name.partition('-')[0]}.txt").write_text(name)
    return d


def test_takes_trash_and_restore(tmp_path):
    _take(tmp_path, "a1")
    _take(tmp_path, "a2", status="rejected")
    out = trash.trash_takes(tmp_path, ["a2"])
    assert out["trashed"] == ["a2"] and out["bytes"] > 2048
    assert not (tmp_path / "takes" / "a2").exists()
    assert (tmp_path / "trash" / "takes" / "a2" / "samples.npz").exists()
    listed = trash.list_trashed_takes(tmp_path)
    assert [t["take_id"] for t in listed] == ["a2"] and listed[0]["status"] == "rejected" and listed[0]["trashed_at"]
    trash.restore_takes(tmp_path, ["a2"])
    assert (tmp_path / "takes" / "a2" / "meta.json").exists()
    assert not (tmp_path / "takes" / "a2" / trash.STAMP).exists()
    assert trash.list_trashed_takes(tmp_path) == []
    with pytest.raises(trash.TrashError):
        trash.trash_takes(tmp_path, ["nope"])
    with pytest.raises(trash.TrashError):
        trash.trash_takes(tmp_path, ["../takes"])


def test_old_arm_take_kept_with_its_head_takes(tmp_path):
    _take(tmp_path, "arm")
    _take(tmp_path, "head", kind="head", arm_take_id="arm", arm_source="take")
    with pytest.raises(trash.TrashError, match="head"):
        trash.trash_takes(tmp_path, ["arm"])
    assert (tmp_path / "takes" / "arm").exists()
    trash.trash_takes(tmp_path, ["arm", "head"])
    with pytest.raises(trash.TrashError, match="arm"):
        trash.restore_takes(tmp_path, ["head"])
    trash.restore_takes(tmp_path, ["head", "arm"])  # arm takes go back first
    assert (tmp_path / "takes" / "head").exists() and (tmp_path / "takes" / "arm").exists()


def test_empty_is_final(tmp_path):
    for t in ("a1", "a2", "a3"):
        _take(tmp_path, t)
    trash.trash_takes(tmp_path, ["a1", "a2", "a3"])
    out = trash.empty(data_dir=tmp_path, items=["a1"])
    assert out["deleted"] == ["a1"]
    assert {t["take_id"] for t in trash.list_trashed_takes(tmp_path)} == {"a2", "a3"}
    trash.empty(data_dir=tmp_path)
    assert trash.list_trashed_takes(tmp_path) == []


def test_runs_trash_moves_active_model(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "arms-20261101-100000-000001")
    _run(runs, "arms-20261102-100000-000001")
    _run(runs, "arms-20261103-100000-000001", latest=True)
    _run(runs, "head-20261101-100000-000001", latest=True)
    out = trash.trash_runs(runs, ["arms-20261103-100000-000001", "arms-20261101-100000-000001"])
    assert out["active_changed"] == {"arms": "arms-20261102-100000-000001"}
    assert trash.active_runs(runs) == {"arms": "arms-20261102-100000-000001", "head": "head-20261101-100000-000001"}
    assert sorted(p.parent.name for p in runs.glob("*/metrics.json")) == \
        ["arms-20261102-100000-000001", "head-20261101-100000-000001"]
    listed = {r["run"]: r for r in trash.list_trashed_runs(runs)}
    assert listed["arms-20261103-100000-000001"]["was_active"] and not listed["arms-20261101-100000-000001"]["was_active"]
    # last model of a group: the group has no model any more
    out = trash.trash_runs(runs, ["head-20261101-100000-000001"])
    assert out["active_changed"] == {"head": None} and not (runs / "latest_head.txt").exists()
    # restore: a group without a model gets it back, the others keep their active model
    out = trash.restore_runs(runs, ["head-20261101-100000-000001", "arms-20261103-100000-000001"])
    assert out["active_changed"] == {"head": "head-20261101-100000-000001"}
    assert trash.active_runs(runs)["arms"] == "arms-20261102-100000-000001"
    trash.empty(runs_dir=runs)
    assert trash.list_trashed_runs(runs) == []


def test_api_trash_takes_and_runs(tmp_path, cfg):
    """Through the UI API, with real takes: trashed takes leave the list, progress and the export."""
    from nao_motion.demo import collect_synthetic
    from nao_motion.ui.server import Api
    from tests.conftest import make_wb
    wb = make_wb(tmp_path, cfg)
    arm_ids, _ = collect_synthetic(wb, reps=1, intensities=(0.5,), head_contexts=[], log=lambda *a: None)
    api = Api(wb)
    takes = api.get_takes({})
    assert takes and all(t["in_export"] is False for t in takes)  # simulated: never exported
    # pretend two takes are real robot takes: they are exported
    for t in arm_ids[:2]:
        p = wb.store.dir / t / "meta.json"
        d = json.loads(p.read_text())
        d["synthetic"] = False
        p.write_text(json.dumps(d))
    assert {t["take_id"] for t in api.get_takes({}) if t["in_export"]} == set(arm_ids[:2])
    n = len(takes)
    api.post_trash_takes({"ids": [arm_ids[0]]})
    assert len(api.get_takes({})) == n - 1
    assert [m.take_id for m in select_takes(wb.store.root)] == [arm_ids[1]]
    out = export_dataset(wb.store.root, tmp_path / "export", runs_dir=wb.runs_dir)
    assert out["takes"] == 1 and not (tmp_path / "export" / "takes" / arm_ids[0]).exists()
    assert not any("trash" in p.parts for p in (tmp_path / "export").rglob("*"))
    tr = api.get_trash({})
    assert [t["take_id"] for t in tr["takes"]] == [arm_ids[0]] and tr["takes_bytes"] > 0
    api.post_trash_restore({"takes": [arm_ids[0]]})
    assert len(api.get_takes({})) == n
    # runs: active flag in the list, emptying needs confirmation
    _run(wb.runs_dir, "arms-20261101-100000-000001")
    _run(wb.runs_dir, "arms-20261102-100000-000001", latest=True)
    runs = {r["run"]: r for r in api.get_train({})["runs"]}
    assert runs["arms-20261102-100000-000001"]["active"] and not runs["arms-20261101-100000-000001"]["active"]
    out = api.post_trash_runs({"runs": ["arms-20261102-100000-000001"]})
    assert out["active_changed"] == {"arms": "arms-20261101-100000-000001"}
    assert api.get_train({})["runs"][-1]["active"]
    with pytest.raises(ValueError):
        api.post_trash_empty({"what": "runs"})
    assert api.post_trash_empty({"what": "runs", "confirm": True})["deleted"] == ["arms-20261102-100000-000001"]
    api.post_trash_takes({"ids": [arm_ids[1]]})
    api.post_trash_empty({"what": "takes", "confirm": True})
    assert api.get_trash({})["takes"] == [] and not (wb.store.dir / arm_ids[1]).exists()
