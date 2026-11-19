"""Calibration conduite depuis l'interface (onglet Calibration)."""
import shutil

import numpy as np
import pytest

from nao_motion.calibration import Calibration, CalibrationError, active_calibration_path
from nao_motion.config import load_config
from nao_motion.control.controller import ControllerError
from nao_motion.ui.server import Api

from .conftest import make_wb


@pytest.fixture
def ui(tmp_path, cfg):
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    wb = make_wb(tmp_path / "w", load_config(d))
    return wb, Api(wb)


def _full_procedure(wb, api):
    cal = wb.require_calibration()
    for g in wb.cfg.groups:
        cal.sweep_start(g, max_duration_s=20, background=False)
        for pose in ("rest", "start"):
            api.post_calibration_capture({"pose": pose, "group": g})
    return api.post_calibration_compute({"version": "ui-test"})


def test_wizard_end_to_end_in_simulation(ui):
    wb, api = ui
    assert api.get_calibration({}) == {"active": False}
    st = api.post_calibration_open({})
    assert st["active"] and st["sim"] and st["swept"] == [] and len(st["missing_poses"]) == 6
    assert set(st["angles"]) == set(wb.cfg.all_joints)
    st = _full_procedure(wb, api)
    assert st["result"]["version"] == "ui-test" and st["result"]["coupled"] > 0
    assert sorted(st["swept"]) == sorted(wb.cfg.groups) and st["missing_poses"] == []
    pair = next(p for p in st["pairs"] if p["joints"] == ["HeadYaw", "HeadPitch"])
    assert pair["n"] > 50 and len(pair["points"]) <= 400 and len(pair["hull"]) >= 3
    for j, lim in st["result"]["limits"].items():
        assert st["ranges"][j]["min"] < lim[0] < lim[1] < st["ranges"][j]["max"]
    st = api.post_calibration_save({})
    path = wb.cfg.root / "calibration" / st["saved"]
    assert Calibration.load(path).version == "ui-test"
    with pytest.raises(CalibrationError, match="simulée"):  # simulation: never activated
        api.post_calibration_activate({"confirm": True})
    assert active_calibration_path(wb.cfg.root) is None


def test_activation_reloads_guardian(ui):
    wb, api = ui
    api.post_calibration_open({})
    _full_procedure(wb, api)
    api.post_calibration_save({})
    cal = wb.calibration
    cal.sim = False  # as on a real robot
    with pytest.raises(ValueError):
        api.post_calibration_activate({})  # without confirmation
    st = api.post_calibration_activate({"confirm": True})
    assert st["activated"]
    assert wb.cfg.calibrated and wb.cfg.calibration_version == "ui-test"
    assert wb.guardian.cfg is wb.cfg and wb.ctrl.cfg is wb.cfg and not wb.ctrl.armed


def test_compute_refused_before_every_step(ui):
    wb, api = ui
    api.post_calibration_open({})
    with pytest.raises(CalibrationError):
        api.post_calibration_compute({})
    with pytest.raises(CalibrationError):
        api.post_calibration_save({})


def test_open_calibration_blocks_motion_and_disarms(ui):
    wb, api = ui
    wb.ctrl.arm(confirm=True)
    api.post_calibration_open({})
    assert not wb.ctrl.armed
    with pytest.raises(ControllerError, match="Calibration ouverte"):
        api.post_arm({"armed": True, "confirm": True})
    with pytest.raises(ControllerError, match="Calibration ouverte"):
        wb.prepare_arm_take("joie", 0.5, confirm=True)
    with pytest.raises(ValueError, match="Calibration ouverte"):
        api.post_connect({"mode": "sim"})
    api.post_calibration_close({})
    assert api.get_calibration({}) == {"active": False}
    api.post_arm({"armed": True, "confirm": True})


def test_no_motion_command_during_calibration(ui):
    """Only "stiffen in place" targets (measured position) are sent."""
    wb, api = ui
    api.post_calibration_open({})
    sent, orig = [], wb.adapter._send

    def spy(joints, angles, speed):  # target compared with the position measured at the same moment
        sent.append(np.allclose(angles, [wb.adapter.q[j] for j in joints], atol=1e-9))
        orig(joints, angles, speed)
    wb.adapter._send = spy
    _full_procedure(wb, api)
    api.post_calibration_hold({"group": "head"})
    assert sent and all(sent)
    # no movement executed by the controller, no "rest" pose imposed on closing
    q_before = dict(wb.adapter.q)
    api.post_calibration_close({})
    assert dict(wb.adapter.q) == q_before


def test_real_robot_release_requires_confirmation(ui):
    wb, api = ui
    api.post_calibration_open({})
    wb.adapter.is_real = True
    with pytest.raises(CalibrationError, match="tenu"):
        api.post_calibration_release({"group": "head"})
    with pytest.raises(CalibrationError, match="tenu"):
        api.post_calibration_sweep_start({"group": "head"})
    api.post_calibration_release({"group": "head", "confirm": True})
    assert all(wb.adapter.roles[j] == "guided" for j in wb.cfg.groups["head"])
    api.post_calibration_hold({"group": "head"})
    assert all(wb.adapter.roles[j] == "driven" for j in wb.cfg.groups["head"])


def test_factory_limits_shown_during_calibration(ui, monkeypatch):
    """Factory limits read once on opening (read-only) and shown
    during exploration; point cloud frame = factory limits."""
    wb, api = ui
    st = api.post_calibration_open({})
    assert st["factory"] is None  # simulation: no getLimits
    fac = {j: [lo - 0.1, hi + 0.1, 5.0, 1.0] for j, (lo, hi) in wb.cfg.limits.items()}
    monkeypatch.setattr(wb.adapter, "factory_limits", lambda joints: {j: fac[j] for j in joints}, raising=False)
    st = api.post_calibration_open({})
    assert st["factory"]["HeadPitch"] == [round(fac["HeadPitch"][0], 4), round(fac["HeadPitch"][1], 4)]
    wb.require_calibration().sweep_start("head", max_duration_s=5, background=False)
    st = api.get_calibration({})
    pair = next(p for p in st["pairs"] if p["joints"] == ["HeadYaw", "HeadPitch"])
    assert pair["limits"][1] == fac["HeadPitch"][:2]


def test_pairs_show_zone_and_poses_and_sweep_accumulates(ui):
    """Pair plots: retained zone, captured poses (start inside / outside the zone),
    current position; a new exploration adds to the previous one."""
    wb, api = ui
    api.post_calibration_open({})
    cal = wb.require_calibration()
    cal.sweep_start("head", max_duration_s=10, background=False)
    n1 = len(cal.session.sweeps["head"]["q"])
    cal.sweep_start("head", max_duration_s=10, background=False)
    assert len(cal.session.sweeps["head"]["q"]) > n1
    api.post_calibration_capture({"pose": "start", "group": "head"})
    st = api.get_calibration({})
    pair = next(p for p in st["pairs"] if p["joints"] == ["HeadYaw", "HeadPitch"])
    assert len(pair["zone"]) >= 3 and pair["current"] is not None
    assert "start" in pair["poses"] and isinstance(pair["poses"]["start"]["inside"], bool)
    # a start pose outside the zone is flagged even before computing
    cal.session.poses["start"]["head"] = {"HeadYaw": 5.0, "HeadPitch": 5.0}
    pair = next(p for p in api.get_calibration({})["pairs"] if p["joints"] == ["HeadYaw", "HeadPitch"])
    assert pair["poses"]["start"]["inside"] is False
    api.post_calibration_sweep_clear({"group": "head"})
    assert "head" not in api.get_calibration({})["swept"]


# -- saved calibrations: list, later activation, progress restore ----
from pathlib import Path

import yaml

from nao_motion.calibration_wizard import CalibrationWizard

OLD_FILE = Path(__file__).parent / "data" / "calibration-v1-5104ddf.yaml"


def _as_real(src: Path, dst: Path, version: str):
    """Copy of a simulated calibration presented as coming from the real robot (for activation)."""
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    d["robot"] = {"robot_name": "nao", "name": "nao", "adapter": "bridge"}
    d["version"] = version
    dst.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dst


def test_old_calibration_files_still_load_and_activate(ui):
    """A file produced by an earlier version (before hands, distance, etc.) remains
    usable: listed as activatable and activatable from the tab, without an open session."""
    wb, api = ui
    d = wb.cfg.root / "calibration"
    _as_real(OLD_FILE, d / "ancienne.yaml", "ancienne")
    (d / "sim.yaml").write_text(OLD_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (d / "casse.yaml").write_text("kind: autre\n", encoding="utf-8")
    files = {f["file"]: f for f in api.get_calibrations({})["files"]}
    assert files["ancienne.yaml"]["activable"] and not files["ancienne.yaml"]["active"]
    assert files["sim.yaml"]["simulated"]
    assert not files["casse.yaml"]["valid"] and files["casse.yaml"]["reason"]
    assert wb.calibration is None  # aucune calibration ouverte
    res = api.post_calibrations_activate({"file": "ancienne.yaml"})
    assert res["active"] == "ancienne" and wb.cfg.calibrated and not wb.ctrl.armed
    assert next(f for f in res["files"] if f["file"] == "ancienne.yaml")["active"]
    assert wb.guardian.cfg is wb.cfg


def test_simulated_or_incompatible_calibration_refused_on_real_robot(ui, monkeypatch):
    wb, api = ui
    d = wb.cfg.root / "calibration"
    (d / "sim.yaml").write_text(OLD_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    bad = yaml.safe_load(OLD_FILE.read_text(encoding="utf-8"))
    bad["robot"] = {"name": "nao", "adapter": "bridge"}
    del bad["limits"]["HeadYaw"]
    (d / "incomplete.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(type(wb.adapter), "is_real", True, raising=False)
    files = {f["file"]: f for f in api.get_calibrations({})["files"]}
    assert not files["sim.yaml"]["activable"] and "simulée" in files["sim.yaml"]["reason"]
    assert not files["incomplete.yaml"]["activable"] and "HeadYaw" in files["incomplete.yaml"]["reason"]
    for f in ("sim.yaml", "incomplete.yaml"):
        with pytest.raises(CalibrationError):
            api.post_calibrations_activate({"file": f})
    assert active_calibration_path(wb.cfg.root) is None and not wb.cfg.calibrated


def test_failed_reload_restores_previous_activation(ui, monkeypatch):
    wb, api = ui
    d = wb.cfg.root / "calibration"
    _as_real(OLD_FILE, d / "a.yaml", "a")
    _as_real(OLD_FILE, d / "b.yaml", "b")
    api.post_calibrations_activate({"file": "a.yaml"})

    def boom():
        raise RuntimeError("rechargement impossible")
    monkeypatch.setattr(wb, "reload_config", boom)
    with pytest.raises(RuntimeError):
        api.post_calibrations_activate({"file": "b.yaml"})
    assert active_calibration_path(wb.cfg.root).name == "a.yaml"


def test_progress_survives_close_and_can_be_discarded(ui, monkeypatch):
    wb, api = ui
    monkeypatch.setattr(CalibrationWizard, "PERSIST_SIM", True)
    api.post_calibration_open({})
    cal = wb.require_calibration()
    cal.sweep_start("left_arm", max_duration_s=20, background=False)
    api.post_calibration_capture({"pose": "rest", "group": "left_arm"})
    n = len(cal.session.sweeps["left_arm"]["q"])
    assert cal.draft_path.exists()
    api.post_calibration_close({})
    st = api.post_calibration_open({})  # new session (or server restarted)
    assert st["swept"] == ["left_arm"] and st["restored"] == ["left_arm"]
    assert "left_arm" in st["poses"]["rest"]
    assert len(wb.calibration.session.sweeps["left_arm"]["q"]) == n
    st = api.post_calibration_discard({})
    assert st["swept"] == [] and not wb.calibration.draft_path.exists()


def test_reuse_groups_from_saved_calibration(ui, monkeypatch):
    """Keep the arms of a saved calibration and redo only the head."""
    wb, api = ui
    d = wb.cfg.root / "calibration"
    old = Calibration.load(_as_real(OLD_FILE, d / "bras-ok.yaml", "bras-ok"))
    api.post_calibration_open({})
    cal = wb.require_calibration()
    for g in ("left_arm", "right_arm"):
        api.post_calibration_import({"file": "bras-ok.yaml", "group": g})
    st = api.get_calibration({})
    assert set(st["imported"]) == {"left_arm", "right_arm"}
    assert st["missing_poses"] == ["rest/head", "start/head"]  # arm poses imported too
    cal.sweep_start("head", max_duration_s=20, background=False)
    for pose in ("rest", "start"):
        api.post_calibration_capture({"pose": pose, "group": "head"})
    st = api.post_calibration_compute({"version": "tete-refaite"})
    res = cal.result
    for j in wb.cfg.arm_joints:
        assert res.limits[j] == pytest.approx(old.limits[j])
        assert res.measured[j]["imported_from"] == "bras-ok"
    arm_cc = [c for c in old.coupled if set(c["terms"]) <= set(wb.cfg.arm_joints)]
    assert all(c in res.coupled for c in arm_cc) and len(res.coupled) > len(arm_cc)
    assert "repris" in res.notes
    # exploring an imported group cancels the import
    cal.sweep_start("left_arm", max_duration_s=20, background=False)
    assert "left_arm" not in api.get_calibration({})["imported"]


def test_change_only_rest_pose_of_saved_calibration(ui, monkeypatch):
    """Import a saved calibration, recapture only one group's rest pose and
    save: everything else (limits, hulls, start, margin, velocities) is kept."""
    wb, api = ui
    d = wb.cfg.root / "calibration"
    src = _as_real(OLD_FILE, d / "base.yaml", "base")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["kinematics"] = {"source": "takes p100", "velocity_rad_s": {"LShoulderPitch": 3.1, "LHand": 2.0},
                         "acceleration_rad_s2": {"LShoulderPitch": 20.0}}
    src.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    old = Calibration.load(src)
    api.post_calibration_open({})
    cal = wb.require_calibration()
    st = api.post_calibration_import_all({"file": "base.yaml"})
    assert set(st["imported"]) == set(wb.cfg.groups) and st["missing_poses"] == []
    # new left-arm rest: placed by hand, then captured
    new_rest = {j: v + 0.05 for j, v in old.poses["rest"]["left_arm"].items()}
    for j, v in new_rest.items():
        wb.adapter.q[j] = wb.adapter.target[j] = v
    monkeypatch.setattr(cal, "sim", False)  # otherwise the simulation "places" the placeholder pose
    api.post_calibration_capture({"pose": "rest", "group": "left_arm"})
    api.post_calibration_compute({"version": "base-repos"})
    api.post_calibration_save({})
    new = Calibration.load(cal.saved_path)
    assert new.poses["rest"]["left_arm"] == pytest.approx(new_rest)
    for n in ("rest", "start"):
        for g in wb.cfg.groups:
            if (n, g) != ("rest", "left_arm"):
                assert new.poses[n][g] == pytest.approx(old.poses[n][g])
    assert {j: list(v) for j, v in new.limits.items()} == pytest.approx({j: list(v) for j, v in old.limits.items()})
    assert sorted(c["name"] for c in new.coupled) == sorted(c["name"] for c in old.coupled)
    assert new.margin_rad == old.margin_rad
    assert new.kinematics["velocity_rad_s"] == {"LShoulderPitch": 3.1, "LHand": 2.0}
    assert new.kinematics["acceleration_rad_s2"] == {"LShoulderPitch": 20.0}
    assert "repos recapturée" in new.notes and src.exists()
    assert Calibration.load(src).poses == old.poses  # the original is not touched
