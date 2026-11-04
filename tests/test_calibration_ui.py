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
    with pytest.raises(CalibrationError, match="simulée"):  # simulation : jamais activée
        api.post_calibration_activate({"confirm": True})
    assert active_calibration_path(wb.cfg.root) is None


def test_activation_reloads_guardian(ui):
    wb, api = ui
    api.post_calibration_open({})
    _full_procedure(wb, api)
    api.post_calibration_save({})
    cal = wb.calibration
    cal.sim = False  # comme sur un robot réel
    with pytest.raises(ValueError):
        api.post_calibration_activate({})  # sans confirmation
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
    """Seules les consignes « raidir sur place » (position mesurée) sont envoyées."""
    wb, api = ui
    api.post_calibration_open({})
    sent, orig = [], wb.adapter._send

    def spy(joints, angles, speed):  # consigne comparée à la position mesurée au même instant
        sent.append(np.allclose(angles, [wb.adapter.q[j] for j in joints], atol=1e-9))
        orig(joints, angles, speed)
    wb.adapter._send = spy
    _full_procedure(wb, api)
    api.post_calibration_hold({"group": "head"})
    assert sent and all(sent)
    # aucun mouvement exécuté par le contrôleur, aucune pose « repos » imposée à la fermeture
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
    """Limites d'usine lues une fois à l'ouverture (lecture seule) et affichées
    pendant l'exploration ; cadre des nuages = limites d'usine."""
    wb, api = ui
    st = api.post_calibration_open({})
    assert st["factory"] is None  # simulation : pas de getLimits
    fac = {j: [lo - 0.1, hi + 0.1, 5.0, 1.0] for j, (lo, hi) in wb.cfg.limits.items()}
    monkeypatch.setattr(wb.adapter, "factory_limits", lambda joints: {j: fac[j] for j in joints}, raising=False)
    st = api.post_calibration_open({})
    assert st["factory"]["HeadPitch"] == [round(fac["HeadPitch"][0], 4), round(fac["HeadPitch"][1], 4)]
    wb.require_calibration().sweep_start("head", max_duration_s=5, background=False)
    st = api.get_calibration({})
    pair = next(p for p in st["pairs"] if p["joints"] == ["HeadYaw", "HeadPitch"])
    assert pair["limits"][1] == fac["HeadPitch"][:2]
