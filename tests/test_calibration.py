import shutil

import numpy as np
import pytest

from nao_motion.calibration import (Calibration, CalibrationError, activate, compute_calibration, convex_hull,
                                    hull_constraints, load_procedure, simplify_hull)
from nao_motion.config import load_config
from nao_motion.control.controller import ControllerError
from nao_motion.demo import simulate_calibration
from nao_motion.guardian import Guardian
from nao_motion.trajectory import Trajectory

from .conftest import make_wb


def test_hull_constraints_contain_points_and_shrink():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(500, 2))
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 1]  # disque
    hull = simplify_hull(convex_hull(pts), 8)
    cons = hull_constraints(hull, 0.0)
    for nx, ny, b in cons:  # sommets conservés dans l'enveloppe (simplification conservatrice)
        assert np.all(hull @ np.array([nx, ny]) <= b + 1e-9)
    shrunk = hull_constraints(hull, 0.05)
    assert all(b2 == pytest.approx(b1 - 0.05) for (_, _, b1), (_, _, b2) in zip(cons, shrunk))
    assert any(np.array([0.99, 0.0]) @ np.array([nx, ny]) > b for nx, ny, b in shrunk)


def test_compute_refuses_insufficient_exploration(cfg):
    proc = load_procedure(cfg.root)
    q = np.zeros((100, 2)) + np.linspace(0, 0.05, 100)[:, None]  # amplitude trop faible
    with pytest.raises(CalibrationError):
        compute_calibration({"head": {"joints": cfg.head_joints, "q": q}}, cfg.head_joints, proc)
    with pytest.raises(CalibrationError):
        compute_calibration({"head": {"joints": cfg.head_joints, "q": q[:5]}}, cfg.head_joints, proc)


def test_factory_limits_intersect_measured(cfg):
    proc = dict(load_procedure(cfg.root), pairs=[])
    q = np.stack([np.linspace(-2.5, 2.5, 200), np.linspace(-0.5, 0.4, 200)], axis=1)
    cal = compute_calibration({"head": {"joints": cfg.head_joints, "q": q}}, cfg.head_joints, proc,
                              factory={"HeadYaw": [-2.0, 2.0], "HeadPitch": [-0.6, 0.5]})
    assert cal.limits["HeadYaw"] == pytest.approx((-2.0 + 0.05, 2.0 - 0.05))
    assert cal.limits["HeadPitch"] == pytest.approx((-0.5 + 0.05, 0.4 - 0.05))


@pytest.fixture
def calibrated_cfg(tmp_path, cfg):
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    base = load_config(d)
    assert not base.calibrated
    wb = make_wb(tmp_path / "w", base)
    cal = simulate_calibration(wb)
    path = cal.save(d / "calibration" / "sim-test.yaml")
    with pytest.raises(CalibrationError):
        cal.save(path)  # jamais écrasée
    activate(d, path)
    return load_config(d), cal


def test_active_calibration_drives_guardian(calibrated_cfg):
    c, cal = calibrated_cfg
    assert c.calibrated and c.poses_calibrated and c.calibration_version == cal.version
    assert c.limits["HeadYaw"] == pytest.approx(tuple(cal.limits["HeadYaw"]))
    assert any(k.name.startswith("calib:") for k in c.guardian.coupled_constraints)
    g = Guardian(c)
    q = np.array(c.pose("start", c.all_joints))
    assert g.validate(Trajectory(c.all_joints, [0, 1], [q, q]), check_duration=False).ok


def test_real_robot_requires_matching_calibration(tmp_path, cfg, calibrated_cfg):
    wb = make_wb(tmp_path, cfg)
    wb.adapter.is_real = True
    with pytest.raises(ControllerError, match="calibration"):
        wb.ctrl.arm(confirm=True)
    c, _ = calibrated_cfg
    wb2 = make_wb(tmp_path / "b", c)
    wb2.adapter.is_real = True
    wb2.ctrl.arm(confirm=True)  # calibration faite sur « sim », robot « sim »
    wb2.ctrl.disarm()
    wb2.adapter.capabilities = lambda: {"robot_name": "autre-nao"}
    with pytest.raises(ControllerError, match="autre-nao"):
        wb2.ctrl.arm(confirm=True)


def test_calibration_file_validation(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("schema_version: 1\nkind: autre\n")
    with pytest.raises(CalibrationError):
        Calibration.load(p)
