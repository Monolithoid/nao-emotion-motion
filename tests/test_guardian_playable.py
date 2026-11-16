"""The guardian protects playback, never training: gestures are made playable before the
check, and a calibration can be extended with the recorded takes."""
import shutil

import numpy as np
import pytest

from nao_motion.calibration import Calibration, CalibrationError
from nao_motion.config import load_config
from nao_motion.guardian import Guardian
from nao_motion.guardian.playable import make_playable
from nao_motion.trajectory import Trajectory

from .conftest import make_wb


def _fast_arm_gesture(cfg, amp=0.3, freq=2.0, dur=1.5):
    t = np.linspace(0, dur, int(dur * 50) + 1)
    q0 = np.array(cfg.pose("start", cfg.arm_joints))
    w = np.sin(np.pi * t / dur) ** 2 * np.sin(2 * np.pi * freq * t)  # starts and ends at rest
    return Trajectory(cfg.arm_joints, t, q0 + amp * w[:, None])


def test_make_playable_slows_down_without_changing_the_path(cfg):
    g = Guardian(cfg)
    raw = _fast_arm_gesture(cfg)
    rep = g.validate(raw)
    assert not rep.ok and {v.code for v in rep.violations} & {"velocity", "acceleration"}
    a = make_playable(cfg, raw)
    assert g.validate(a.traj).ok, g.validate(a.traj).summary()
    assert a.time_scale > 1 and a.traj.duration == pytest.approx(raw.duration * a.time_scale, rel=0.02)
    assert a.max_shape_change_rad < 1e-6  # inside the limits: same path, only slower
    assert np.allclose(a.traj.q[0], raw.q[0]) and np.allclose(a.traj.q[-1], raw.q[-1], atol=1e-6)


def test_make_playable_keeps_an_acceptable_gesture(cfg):
    g = Guardian(cfg)
    raw = _fast_arm_gesture(cfg, amp=0.05, freq=0.5, dur=3.0)
    assert g.validate(raw).ok
    a = make_playable(cfg, raw)
    assert not a.changed and g.validate(a.traj).ok


def test_make_playable_brings_the_gesture_back_inside_the_limits(cfg):
    g = Guardian(cfg)
    j = "LShoulderPitch"
    k = cfg.arm_joints.index(j)
    raw = _fast_arm_gesture(cfg, amp=0.0, dur=3.0)
    q = raw.q.copy()
    lo, hi = cfg.limits[j]
    bump = np.sin(np.pi * raw.t / raw.duration) ** 2
    q[:, k] = q[0, k] + (hi + 0.2 - q[0, k]) * bump  # goes 0.2 rad beyond the limit
    raw = Trajectory(raw.joints, raw.t, q)
    assert "limit" in {v.code for v in g.validate(raw).violations}
    a = make_playable(cfg, raw)
    assert g.validate(a.traj).ok, g.validate(a.traj).summary()
    assert j in a.joints_changed and a.traj.q[:, k].max() <= hi - cfg.guardian.margin(j) + 1e-9


@pytest.fixture
def narrow_setup(tmp_path, cfg):
    """Real-looking calibration whose LShoulderRoll range is narrower than the takes."""
    from nao_motion.demo import collect_arms, simulate_calibration
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    base_cfg = load_config(d)
    wb = make_wb(tmp_path / "w", base_cfg)
    cal = simulate_calibration(wb)
    cal.robot = {"name": "nao", "adapter": "bridge", "backend": "naoqi"}
    for jj in cal.limits:  # factory limits, as read on a real robot (simulation: replacement limits)
        lo_f, hi_f = base_cfg.limits[jj]
        cal.measured[jj] = dict(cal.measured.get(jj) or {}, factory=[lo_f - 0.05, hi_f + 0.05])
    ids = collect_arms(wb, 1, (0.5, 1.0), log=lambda *a: None)
    for tid in ids:  # synthetic takes are never used: pretend they were recorded on the robot
        m = wb.store.load_meta(tid)
        m.synthetic = False
        m.status = "accepted"
        wb.store._write_meta(m)
    j = "LShoulderRoll"
    vals = np.concatenate([wb.store.load(t).segment_trajectory([j]).q[:, 0] for t in ids])
    lo, hi = cal.limits[j]
    cut = float(np.percentile(vals, 30))  # most take samples now lie below the calibrated range
    cal.limits[j] = (max(lo, cut), hi)
    cal.measured[j] = dict(cal.measured[j], min=cal.limits[j][0] - cal.margin_rad)
    path = cal.save(d / "calibration" / "narrow.yaml")
    return d, wb, path, j, vals


def test_extend_calibration_with_takes(narrow_setup):
    from nao_motion.calibration_extend import extend_with_takes
    d, wb, path, j, vals = narrow_setup
    c = load_config(d, calibration_file=path)
    new, pv = extend_with_takes(c, wb.store, path, speeds=True)
    base = Calibration.load(path)
    assert pv["n_takes"] >= 2 and new.version != base.version
    assert new.limits[j][0] < base.limits[j][0]  # extended towards the takes
    fac = base.measured[j]["factory"]
    assert new.limits[j][0] >= fac[0] + base.margin_rad - 1e-9  # factory stays the ceiling
    for k, (lo, hi) in new.limits.items():
        assert lo <= base.limits[k][0] + 1e-9 and hi >= base.limits[k][1] - 1e-9  # never narrower
    row = next(r for r in pv["joints"] if r["joint"] == j)
    assert row["extended"]
    assert pv["kinematics"] and new.kinematics
    # saved as a NEW file, loadable, and its speed limits drive the guardian once active
    p2 = new.save(d / "calibration" / f"{new.version}.yaml")
    c2 = load_config(d, calibration_file=p2)
    for jj, v in new.kinematics["velocity_rad_s"].items():
        assert c2.guardian.velocity_limit(jj) == pytest.approx(v)
        assert v >= c.guardian.velocity_limit(jj) - 1e-9
    assert c2.limits[j][0] == pytest.approx(new.limits[j][0])


def test_extend_refuses_simulated_calibration(tmp_path, cfg):
    from nao_motion.calibration_extend import extend_with_takes
    from nao_motion.demo import simulate_calibration
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    wb = make_wb(tmp_path / "w", load_config(d))
    path = simulate_calibration(wb).save(d / "calibration" / "sim.yaml")
    with pytest.raises(CalibrationError):
        extend_with_takes(load_config(d), wb.store, path)


def test_extend_api_preview_then_save(narrow_setup):
    d, wb, path, j, _ = narrow_setup
    wb.cfg = load_config(d)
    pv = wb.extend_calibration(path.name, speeds=False)
    assert "saved" not in pv and not pv["kinematics"]
    before = set(p.name for p in (d / "calibration").glob("*.yaml"))
    out = wb.extend_calibration(path.name, speeds=False, save=True)
    after = set(p.name for p in (d / "calibration").glob("*.yaml"))
    assert after - before == {out["saved"]}
    assert not (d / "calibration" / "ACTIVE").exists()  # nothing activated


def test_training_report_guardian_is_information_only(trained):
    """The guardian never filters takes: it only reports, on generated gestures and on the takes."""
    import json
    from pathlib import Path
    m = json.loads((Path(trained["arm_run"]) / "metrics.json").read_text())["metrics"]
    assert {"accepted", "raw_accepted", "total", "median_slow_down"} <= set(m["guardian"])
    gt = m["guardian_takes"]
    assert gt["total"] == len(set(json.loads((Path(trained["arm_run"]) / "split.json").read_text())["train"]) |
                              set(json.loads((Path(trained["arm_run"]) / "split.json").read_text())["val"]))
    assert m["n_examples"] >= gt["total"]  # every take is trained on, whatever the guardian says
    assert gt["median_excess_rad"] is not None and gt["max_excess_rad"] >= gt["median_excess_rad"]
