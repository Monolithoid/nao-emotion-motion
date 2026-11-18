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
    lo, hi = cfg.guardian_limits[j]  # motor (factory) limits
    bump = np.sin(np.pi * raw.t / raw.duration) ** 2
    q[:, k] = q[0, k] + (lo - 0.2 - q[0, k]) * bump  # arm raised 0.2 rad beyond the limit
    raw = Trajectory(raw.joints, raw.t, q)
    assert "limit" in {v.code for v in g.validate(raw).violations}
    a = make_playable(cfg, raw)
    assert g.validate(a.traj).ok, g.validate(a.traj).summary()
    assert j in a.joints_changed and a.traj.q[:, k].min() >= lo + cfg.guardian.margin(j) - 1e-9
    # only the joint that went past its limit is changed
    assert a.joints_changed == [j]


def _human_like_gesture(cfg, seed=1, dur=3.9):
    """Quick hand-made-like gesture: min-jerk strokes of up to 0.8 rad in 0.4-0.6 s with pauses."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur, int(dur * 50) + 1)
    q0 = np.array(cfg.pose("start", cfg.arm_joints))
    q = np.tile(q0, (len(t), 1))

    def mj(u):
        u = np.clip(u, 0, 1)
        return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    for k, j in enumerate(cfg.arm_joints):
        lo, hi = cfg.limits[j]
        cur, ts = q0[k], 0.2
        while ts < dur - 0.5:
            d = rng.uniform(0.4, 0.6)
            tgt = cur + np.clip(rng.uniform(lo + 0.15, hi - 0.15) - cur, -0.8, 0.8)
            q[:, k] += (tgt - cur) * mj((t - ts) / d)
            cur, ts = tgt, ts + d + rng.uniform(0.05, 0.3)
        q[:, k] += (q0[k] - cur) * mj((t - (dur - 0.4)) / 0.4)
    return Trajectory(cfg.arm_joints, t, q)


def test_velocity_limits_follow_factory_speeds(cfg):
    from nao_motion.config import factory_velocity
    g, guard = cfg.guardian, cfg.raw["guardian"]
    frac = float(guard["velocity_fraction_of_factory"])
    speed = float(cfg.robot["execution"]["max_fraction_speed"])
    assert factory_velocity(guard, "LShoulderPitch") == factory_velocity(guard, "RShoulderPitch") > 8
    for j in cfg.arm_joints:
        f = factory_velocity(guard, j)
        assert g.velocity_limit(j) == pytest.approx(min(frac, speed) * f, abs=1e-3)
        assert g.velocity_sources[j].startswith("factory")
    # an explicit value in guardian.yaml wins, the hands (not angles) keep theirs
    assert g.velocity_limit("HeadPitch") == pytest.approx(guard["per_joint_velocity_rad_s"]["HeadPitch"])
    assert g.velocity_limit("LHand") == pytest.approx(guard["per_joint_velocity_rad_s"]["LHand"])


def test_velocity_limits_never_exceed_what_setangles_allows(tmp_path, cfg):
    import yaml
    from nao_motion.config import factory_velocity
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    g = yaml.safe_load((d / "guardian.yaml").read_text())
    g["per_joint_velocity_rad_s"]["LWristYaw"] = 50.0  # above the motor
    (d / "guardian.yaml").write_text(yaml.safe_dump(g))
    r = yaml.safe_load((d / "robot.yaml").read_text())
    r["execution"]["max_fraction_speed"] = 0.3  # streaming slower than the guardian fraction (0.5)
    (d / "robot.yaml").write_text(yaml.safe_dump(r))
    c = load_config(d)
    for j in ("LWristYaw", "RShoulderPitch"):
        assert c.guardian.velocity_limit(j) == pytest.approx(0.3 * factory_velocity(g, j), abs=1e-3)
        assert "ceiling" in c.guardian.velocity_sources[j]


def test_local_slow_down_keeps_the_tempo_of_a_human_like_gesture(cfg):
    g = Guardian(cfg)
    raw = _human_like_gesture(cfg)
    assert not g.validate(raw).ok  # a few peaks above the limits
    uni = make_playable(cfg, raw, mode="uniform")
    loc = make_playable(cfg, raw, mode="local")
    for a in (uni, loc):
        assert g.validate(a.traj).ok, g.validate(a.traj).summary()
        assert a.max_shape_change_rad < 1e-6
        assert a.traj.duration == pytest.approx(raw.duration * a.time_scale, rel=0.02)
        assert np.allclose(a.traj.q[0], raw.q[0]) and np.allclose(a.traj.q[-1], raw.q[-1], atol=1e-6)
    # only the fast moments are slowed down: much closer to the duration of the take
    assert 1.0 < loc.time_scale < 1.25 and loc.time_scale < uni.time_scale - 0.3
    # same path: the visited postures stay on the original curve (dense comparison)
    dense = uni.traj.q
    dist = np.sqrt(((loc.traj.q[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert dist.max() < 0.05


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
