import numpy as np
import pytest

from nao_motion.data.representation import (bspline_basis, decode, fit_control_points, resample_phase,
                                            suggest_segment)
from nao_motion.data.store import TakeError, TakeMeta, TakeStore, new_id
from nao_motion.events import EventContext
from nao_motion.trajectory import Trajectory, TrajectoryError


def arm_take(cfg, n=100, duration=2.0, emotion="joie", intensity=0.5, **kw):
    t = np.linspace(0, duration, n)
    base = np.array(cfg.pose("start", cfg.arm_joints))
    q = base + 0.2 * np.sin(np.pi * t / duration)[:, None]
    meta = TakeMeta(take_id=new_id("arms"), session_id="s1", kind="arms", emotion=emotion, intensity=intensity,
                    joints=cfg.arm_joints, guided_joints=cfg.arm_joints, driven_joints=cfg.head_joints,
                    event=EventContext.none().to_dict(), calibration_version=cfg.calibration_version,
                    config_fingerprint=cfg.fingerprint(), adapter="test", sample_rate_hz=50,
                    segment=(0.0, duration), **kw)
    return meta, Trajectory(cfg.arm_joints, t, q)


def test_save_load_roundtrip_keeps_raw(tmp_path, cfg):
    st = TakeStore(tmp_path, cfg)
    meta, tr = arm_take(cfg)
    st.save(meta, tr)
    tk = st.load(meta.take_id)
    np.testing.assert_array_equal(tk.traj.t, tr.t)
    np.testing.assert_array_equal(tk.traj.q, tr.q)
    assert tk.meta.unit == "rad" and tk.meta.event_frame == "torso"
    st.update(meta.take_id, status="accepted", notes="belle prise", tags=["a"])
    tk2 = st.load(meta.take_id)
    np.testing.assert_array_equal(tk2.traj.q, tr.q)  # brut intact
    assert tk2.meta.status == "accepted" and tk2.meta.annotations["notes"] == "belle prise"


@pytest.mark.parametrize("mutate", [
    lambda m: setattr(m, "emotion", "inconnue"),
    lambda m: setattr(m, "intensity", 1.5),
    lambda m: setattr(m, "intention", "joie"),
    lambda m: setattr(m, "segment", (0.0, 99.0)),
    lambda m: setattr(m, "driven_joints", m.guided_joints),
    lambda m: setattr(m, "unit", "deg"),
])
def test_invalid_takes_rejected(tmp_path, cfg, mutate):
    st = TakeStore(tmp_path, cfg)
    meta, tr = arm_take(cfg)
    mutate(meta)
    with pytest.raises(TakeError):
        st.save(meta, tr)


def test_non_finite_and_non_monotonic_rejected():
    with pytest.raises(TrajectoryError):
        Trajectory(["HeadYaw"], np.array([0.0, 0.1, 0.1]), np.zeros((3, 1)))
    with pytest.raises(TrajectoryError):
        Trajectory(["HeadYaw"], [0, 1], [[0.0], [np.nan]])


def test_head_take_requires_existing_arm_take(tmp_path, cfg):
    st = TakeStore(tmp_path, cfg)
    j = cfg.all_joints
    t = np.linspace(0, 1, 20)
    q = np.tile(cfg.pose("start", j), (20, 1))
    meta = TakeMeta(take_id=new_id("head"), session_id="s", kind="head", emotion="joie", intensity=0.5, joints=j,
                    guided_joints=cfg.head_joints, driven_joints=cfg.arm_joints,
                    event=EventContext.none().to_dict(), calibration_version="x", config_fingerprint="x",
                    adapter="test", sample_rate_hz=50, segment=(0.0, 1.0), arm_take_id="arms-absente")
    with pytest.raises(TakeError):
        st.save(meta, Trajectory(j, t, q))


def test_resampling_different_durations_same_shape():
    for dur, n in ((1.3, 40), (4.2, 300)):
        t = np.sort(np.random.default_rng(0).uniform(0, dur, n))
        t[0], t[-1] = 0, dur
        q = np.sin(t)[:, None] * np.ones((1, 3))
        qu, d = resample_phase(t, q, 100)
        assert qu.shape == (100, 3) and abs(d - dur) < 1e-12


def test_bspline_partition_and_fit():
    u = np.linspace(0, 1, 101)
    B = bspline_basis(8, u)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
    q = np.stack([np.sin(3 * u), u ** 2], axis=-1)
    C = fit_control_points(q, 8)
    assert np.abs(B @ C - q).max() < 1e-3
    np.testing.assert_allclose(C[0], q[0])
    np.testing.assert_allclose(C[-1], q[-1])
    tr = decode(C, 2.0, ["a", "b"], 50)
    assert tr.t[0] == 0 and abs(tr.duration - 2.0) < 1e-12


def test_suggest_segment_trims_stillness():
    t = np.linspace(0, 4, 201)
    q = np.where((t > 1) & (t < 3), np.sin(np.pi * (t - 1) / 2), 0.0)[:, None]
    s0, s1 = suggest_segment(Trajectory(["x"], t, q))
    assert 0.8 < s0 < 1.1 and 2.9 < s1 < 3.2


def test_stats_counts(tmp_path, cfg):
    st = TakeStore(tmp_path, cfg)
    for i in (0.2, 0.25, 0.9):
        m, tr = arm_take(cfg, intensity=i)
        st.save(m, tr)
    row = st.stats()["table"]["arms"]["joie"]
    assert sum(sum(c.values()) for c in row.values()) == 3
