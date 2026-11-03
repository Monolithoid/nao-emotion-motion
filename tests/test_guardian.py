import numpy as np
import pytest

from nao_motion.guardian import Guardian, GuardianRejection
from nao_motion.trajectory import Trajectory


def smooth(cfg, joints, amp=0.2, dur=2.0, rate=50):
    t = np.linspace(0, dur, int(dur * rate) + 1)
    base = np.array(cfg.pose("start", joints))
    return Trajectory(joints, t, base + amp * np.sin(np.pi * t / dur)[:, None] ** 2)


def g_ok(cfg, tr):
    rep = Guardian(cfg).validate(tr, context_pose={"HeadYaw": 0.0})
    return rep.ok


def codes(rep):
    return {v.code for v in rep.violations}


def test_accepts_smooth(cfg):
    assert Guardian(cfg).validate(smooth(cfg, cfg.all_joints, amp=0.05)).ok


def test_rejects_limits(cfg):
    tr = smooth(cfg, ["HeadPitch"], amp=2.0)
    assert "limit" in codes(Guardian(cfg).validate(tr))


def test_safe_endpoints_unsafe_middle(cfg):
    """Extrémités sûres ne suffisent pas : le milieu du trajet est vérifié."""
    j = ["HeadYaw", "HeadPitch"]
    t = np.array([0.0, 1.0, 2.0])
    q = np.array([[0.0, 0.0], [0.0, 0.7], [0.0, 0.0]])  # HeadPitch hors limite au milieu
    rep = Guardian(cfg).validate(Trajectory(j, t, q))
    assert not rep.ok and "limit" in codes(rep)


def test_coupled_constraint_mid_path(cfg):
    """Chaque articulation reste dans ses min/max, mais la combinaison tête/tête viole une contrainte couplée."""
    j = ["HeadYaw", "HeadPitch"]
    t = np.linspace(0, 4, 201)
    s = np.sin(np.pi * t / 4) ** 2
    q = np.stack([1.4 * s, 0.4 * s], axis=1)  # 0.4 + 0.25*1.4 = 0.75 > 0.5
    rep = Guardian(cfg).validate(Trajectory(j, t, q))
    assert "coupled" in codes(rep) and "limit" not in codes(rep)


def test_head_arm_coupled_constraint_uses_context(cfg):
    """Un mouvement des bras seul est jugé avec la posture de la tête fournie en contexte."""
    j = ["LShoulderPitch", "LShoulderRoll"]
    t = np.linspace(0, 6, 301)
    s = np.sin(np.pi * t / 6) ** 2
    start = np.array(cfg.pose("start", j))
    q = start + np.stack([-2.4 * s, -0.1 * s], axis=1)  # bras gauche levé
    assert g_ok(cfg, Trajectory(j, t, q))
    g = Guardian(cfg)
    assert "coupled" not in codes(g.validate(Trajectory(j, t, q), context_pose={"HeadYaw": -1.0}))
    assert "coupled" in codes(g.validate(Trajectory(j, t, q), context_pose={"HeadYaw": 1.2}))


def test_velocity_and_acceleration(cfg):
    g = Guardian(cfg)
    j = ["HeadYaw"]
    t = np.linspace(0, 0.5, 26)
    fast = Trajectory(j, t, (1.2 * np.sin(2 * np.pi * t / 0.5))[:, None])
    c = codes(g.validate(fast))
    assert "velocity" in c and "acceleration" in c
    rep = g.validate(fast)
    assert rep.stats["suggested_time_scale"] > 1


def test_incompatible_and_invalid(cfg):
    g = Guardian(cfg)
    tr = smooth(cfg, cfg.head_joints, amp=0.05)
    assert "incompatible" in codes(g.validate(tr, expected_joints=cfg.all_joints))
    bad = {"joints": ["HeadYaw"], "t": [0, 1], "q": [[0.0], [float("nan")]]}
    assert "format" in codes(g.validate_raw(bad))
    assert "format" in codes(g.validate_raw({"joints": ["HeadYaw"], "t": [0, 1], "q": [[0.0], [0.1]], "unit": "deg"}))
    assert "format" in codes(g.validate_raw("pas une trajectoire"))
    legs = Trajectory(["LKneePitch"], [0, 1], [[0.0], [0.0]])
    assert "legs" in codes(g.validate(legs))


def test_start_mismatch_and_duration(cfg):
    g = Guardian(cfg)
    tr = smooth(cfg, cfg.head_joints, amp=0.05)
    rep = g.validate(tr, start_state={"HeadYaw": 0.8, "HeadPitch": -0.05})
    assert "start_mismatch" in codes(rep)
    long = smooth(cfg, cfg.head_joints, amp=0.05, dur=60)
    assert "duration" in codes(g.validate(long))
    with pytest.raises(GuardianRejection):
        g.require(long)
