import copy

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


def with_constraints(cfg, *cs):
    """Copy of the configuration with coupled constraints chosen by the operator (none by default)."""
    from nao_motion.config import CoupledConstraint
    c = copy.deepcopy(cfg)
    c.guardian.coupled_constraints = [CoupledConstraint(n, t, b) for n, t, b in cs]
    return c


def test_no_coupled_constraint_by_default(cfg):
    """The guardian protects the motors and checks collisions: no joint-space envelope by default."""
    assert cfg.guardian.coupled_constraints == [] and not cfg.guardian.calibrated_zone


def test_coupled_constraint_mid_path(cfg):
    """Chaque articulation reste dans ses min/max, mais la combinaison tête/tête viole une contrainte couplée."""
    c = with_constraints(cfg, ("tete", {"HeadPitch": 1.0, "HeadYaw": 0.25}, 0.5))
    j = ["HeadYaw", "HeadPitch"]
    t = np.linspace(0, 4, 201)
    s = np.sin(np.pi * t / 4) ** 2
    q = np.stack([1.4 * s, 0.4 * s], axis=1)  # 0.4 + 0.25*1.4 = 0.75 > 0.5
    rep = Guardian(c).validate(Trajectory(j, t, q))
    assert "coupled" in codes(rep) and "limit" not in codes(rep)
    assert Guardian(cfg).validate(Trajectory(j, t, q)).ok  # without the constraint: accepted


def test_head_arm_coupled_constraint_uses_context(cfg):
    """Un mouvement des bras seul est jugé avec la posture de la tête fournie en contexte."""
    c = with_constraints(cfg, ("bras", {"LShoulderPitch": -1.0, "LShoulderRoll": -1.0, "HeadYaw": 0.5}, 1.4))
    j = ["LShoulderPitch", "LShoulderRoll"]
    t = np.linspace(0, 6, 301)
    s = np.sin(np.pi * t / 6) ** 2
    start = np.array(cfg.pose("start", j))
    q = start + np.stack([-2.4 * s, -0.1 * s], axis=1)  # bras gauche levé
    g = Guardian(c)
    assert "coupled" not in codes(g.validate(Trajectory(j, t, q), context_pose={"HeadYaw": -1.0}))
    assert "coupled" in codes(g.validate(Trajectory(j, t, q), context_pose={"HeadYaw": 1.2}))


def _arms_move(cfg, target, dur=3.0, head=None):
    """Arms (and head) from the start pose to `target` and back, slowly (no speed issue)."""
    js = cfg.all_joints
    t = np.linspace(0, dur, int(dur * 50) + 1)
    q0 = np.array(cfg.pose("start", js))
    q1 = q0.copy()
    for j, v in dict(target, **(head or {})).items():
        q1[js.index(j)] = v
    s = np.sin(np.pi * t / dur) ** 2
    return Trajectory(js, t, q0 + s[:, None] * (q1 - q0))


def test_wide_gestures_are_accepted(cfg):
    """Arm raised high, arms wide open, head turned: far from the body, nothing to refuse
    (the old guardian refused the raised arm with an EXAMPLE constraint)."""
    g = Guardian(cfg)
    for target in ({"LShoulderPitch": -1.9, "LShoulderRoll": 0.4, "LElbowRoll": -0.3},
                   {"LShoulderPitch": 0.0, "LShoulderRoll": 1.2, "RShoulderPitch": 0.0, "RShoulderRoll": -1.2},
                   {"RShoulderPitch": -1.0, "RShoulderRoll": -0.3, "RElbowRoll": 0.4, "HeadYaw": 1.2}):
        rep = g.validate(_arms_move(cfg, target))
        assert rep.ok, rep.summary()


@pytest.mark.parametrize("target,pair", [
    # forearms crossed in front of the chest: the hands go through each other
    ({"LShoulderPitch": 0.0, "LShoulderRoll": -0.3, "LElbowYaw": 0.0, "LElbowRoll": -1.4,
      "RShoulderPitch": 0.0, "RShoulderRoll": 0.3, "RElbowYaw": 0.0, "RElbowRoll": 1.4}, "L_hand/R_hand"),
    # hand brought onto the face
    ({"LShoulderPitch": -0.6, "LShoulderRoll": -0.2, "LElbowYaw": 0.0, "LElbowRoll": -1.5}, "L_hand/tete"),
    # arm folded down into the belly
    ({"LShoulderPitch": 1.5, "LShoulderRoll": -0.3}, "L_hand/torse"),
])
def test_self_collisions_are_refused(cfg, target, pair):
    rep = Guardian(cfg).validate(_arms_move(cfg, target))
    assert "collision" in codes(rep), rep.summary()
    assert pair in {v.joint for v in rep.violations if v.code == "collision"}
    v = next(v for v in rep.violations if v.joint == pair)
    assert 0 < v.time_s < 3.0 and v.value < cfg.guardian.self_collision.get("clearance_m", 0.01)


def test_head_arm_collision_uses_context(cfg):
    """An arm-only gesture is checked against the head posture given in context."""
    js = cfg.arm_joints
    tr = _arms_move(cfg, {"LShoulderPitch": -0.6, "LShoulderRoll": -0.2, "LElbowYaw": 0.0, "LElbowRoll": -1.5})
    arms = tr.select(js)
    g = Guardian(cfg)
    assert g.validate(arms).ok  # head unknown: head pairs cannot be checked
    assert "collision" in codes(g.validate(arms, context_pose={"HeadYaw": 0.0, "HeadPitch": 0.0}))


def test_numpy_and_standalone_collision_models_agree(cfg):
    import json
    from nao_motion.guardian import collision
    from nao_motion.standalone.nao_guardian import BodyModel
    m = collision.for_config(cfg)
    sb = BodyModel(json.loads(json.dumps(m.to_dict())))
    rng = np.random.default_rng(3)
    js = cfg.all_joints
    lo = np.array([cfg.guardian_limits[j][0] for j in js])
    hi = np.array([cfg.guardian_limits[j][1] for j in js])
    Q = lo + (hi - lo) * rng.random((200, len(js)))
    ref = m.distances(js, Q)
    for i in range(len(Q)):
        d = sb.distances(dict(zip(js, Q[i].tolist())))
        assert set(d) == set(ref)
        assert max(abs(d[p] - ref[p][i]) for p in d) < 1e-9


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
