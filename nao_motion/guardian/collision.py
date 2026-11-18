"""Self-collision model of NAO: forward kinematics + capsules, checked along a whole trajectory.

The guardian used to describe "where the arms may go" with joint-space boxes and convex hulls of
joint pairs measured during the calibration. Those never described the robot's geometry: they
refused most of the factory range while letting a hand go through the head or the other arm.
This module checks the real question, in Cartesian space: does a link come closer than
`clearance_m` to another link or to the body?

Geometry (torso frame, metres, x forward, y left, z up), from the Aldebaran NAO H25 V4-V6
documentation ("NAO - Links"): NeckOffsetZ 126.5 mm, ShoulderOffsetY 98 mm, ShoulderOffsetZ
100 mm, UpperArmLength 105 mm, ElbowOffsetY 15 mm, LowerArmLength 55.95 mm, HandOffsetX
57.75 mm, HandOffsetZ 12.31 mm, HipOffsetY 50 mm, HipOffsetZ 85 mm. Every link is a capsule
(segment + radius; a sphere is a capsule of length 0). The radii and the body volumes are in
guardian.yaml (`self_collision`), to be adjusted on the robot.

The legs are never commanded. Their volume comes from the calibrated REST pose (the hands rest
on the knees when NAO sits): the knees are placed just under the hands at rest, the thighs go
from the hips to the knees, the shins down from the knees. `legs: none` drops them.

Pairs checked: each arm link (upper arm, forearm, hand) against the body (torso, pelvis, legs),
the head and every link of the other arm. Links of the same arm are not checked against each
other (the joint limits already keep them apart), nor the head against the torso (the factory
limits of the neck handle it).

The same computations exist without numpy in standalone/nao_guardian.py (exported guardian);
tests check that both give the same distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NAO H25 V4-V6 (metres)
NECK_Z = 0.1265
SHOULDER_Y, SHOULDER_Z = 0.098, 0.100
UPPER_ARM, ELBOW_Y = 0.105, 0.015
LOWER_ARM = 0.05595
HAND_X, HAND_Z = 0.05775, 0.01231
HIP_Y, HIP_Z = 0.050, 0.085

ARM_JOINTS = {
    "L": ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"],
    "R": ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"],
}
HEAD_JOINTS = ["HeadYaw", "HeadPitch"]

DEFAULTS = {
    "enabled": True,
    "clearance_m": 0.01,
    "radius_m": {"upper_arm": 0.030, "forearm": 0.028, "hand": 0.030, "head": 0.065},
    # where the upper arm capsule starts along the arm (the shoulder housing is part of the torso)
    "upper_arm_start_m": 0.035,
    "hand_length_m": 0.075,         # from the wrist, along the hand
    "head_center_m": [0.0, 0.0, 0.058],  # in the head frame (after HeadYaw / HeadPitch)
    "body": [
        {"name": "torse", "a": [0.0, 0.0, -0.030], "b": [0.0, 0.0, 0.075], "radius": 0.055},
        {"name": "bassin", "a": [-0.01, -0.045, -0.085], "b": [-0.01, 0.045, -0.085], "radius": 0.050},
    ],
    "legs": "from_rest",            # from_rest | none
    "thigh_radius_m": 0.040,
    "shin_radius_m": 0.035,
    "shin_length_m": 0.13,
    "shin_direction": [0.45, 0.0, -1.0],
}


def _rot(axis: str, a: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices about x, y or z."""
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def arm_points(side: str, q: np.ndarray, upper_start: float, hand_length: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """q: (N, 5) [ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw]. Segments (a, b) of
    the upper arm, forearm and hand, each (N, 3)."""
    sy = 1.0 if side == "L" else -1.0
    n = q.shape[0]
    p_sh = np.tile([0.0, sy * SHOULDER_Y, SHOULDER_Z], (n, 1))
    R = _rot("y", q[:, 0]) @ _rot("z", q[:, 1])
    ex = R[:, :, 0]
    p_el = p_sh + R @ np.array([UPPER_ARM, sy * ELBOW_Y, 0.0])
    R = R @ _rot("x", q[:, 2]) @ _rot("z", q[:, 3])
    p_wr = p_el + R[:, :, 0] * LOWER_ARM
    R = R @ _rot("x", q[:, 4])
    p_hand = p_wr + R @ np.array([hand_length, 0.0, -HAND_Z])
    return {"upper_arm": (p_sh + ex * upper_start, p_el), "forearm": (p_el, p_wr), "hand": (p_wr, p_hand)}


def head_center(q: np.ndarray, center) -> np.ndarray:
    """q: (N, 2) [HeadYaw, HeadPitch] -> (N, 3) centre of the head sphere."""
    R = _rot("z", q[:, 0]) @ _rot("y", q[:, 1])
    return np.array([0.0, 0.0, NECK_Z]) + R @ np.asarray(center, float)


def segment_distance(p1, q1, p2, q2) -> np.ndarray:
    """Distance between segments [p1, q1] and [p2, q2], vectorised over the first axis
    (closest points, Ericson, Real-Time Collision Detection 5.1.9; degenerate segments ok)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    eps = 1e-12
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
    s = np.where(a > eps, s, 0.0)
    t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1.0), 0.0)
    # t outside [0, 1]: clamp and recompute s
    s = np.where(t < 0.0, np.where(a > eps, np.clip(-c / np.where(a > eps, a, 1.0), 0.0, 1.0), 0.0), s)
    s = np.where(t > 1.0, np.where(a > eps, np.clip((b - c) / np.where(a > eps, a, 1.0), 0.0, 1.0), 0.0), s)
    t = np.clip(t, 0.0, 1.0)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    return np.linalg.norm(c1 - c2, axis=1)


@dataclass
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass
class BodyModel:
    clearance: float
    radius: Dict[str, float]
    upper_start: float
    hand_length: float
    head_center: List[float]
    body: List[Capsule]
    enabled: bool = True
    # pairs in contact in the rest pose: tolerated on the rest <-> start segments only
    rest_contacts: set = field(default_factory=set)

    @classmethod
    def from_config(cls, sc: Optional[dict], rest_pose: Optional[Dict[str, float]] = None) -> "BodyModel":
        d = dict(DEFAULTS)
        d.update(sc or {})
        rad = dict(DEFAULTS["radius_m"])
        rad.update(d.get("radius_m") or {})
        body = [Capsule(str(c["name"]), np.array(c["a"], float), np.array(c.get("b", c["a"]), float),
                        float(c["radius"])) for c in (d.get("body") or [])]
        m = cls(clearance=float(d["clearance_m"]), radius=rad, upper_start=float(d["upper_arm_start_m"]),
                hand_length=float(d["hand_length_m"]), head_center=list(d["head_center_m"]), body=body,
                enabled=bool(d.get("enabled", True)))
        if d.get("legs", "from_rest") == "from_rest" and rest_pose:
            m.body += legs_from_rest(m, rest_pose, d)
        if rest_pose:
            m.rest_contacts = {p for p, dist in m.pose_distances(rest_pose).items() if dist < m.clearance}
        return m

    # -- geometry -----------------------------------------------------------------------
    def links(self, joints: Sequence[str], Q: np.ndarray, context: Optional[Dict[str, float]] = None):
        """Moving capsules {name: (a (N,3), b (N,3), radius)} for the limbs whose joints are all
        known (in the trajectory or in `context`)."""
        idx = {j: k for k, j in enumerate(joints)}
        ctx = dict(context or {})
        n = Q.shape[0]

        def cols(names):
            if not all(j in idx or j in ctx for j in names):
                return None
            return np.stack([Q[:, idx[j]] if j in idx else np.full(n, float(ctx[j])) for j in names], axis=1)

        out = {}
        for side in ("L", "R"):
            q = cols(ARM_JOINTS[side])
            if q is None:
                continue
            for name, (a, b) in arm_points(side, q, self.upper_start, self.hand_length).items():
                out[f"{side}_{name}"] = (a, b, self.radius[name])
        q = cols(HEAD_JOINTS)
        if q is not None:
            c = head_center(q, self.head_center)
            out["tete"] = (c, c, self.radius["head"])
        return out

    def pairs(self, names: Sequence[str]) -> List[Tuple[str, str]]:
        arm = [x for x in names if x[:2] in ("L_", "R_")]
        out = []
        for x in arm:
            out += [(x, c.name) for c in self.body]
            if "tete" in names:
                out.append((x, "tete"))
        left = [x for x in arm if x.startswith("L_")]
        right = [x for x in arm if x.startswith("R_")]
        out += [(x, y) for x in left for y in right]
        return out

    def distances(self, joints: Sequence[str], Q: np.ndarray, context=None) -> Dict[Tuple[str, str], np.ndarray]:
        """Surface distance (m, negative = interpenetration) of every checked pair, per sample."""
        Q = np.atleast_2d(np.asarray(Q, float))
        lk = self.links(joints, Q, context)
        n = Q.shape[0]
        fixed = {c.name: (np.tile(c.a, (n, 1)), np.tile(c.b, (n, 1)), c.radius) for c in self.body}
        allc = dict(lk, **fixed)
        out = {}
        for x, y in self.pairs(list(lk)):
            a1, b1, r1 = allc[x]
            a2, b2, r2 = allc[y]
            out[(x, y)] = segment_distance(a1, b1, a2, b2) - r1 - r2
        return out

    def pose_distances(self, pose: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        js = list(pose)
        return {k: float(v[0]) for k, v in self.distances(js, np.array([[pose[j] for j in js]])).items()}

    def violations(self, joints, Q, context=None, ignore=(), threshold: Optional[float] = None) -> List[dict]:
        """[{pair, first index, number of samples, min distance, index of the min}] for the pairs
        closer than `threshold` (default: the clearance)."""
        lim = self.clearance if threshold is None else threshold
        out = []
        for pair, d in self.distances(joints, Q, context).items():
            if pair in ignore:
                continue
            bad = np.nonzero(d < lim)[0]
            if bad.size:
                k = int(np.argmin(d))
                out.append({"pair": pair, "first": int(bad[0]), "count": int(bad.size),
                            "min_distance": float(d[k]), "at": k})
        return out

    def to_dict(self) -> dict:
        """Resolved model (legs included), for the exported guardian."""
        return {"enabled": self.enabled, "clearance_m": self.clearance, "radius_m": dict(self.radius),
                "upper_arm_start_m": self.upper_start, "hand_length_m": self.hand_length,
                "head_center_m": [float(x) for x in self.head_center],
                "body": [{"name": c.name, "a": c.a.tolist(), "b": c.b.tolist(), "radius": c.radius} for c in self.body],
                "legs": "none",  # already resolved into `body`
                "rest_contacts": sorted([list(p) for p in self.rest_contacts])}


# the rest hand must be at least this far in front of the hips to stand for a knee (sitting)
LEGS_MIN_FORWARD_M = 0.07


def legs_from_rest(m: BodyModel, rest: Dict[str, float], d: dict) -> List[Capsule]:
    """Thighs and shins placed from the rest pose: the knee sits just under the hand at rest
    (NAO sitting, hands on the knees). A side is skipped without its arm joints, or when its
    rest hand is not in front of the body (rest pose that is not sitting: arms hanging...)."""
    out = []
    tr, sr = float(d["thigh_radius_m"]), float(d["shin_radius_m"])
    sd = np.array(d["shin_direction"], float)
    sd = sd / np.linalg.norm(sd)
    for side, sy in (("L", 1.0), ("R", -1.0)):
        if not all(j in rest for j in ARM_JOINTS[side]):
            continue
        q = np.array([[rest[j] for j in ARM_JOINTS[side]]])
        _, tip = arm_points(side, q, m.upper_start, m.hand_length)["hand"]
        wr, _ = arm_points(side, q, m.upper_start, m.hand_length)["hand"]
        palm = (wr[0] + tip[0]) / 2
        if palm[0] < LEGS_MIN_FORWARD_M or palm[2] > SHOULDER_Z - 0.03:
            continue  # hand not resting in front, below the shoulder: not a sitting rest pose
        knee = palm - np.array([0.0, 0.0, m.radius["hand"] + tr])
        hip = np.array([0.0, sy * HIP_Y, -HIP_Z])
        out.append(Capsule(f"cuisse_{side}", hip, knee, tr))
        out.append(Capsule(f"tibia_{side}", knee, knee + sd * float(d["shin_length_m"]), sr))
    return out


def rest_pose(cfg) -> Dict[str, float]:
    flat = {}
    for g in cfg.poses.get("rest", {}).values():
        flat.update(g)
    return {j: float(v) for j, v in flat.items()}


def for_config(cfg) -> BodyModel:
    """Body model of a configuration (built once per Config object)."""
    m = cfg.__dict__.get("_body_model")
    if m is None:
        m = BodyModel.from_config(cfg.guardian.self_collision, rest_pose(cfg))
        cfg.__dict__["_body_model"] = m
    return m
