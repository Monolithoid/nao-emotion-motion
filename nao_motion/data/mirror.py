"""Left/right mirror of recorded takes, used as data augmentation at training time only.

A gesture done with one side of NAO, played with the other side, is the same gesture
seen in a mirror. Takes are never modified on disk: mirrored copies only exist in memory,
are only added to the TRAINING part of the split (never to validation), and only for
takes WITHOUT an event (a take with an event points or looks somewhere: its mirror would
contradict the event it was recorded for).

NAO joint conventions (Aldebaran documentation, same zero pose on both sides, axes
mirrored by the sagittal plane), mirror of a left joint value onto the right joint:

    ShoulderPitch  same sign   (both arms: positive = arm down)
    ShoulderRoll   opposite    (L: positive = outwards ; R: negative = outwards)
    ElbowYaw       opposite
    ElbowRoll      opposite    (L in [-1.54, -0.03], R in [0.03, 1.54])
    WristYaw       opposite
    Hand           same value  (opening 0..1, not an angle)
    HeadYaw        opposite    (single joint: looking left <-> looking right)
    HeadPitch      same sign

The mirror must also be FEASIBLE on the robot: left and right ranges are not exactly
symmetric (factory limits, and above all the calibrated zone measured on each side).
A mirrored take is kept only if
  - it stays inside the factory limits of every joint (absolute ceiling), and
  - it does not go beyond the calibrated zone (limits and coupled envelopes, without the
    guardian margin) more than the original take already does on its own side.
Otherwise it is dropped, and the training report says which take and why.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..trajectory import Trajectory

# sign applied to the value when it moves to the other side, by joint name without L/R prefix
_SIDE_SIGN = {"ShoulderPitch": 1.0, "ShoulderRoll": -1.0, "ElbowYaw": -1.0, "ElbowRoll": -1.0,
              "WristYaw": -1.0, "Hand": 1.0}
# joints of the sagittal plane: same joint, value kept or negated
_CENTER_SIGN = {"HeadYaw": -1.0, "HeadPitch": 1.0}

# tolerance on "no further outside than the original" (sensor noise, interpolation)
TOLERANCE_RAD = 0.01


class MirrorError(ValueError):
    pass


def mirror_joint(j: str) -> Tuple[str, float]:
    """(joint receiving the value of `j` in the mirror, sign applied)."""
    if j in _CENTER_SIGN:
        return j, _CENTER_SIGN[j]
    if j[:1] in ("L", "R") and j[1:] in _SIDE_SIGN:
        return ("R" if j[0] == "L" else "L") + j[1:], _SIDE_SIGN[j[1:]]
    raise MirrorError(f"No mirror convention for joint {j}")


def mirror_trajectory(traj: Trajectory) -> Trajectory:
    """Same timing, values moved to the other side with NAO's sign conventions.
    The joint list (and its order) is unchanged: it must contain both sides of every pair."""
    idx = {j: k for k, j in enumerate(traj.joints)}
    q = np.empty_like(traj.q)
    for k, j in enumerate(traj.joints):
        src, sign = mirror_joint(j)   # the mirror is its own inverse: value of j comes from src
        if src not in idx:
            raise MirrorError(f"Cannot mirror {j}: {src} is not in the take")
        q[:, k] = sign * traj.q[:, idx[src]]
    return Trajectory(list(traj.joints), traj.t.copy(), q)


def factory_limits(cfg: Config, joints: Sequence[str]) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Factory limits read from the robot during calibration; joints.yaml limits otherwise."""
    meas = (cfg.calibration.measured or {}) if cfg.calibration is not None else {}
    out, source = {}, "calibration (read from the robot)"
    for j in joints:
        f = (meas.get(j) or {}).get("factory")
        if f:
            out[j] = (float(f[0]), float(f[1]))
        else:
            # hands: factory range from joints.yaml ; without calibration: joints.yaml placeholders
            out[j] = tuple(cfg.limits[j])
            if j not in cfg.hand_joints:
                source = "joints.yaml (no factory limits read for some joints)"
    return out, source


def _limit_excess(q: np.ndarray, joints, limits) -> Dict[str, float]:
    """Per joint: how far (>= 0) the samples go below/above the limits."""
    out = {}
    for k, j in enumerate(joints):
        lo, hi = limits[j]
        out[j] = float(max(0.0, lo - q[:, k].min(), q[:, k].max() - hi))
    return out


def _coupled_excess(cfg: Config, traj: Trajectory) -> float:
    """Largest excess over all coupled envelopes involving these joints (other joints at the start pose)."""
    others = [j for j in cfg.all_joints if j not in traj.joints]
    ctx = dict(zip(others, cfg.pose("start", others))) if others else {}
    worst = 0.0
    for c in cfg.guardian.coupled_constraints:
        if not any(j in traj.joints for j in c.terms):
            continue
        total = np.zeros(traj.t.shape[0])
        for j, coef in c.terms.items():
            if j in traj.joints:
                total += coef * traj.q[:, traj.joints.index(j)]
            elif j in ctx:
                total += coef * float(ctx[j])
            else:
                total = None
                break
        if total is not None:
            worst = max(worst, float(total.max() - c.bound))
    return worst


def check_mirror(cfg: Config, original: Trajectory, mirrored: Trajectory,
                 factory: Dict[str, Tuple[float, float]], tol: float = TOLERANCE_RAD) -> Optional[str]:
    """None if the mirrored take is feasible, else the reason (English, one line)."""
    fx = _limit_excess(mirrored.q, mirrored.joints, factory)
    bad = {j: round(v, 3) for j, v in fx.items() if v > 0}
    if bad:
        return f"outside factory limits: {bad}"
    zone = {j: cfg.limits[j] for j in original.joints}
    ex_orig = _limit_excess(original.q, original.joints, zone)
    ex_mir = _limit_excess(mirrored.q, mirrored.joints, zone)
    bad = {}
    for j in mirrored.joints:
        src, _ = mirror_joint(j)
        if ex_mir[j] > ex_orig[src] + tol:
            bad[j] = round(ex_mir[j] - ex_orig[src], 3)
    if bad:
        return f"beyond the calibrated zone ({cfg.calibration_version}) by (rad, more than the original): {bad}"
    c_orig, c_mir = _coupled_excess(cfg, original), _coupled_excess(cfg, mirrored)
    if c_mir > max(c_orig, 0.0) + tol:
        return f"beyond a coupled envelope of the calibration by {round(c_mir - max(c_orig, 0.0), 3)}"
    return None


def can_mirror(joints: Sequence[str]) -> bool:
    try:
        return all(mirror_joint(j)[0] in joints for j in joints)
    except MirrorError:
        return False
