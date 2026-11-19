"""Building the training sets from the recorded takes.

The three groups (arms, head, hands) are INDEPENDENT: each example maps
[emotion, intensity, event] to the control points and duration of the group
concerned only. Head or hands takes formerly recorded over replayed arms remain
usable: only the group's joints are read.

Each example carries a GROUP identifier used for the training/validation split:
a take forms its own group, except old head/hands takes recorded over the same
arm gesture, which share it (caution: they were demonstrated in the same context).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import Config
from ..data.representation import encode_trajectory
from ..data.store import Take, TakeStore
from ..trajectory import Trajectory
from .features import Norm, motion_input


@dataclass
class Examples:
    X: np.ndarray
    Y: np.ndarray            # flattened normalised control points
    log_dur: np.ndarray
    groups: List[str]
    take_ids: List[str]
    emotions: List[str]
    intensities: np.ndarray
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.take_ids)


def _rep(cfg, n_ctrl: Optional[int] = None):
    """(control points per joint, phase samples). `n_ctrl` overrides model.yaml for one training."""
    r = cfg.model["representation"]
    return int(n_ctrl or r["n_control_points"]), int(r["n_eval_samples"])


def group_joints(cfg: Config, kind: str):
    return {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}[kind]


# nom historique
def follower_joints(cfg: Config, kind: str):
    return group_joints(cfg, kind)


def _target_segment(store: TakeStore, take: Take, joints, target: str) -> Trajectory:
    """Useful segment of the group's joints: measurements, or the target sent (hands)."""
    if target == "command":
        ref = store.load_reference(take.meta.take_id)
        if ref is not None and set(joints) <= set(ref.joints):
            return Take(take.meta, ref).segment_trajectory(joints)
    elif target != "measured":
        raise ValueError(f"cible inconnue : {target}")
    return take.segment_trajectory(joints)


def group_examples(store: TakeStore, cfg: Config, kind: str, statuses: Sequence[str] = ("accepted",),
                   target: str = "measured", n_ctrl: Optional[int] = None) -> Examples:
    """[emotion, intensity, event] -> [normalised control points of the group, log duration]."""
    K, n = _rep(cfg, n_ctrl)
    joints = group_joints(cfg, kind)
    norm = Norm.from_config(cfg, joints)
    X, Y, D, G, T, E, I = [], [], [], [], [], [], []
    for m in store.list(kind=kind):
        if m.status not in statuses:
            continue
        take = store.load(m.take_id)
        ctrl, dur = encode_trajectory(_target_segment(store, take, joints, target), K, n)
        X.append(motion_input(cfg.emotions, m.emotion, m.intensity, m.event_ctx))
        Y.append(norm.fwd(ctrl).ravel())
        D.append(np.log(dur))
        G.append(store.group_id(m)); T.append(m.take_id); E.append(m.emotion); I.append(m.intensity)
    return _pack(X, Y, D, G, T, E, I, len(joints), K)


def mirrored_examples(store: TakeStore, cfg: Config, kind: str, take_ids: Sequence[str],
                      target: str = "measured", n_ctrl: Optional[int] = None):
    """Left/right mirrors of the given takes (in memory only, see data/mirror.py).

    Only takes WITHOUT an event are mirrored; a mirrored take that is not feasible on the
    robot (factory limits, calibrated zone) is dropped. Returns (Examples, report).
    Mirrored examples keep their source's group (they must stay on the same side of the
    split) and are named `<take_id>~mirror`."""
    from ..data.mirror import can_mirror, check_mirror, factory_limits, mirror_trajectory
    K, n = _rep(cfg, n_ctrl)
    joints = group_joints(cfg, kind)
    norm = Norm.from_config(cfg, joints)
    factory, fsource = factory_limits(cfg, joints)
    rep = {"eligible": 0, "added": 0, "dropped": 0, "skipped_with_event": 0, "dropped_takes": [],
           "factory_limits": fsource, "calibration": cfg.calibration_version}
    X, Y, D, G, T, E, I = [], [], [], [], [], [], []
    if not can_mirror(joints):
        rep["unavailable"] = f"no left/right mirror for joints {list(joints)}"
        return _pack(X, Y, D, G, T, E, I, len(joints), K), rep
    for tid in take_ids:
        m = store.load_meta(tid)
        if m.event_ctx.present:
            rep["skipped_with_event"] += 1
            continue
        rep["eligible"] += 1
        take = store.load(tid)
        seg = _target_segment(store, take, joints, target)
        mir = mirror_trajectory(seg)
        reason = check_mirror(cfg, seg, mir, factory)
        if reason:
            rep["dropped"] += 1
            rep["dropped_takes"].append({"take_id": tid, "reason": reason})
            continue
        ctrl, dur = encode_trajectory(mir, K, n)
        X.append(motion_input(cfg.emotions, m.emotion, m.intensity, m.event_ctx))
        Y.append(norm.fwd(ctrl).ravel())
        D.append(np.log(dur))
        G.append(store.group_id(m)); T.append(f"{tid}~mirror"); E.append(m.emotion); I.append(m.intensity)
        rep["added"] += 1
    return _pack(X, Y, D, G, T, E, I, len(joints), K), rep


def concat_examples(a: Examples, b: Examples) -> Examples:
    if len(b) == 0:
        return a
    if len(a) == 0:
        return b
    return Examples(np.concatenate([a.X, b.X]), np.concatenate([a.Y, b.Y]), np.concatenate([a.log_dur, b.log_dur]),
                    a.groups + b.groups, a.take_ids + b.take_ids, a.emotions + b.emotions,
                    np.concatenate([a.intensities, b.intensities]))


def arm_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",)) -> Examples:
    return group_examples(store, cfg, "arms", statuses)


def head_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",), kind: str = "head",
                  target: str = "measured") -> Examples:
    """Head (kind="head") or hands (kind="hands") examples, with no input coming from the arms."""
    return group_examples(store, cfg, kind, statuses, target)


def _pack(X, Y, D, G, T, E, I, J, K):
    if not X:
        return Examples(np.zeros((0, 0), np.float32), np.zeros((0, K * J)), np.zeros(0), [], [], [], np.zeros(0))
    return Examples(np.stack(X).astype(np.float32), np.stack(Y), np.array(D), G, T, E, np.array(I))


def grouped_split(groups: List[str], val_fraction: float, seed: int):
    """Split by groups; returns (training indices, validation indices, validation groups)."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_fraction)) if len(uniq) >= 2 else 0
    n_val = min(max(n_val, 1 if len(uniq) >= 2 else 0), len(uniq) - 1) if uniq else 0
    val_groups = set(uniq[:n_val])
    tr = [i for i, g in enumerate(groups) if g not in val_groups]
    va = [i for i, g in enumerate(groups) if g in val_groups]
    return np.array(tr, dtype=int), np.array(va, dtype=int), sorted(val_groups)


def grouped_kfold(groups: List[str], k: int, seed: int):
    """Cross-validation folds by group: the groups are shuffled (seed) and dealt into k parts
    of near-equal size; returns [(train indices, validation indices)] * k. Every group is
    validated exactly once, never in the same fold as its own training. k is capped at the
    number of groups; fewer than two groups gives no fold."""
    uniq = sorted(set(groups))
    k = min(int(k), len(uniq))
    if k < 2:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    part = {g: i % k for i, g in enumerate(uniq)}
    out = []
    for f in range(k):
        tr = [i for i, g in enumerate(groups) if part[g] != f]
        va = [i for i, g in enumerate(groups) if part[g] == f]
        out.append((np.array(tr, dtype=int), np.array(va, dtype=int)))
    return out
