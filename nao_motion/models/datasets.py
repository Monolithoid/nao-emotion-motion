"""Construction des jeux d'apprentissage à partir des prises enregistrées.

Les trois groupes (bras, tête, mains) sont INDÉPENDANTS : chaque exemple associe
[émotion, intensité, événement] aux points de contrôle et à la durée du seul
groupe concerné. Les prises de tête ou de mains enregistrées autrefois sur des
bras rejoués restent utilisables : seules les articulations du groupe sont lues.

Chaque exemple porte un identifiant de GROUPE servant au découpage
entraînement/validation : une prise forme son propre groupe, sauf les anciennes
prises de tête/mains enregistrées sur un même geste des bras, qui le partagent
(prudence : elles ont été démontrées dans le même contexte).
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
    Y: np.ndarray            # points de contrôle normalisés aplatis
    log_dur: np.ndarray
    groups: List[str]
    take_ids: List[str]
    emotions: List[str]
    intensities: np.ndarray
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.take_ids)


def _rep(cfg):
    r = cfg.model["representation"]
    return int(r["n_control_points"]), int(r["n_eval_samples"])


def group_joints(cfg: Config, kind: str):
    return {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}[kind]


# nom historique
def follower_joints(cfg: Config, kind: str):
    return group_joints(cfg, kind)


def _target_segment(store: TakeStore, take: Take, joints, target: str) -> Trajectory:
    """Segment utile des articulations du groupe : mesures, ou consigne envoyée (mains)."""
    if target == "command":
        ref = store.load_reference(take.meta.take_id)
        if ref is not None and set(joints) <= set(ref.joints):
            return Take(take.meta, ref).segment_trajectory(joints)
    elif target != "measured":
        raise ValueError(f"cible inconnue : {target}")
    return take.segment_trajectory(joints)


def group_examples(store: TakeStore, cfg: Config, kind: str, statuses: Sequence[str] = ("accepted",),
                   target: str = "measured") -> Examples:
    """[émotion, intensité, événement] -> [points de contrôle normalisés du groupe, log durée]."""
    K, n = _rep(cfg)
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


def arm_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",)) -> Examples:
    return group_examples(store, cfg, "arms", statuses)


def head_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",), kind: str = "head",
                  target: str = "measured") -> Examples:
    """Exemples tête (kind="head") ou mains (kind="hands"), sans aucune entrée venant des bras."""
    return group_examples(store, cfg, kind, statuses, target)


def _pack(X, Y, D, G, T, E, I, J, K):
    if not X:
        return Examples(np.zeros((0, 0), np.float32), np.zeros((0, K * J)), np.zeros(0), [], [], [], np.zeros(0))
    return Examples(np.stack(X).astype(np.float32), np.stack(Y), np.array(D), G, T, E, np.array(I))


def grouped_split(groups: List[str], val_fraction: float, seed: int):
    """Découpage par groupes ; retourne (indices entraînement, indices validation, groupes de validation)."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_fraction)) if len(uniq) >= 2 else 0
    n_val = min(max(n_val, 1 if len(uniq) >= 2 else 0), len(uniq) - 1) if uniq else 0
    val_groups = set(uniq[:n_val])
    tr = [i for i, g in enumerate(groups) if g not in val_groups]
    va = [i for i, g in enumerate(groups) if g in val_groups]
    return np.array(tr, dtype=int), np.array(va, dtype=int), sorted(val_groups)
