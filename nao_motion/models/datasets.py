"""Construction des jeux d'apprentissage à partir des prises enregistrées.

Chaque exemple porte un identifiant de GROUPE servant au découpage
entraînement/validation : une prise de bras forme son propre groupe ; toutes
les prises de tête enregistrées sur un même geste des bras (même prise
rejouée, ou même geste généré) partagent ce groupe. Aucune variante d'une même prise ne peut donc se
retrouver à la fois en entraînement et en validation. Les prises de mains (passage 3) suivent la
même règle que la tête.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import Config
from ..data.representation import encode_trajectory
from ..data.store import Take, TakeStore
from ..trajectory import Trajectory
from .features import Norm, arm_input, head_input


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


def arm_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",)) -> Examples:
    K, n = _rep(cfg)
    norm = Norm.from_config(cfg, cfg.arm_joints)
    X, Y, D, G, T, E, I = [], [], [], [], [], [], []
    for m in store.list(kind="arms"):
        if m.status not in statuses:
            continue
        take = store.load(m.take_id)
        ctrl, dur = encode_trajectory(take.segment_trajectory(cfg.arm_joints), K, n)
        X.append(arm_input(cfg.emotions, m.emotion, m.intensity, m.event_ctx))
        Y.append(norm.fwd(ctrl).ravel())
        D.append(np.log(dur))
        G.append(m.take_id); T.append(m.take_id); E.append(m.emotion); I.append(m.intensity)
    return _pack(X, Y, D, G, T, E, I, len(cfg.arm_joints), K)


def follower_joints(cfg: Config, kind: str):
    return {"head": cfg.head_joints, "hands": cfg.hand_joints}[kind]


def _target_segment(store: TakeStore, take: Take, joints, target: str) -> Trajectory:
    """Segment utile des articulations suivies : mesures, ou consigne envoyée (mains)."""
    if target == "command":
        ref = store.load_reference(take.meta.take_id)
        if ref is not None and set(joints) <= set(ref.joints):
            return Take(take.meta, ref).segment_trajectory(joints)
    elif target != "measured":
        raise ValueError(f"cible inconnue : {target}")
    return take.segment_trajectory(joints)


def head_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",),
                  arm_source: str = "recorded", arm_model=None, kind: str = "head",
                  target: str = "measured") -> Examples:
    """Exemples tête (kind="head") ou mains (kind="hands") : [émotion, intensité, bras,
    état initial, événement] -> points de contrôle de la tête ou de l'ouverture des mains."""
    from .features import Norm as _N
    K, n = _rep(cfg)
    fj = follower_joints(cfg, kind)
    an = _N.from_config(cfg, cfg.arm_joints)
    hn = _N.from_config(cfg, fj)
    X, Y, D, G, T, E, I = [], [], [], [], [], [], []
    for m in store.list(kind=kind):
        if m.status not in statuses:
            continue
        take = store.load(m.take_id)
        head_ctrl, dur = encode_trajectory(_target_segment(store, take, fj, target), K, n)
        if arm_source == "recorded":
            arm_ctrl, _ = encode_trajectory(take.segment_trajectory(cfg.arm_joints), K, n)
            arm_norm, ld = an.fwd(arm_ctrl), np.log(dur)
        elif arm_source == "predicted":
            if arm_model is None:
                raise ValueError("arm_source='predicted' exige le modèle des bras")
            out = arm_model.predict(arm_input(cfg.emotions, m.emotion, m.intensity, m.event_ctx))[0]
            arm_norm, ld = out[:-1].reshape(K, -1), float(out[-1])
        else:
            raise ValueError(f"arm_source inconnu : {arm_source}")
        init = m.head_initial if kind == "head" else m.hands_initial
        hc = hn.fwd(np.array(init if init is not None else head_ctrl[0]))
        X.append(head_input(cfg.emotions, m.emotion, m.intensity, arm_norm, ld, hc, m.event_ctx))
        Y.append(hn.fwd(head_ctrl).ravel())
        D.append(np.log(dur))
        G.append(store.group_id(m)); T.append(m.take_id); E.append(m.emotion); I.append(m.intensity)
    return _pack(X, Y, D, G, T, E, I, len(fj), K)


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
