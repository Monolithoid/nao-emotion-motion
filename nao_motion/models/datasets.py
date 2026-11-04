"""Construction des jeux d'apprentissage à partir des prises enregistrées.

Chaque exemple porte un identifiant de GROUPE servant au découpage
entraînement/validation : une prise de bras forme son propre groupe ; toutes
les prises de tête enregistrées sur un même geste des bras (même prise
rejouée, ou même geste généré) partagent ce groupe. Aucune variante d'une même prise ne peut donc se
retrouver à la fois en entraînement et en validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import Config
from ..data.representation import encode_trajectory
from ..data.store import TakeStore
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
        X.append(arm_input(cfg.emotions, m.emotion, m.intensity))
        Y.append(norm.fwd(ctrl).ravel())
        D.append(np.log(dur))
        G.append(m.take_id); T.append(m.take_id); E.append(m.emotion); I.append(m.intensity)
    return _pack(X, Y, D, G, T, E, I, len(cfg.arm_joints), K)


def head_examples(store: TakeStore, cfg: Config, statuses: Sequence[str] = ("accepted",),
                  arm_source: str = "recorded", arm_model=None) -> Examples:
    from .features import Norm as _N
    K, n = _rep(cfg)
    an = _N.from_config(cfg, cfg.arm_joints)
    hn = _N.from_config(cfg, cfg.head_joints)
    X, Y, D, G, T, E, I = [], [], [], [], [], [], []
    for m in store.list(kind="head"):
        if m.status not in statuses:
            continue
        take = store.load(m.take_id)
        head_ctrl, dur = encode_trajectory(take.segment_trajectory(cfg.head_joints), K, n)
        if arm_source == "recorded":
            arm_ctrl, _ = encode_trajectory(take.segment_trajectory(cfg.arm_joints), K, n)
            arm_norm, ld = an.fwd(arm_ctrl), np.log(dur)
        elif arm_source == "predicted":
            if arm_model is None:
                raise ValueError("arm_source='predicted' exige le modèle des bras")
            out = arm_model.predict(arm_input(cfg.emotions, m.emotion, m.intensity))[0]
            arm_norm, ld = out[:-1].reshape(K, -1), float(out[-1])
        else:
            raise ValueError(f"arm_source inconnu : {arm_source}")
        hc = hn.fwd(np.array(m.head_initial if m.head_initial is not None else head_ctrl[0]))
        X.append(head_input(cfg.emotions, m.emotion, m.intensity, arm_norm, ld, hc, m.event_ctx))
        Y.append(hn.fwd(head_ctrl).ravel())
        D.append(np.log(dur))
        G.append(store.group_id(m)); T.append(m.take_id); E.append(m.emotion); I.append(m.intensity)
    return _pack(X, Y, D, G, T, E, I, len(cfg.head_joints), K)


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
