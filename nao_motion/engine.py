"""Moteur de génération : bras puis tête, sur une base temporelle commune.

Les deux modèles reçoivent le même événement (direction + distance) : un geste
de pointage des bras et le regard restent cohérents.

Le moteur PROPOSE des trajectoires ; il n'exécute rien. L'exécution passe par
le contrôleur, qui fait valider le cycle complet par le guardian.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from .config import Config
from .data.representation import decode
from .events import EventContext
from .models.features import Norm, arm_input, check_intensity, emotion_vector, head_input
from .models.nets import CheckpointError, load_model
from .trajectory import Trajectory

Emotion = Union[str, Dict[str, float]]


@dataclass
class Generated:
    arms: Trajectory
    head: Trajectory
    combined: Trajectory
    arm_ctrl: np.ndarray
    head_ctrl: np.ndarray
    duration: float
    inputs: dict
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {"inputs": self.inputs, "duration_s": self.duration, "notes": self.notes,
                "trajectory": self.combined.to_dict()}


class MotionEngine:
    def __init__(self, cfg: Config, arm_model=None, head_model=None, rate_hz: float = 50.0):
        self.cfg = cfg
        self.arm_model = arm_model
        self.head_model = head_model
        self.rate = rate_hz
        self.K = int(cfg.model["representation"]["n_control_points"])
        self.arm_norm = Norm.from_config(cfg, cfg.arm_joints)
        self.head_norm = Norm.from_config(cfg, cfg.head_joints)
        if head_model is not None and arm_model is not None:
            ref = head_model.meta.get("arm_checkpoint_digest")
            if ref and ref != arm_model.digest:
                self_note = (f"Le modèle tête a été entraîné avec le modèle bras {ref}, "
                             f"le modèle bras chargé est {arm_model.digest}")
                head_model.warnings.append(self_note)

    @classmethod
    def from_paths(cls, cfg, arm_path=None, head_path=None):
        arm = load_model(arm_path, cfg, "arms") if arm_path else None
        head = load_model(head_path, cfg, "head") if head_path else None
        return cls(cfg, arm, head)

    def plan_arms(self, emotion: Emotion, intensity: float, variation: float = 0.0, seed: Optional[int] = None,
                  event: Optional[EventContext] = None):
        if self.arm_model is None:
            raise CheckpointError("Aucun modèle des bras chargé")
        out = self.arm_model.predict(arm_input(self.cfg.emotions, emotion, intensity, event))[0]
        ctrl_n = out[:-1].reshape(self.K, -1)
        if variation > 0:
            ctrl_n = ctrl_n + self._smooth_noise(ctrl_n.shape, variation, seed)
        dur = float(np.exp(out[-1]))
        g = self.cfg.guardian
        dur = float(np.clip(dur, g.min_duration_s, g.max_duration_s))
        return ctrl_n, dur

    def _smooth_noise(self, shape, sigma, seed):
        """Variation non apprise : bruit lisse sur les points de contrôle intérieurs.

        C'est une première façon simple de produire des variantes ; une variante
        apprise (latent) est prévue dans les modèles (latent_dim)."""
        rng = np.random.default_rng(seed)
        n = rng.normal(0, sigma, size=shape)
        n = (n + np.roll(n, 1, axis=0) + np.roll(n, -1, axis=0)) / 3
        n[0] = n[-1] = 0.0
        return n

    def plan_head(self, emotion: Emotion, intensity: float, arm_ctrl_n, arm_dur: float,
                  head_current: Optional[Dict[str, float]], event: EventContext):
        hj = self.cfg.head_joints
        cur = np.array([head_current[j] for j in hj]) if head_current else np.array(self.cfg.pose("start", hj))
        if self.head_model is None:
            # repli explicite : tête maintenue en position de départ
            return np.tile(self.head_norm.fwd(np.array(self.cfg.pose("start", hj))), (self.K, 1)), \
                ["Aucun modèle tête : tête maintenue en position de départ"]
        x = head_input(self.cfg.emotions, emotion, intensity, arm_ctrl_n, np.log(arm_dur),
                       self.head_norm.fwd(cur), event)
        return self.head_model.predict(x)[0].reshape(self.K, -1), []

    def generate(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                 head_current: Optional[Dict[str, float]] = None, variation: float = 0.0,
                 seed: Optional[int] = None) -> Generated:
        check_intensity(intensity)
        emotion_vector(self.cfg.emotions, emotion)
        event = event or EventContext.none()
        arm_n, dur = self.plan_arms(emotion, intensity, variation, seed, event)
        head_n, notes = self.plan_head(emotion, intensity, arm_n, dur, head_current, event)
        arm_ctrl = self.arm_norm.inv(arm_n)
        head_ctrl = self.head_norm.inv(head_n)
        arms = decode(arm_ctrl, dur, self.cfg.arm_joints, self.rate)
        head = decode(head_ctrl, dur, self.cfg.head_joints, self.rate)
        notes += list(getattr(self.arm_model, "warnings", [])) + list(getattr(self.head_model, "warnings", []) or [])
        return Generated(arms, head, Trajectory.merge(arms, head), arm_ctrl, head_ctrl, dur,
                         {"emotion": emotion, "intensity": intensity, "event": event.to_dict(),
                          "variation": variation, "seed": seed}, notes)
