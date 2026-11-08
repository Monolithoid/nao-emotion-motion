"""Moteur de génération : bras puis tête, sur une base temporelle commune.

Les deux modèles reçoivent le même événement (direction + distance) : un geste
de pointage des bras et le regard restent cohérents. Un troisième modèle,
optionnel, produit l'ouverture des mains (LHand, RHand : 0 fermée .. 1 ouverte)
sur la même phase que les bras. Sans modèle des mains, les mains ne sont pas
commandées (elles restent comme elles sont).

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
    hands: Optional[Trajectory] = None
    hand_ctrl: Optional[np.ndarray] = None

    def to_dict(self):
        return {"inputs": self.inputs, "duration_s": self.duration, "notes": self.notes,
                "trajectory": self.combined.to_dict()}


class MotionEngine:
    def __init__(self, cfg: Config, arm_model=None, head_model=None, rate_hz: float = 50.0, hands_model=None):
        self.cfg = cfg
        self.arm_model = arm_model
        self.head_model = head_model
        self.hands_model = hands_model
        self.rate = rate_hz
        self.K = int(cfg.model["representation"]["n_control_points"])
        self.arm_norm = Norm.from_config(cfg, cfg.arm_joints)
        self.head_norm = Norm.from_config(cfg, cfg.head_joints)
        self.hand_norm = Norm.from_config(cfg, cfg.hand_joints) if cfg.hand_joints else None
        for name, m in (("tête", head_model), ("mains", hands_model)):
            if m is not None and arm_model is not None:
                ref = m.meta.get("arm_checkpoint_digest")
                if ref and ref != arm_model.digest:
                    m.warnings.append(f"Le modèle {name} a été entraîné avec le modèle bras {ref}, "
                                      f"le modèle bras chargé est {arm_model.digest}")

    @classmethod
    def from_paths(cls, cfg, arm_path=None, head_path=None, hands_path=None):
        arm = load_model(arm_path, cfg, "arms") if arm_path else None
        head = load_model(head_path, cfg, "head") if head_path else None
        hands = load_model(hands_path, cfg, "hands") if hands_path and cfg.hand_joints else None
        return cls(cfg, arm, head, hands_model=hands)

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

    def plan_hands(self, emotion: Emotion, intensity: float, arm_ctrl_n, arm_dur: float,
                   hands_current: Optional[Dict[str, float]], event: EventContext):
        """Ouverture des mains normalisée (K×2), ou None sans modèle des mains."""
        if self.hands_model is None or not self.cfg.hand_joints:
            return None
        hj = self.cfg.hand_joints
        cur = np.array([hands_current[j] for j in hj]) if hands_current else np.array(self.cfg.pose("start", hj))
        x = head_input(self.cfg.emotions, emotion, intensity, arm_ctrl_n, np.log(arm_dur),
                       self.hand_norm.fwd(cur), event)
        return self.hands_model.predict(x)[0].reshape(self.K, -1)

    def generate(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                 head_current: Optional[Dict[str, float]] = None, variation: float = 0.0,
                 seed: Optional[int] = None, hands_current: Optional[Dict[str, float]] = None) -> Generated:
        check_intensity(intensity)
        emotion_vector(self.cfg.emotions, emotion)
        event = event or EventContext.none()
        arm_n, dur = self.plan_arms(emotion, intensity, variation, seed, event)
        head_n, notes = self.plan_head(emotion, intensity, arm_n, dur, head_current, event)
        arm_ctrl = self.arm_norm.inv(arm_n)
        head_ctrl = self.head_norm.inv(head_n)
        arms = decode(arm_ctrl, dur, self.cfg.arm_joints, self.rate)
        head = decode(head_ctrl, dur, self.cfg.head_joints, self.rate)
        combined = Trajectory.merge(arms, head)
        hands = hand_ctrl = None
        hand_n = self.plan_hands(emotion, intensity, arm_n, dur, hands_current, event)
        if hand_n is not None:
            hand_ctrl = self.hand_norm.inv(hand_n)
            hands = decode(hand_ctrl, dur, self.cfg.hand_joints, self.rate)
            combined = Trajectory.merge(combined, hands)
        elif self.cfg.hand_joints:
            notes.append("Aucun modèle des mains : mains non commandées (laissées telles quelles)")
        for m in (self.arm_model, self.head_model, self.hands_model):
            notes += list(getattr(m, "warnings", []) or [])
        return Generated(arms, head, combined, arm_ctrl, head_ctrl, dur,
                         {"emotion": emotion, "intensity": intensity, "event": event.to_dict(),
                          "variation": variation, "seed": seed}, notes, hands, hand_ctrl)
