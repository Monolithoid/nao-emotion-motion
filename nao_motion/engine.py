"""Generation engine: three INDEPENDENT models (arms, head, hands) combined.

Each model receives the same prompt [emotion, intensity, event] and produces the
trajectory of its own group only, with its own duration. No model reads another's
output: arms, head and hands are trained separately. Consistency between groups
comes from the shared prompt (the arm points to the event, the head looks at it),
not from a learned synchronisation.

Assembly (configs/model.yaml, `combine.timing`):
- `hold` (default): all groups start together, each at its learned tempo; a
  shorter group holds its final pose until the end of the longest one;
- `stretch`: each group is stretched over the longest duration (slower, never
  faster than what was learned).

A group without a model: arms or head held in the start position, hands not
commanded (left as they are). At least one model is required.

CVAE models (model.yaml `cvae`): each group draws its own z ~ N(0, T²),
T = `temperature` (0 = central gesture z = 0, default `cvae.temperature`), or
receives a given z (`latent`, e.g. the z of a take on the Generation tab map).

The engine PROPOSES trajectories; it executes nothing. Execution goes through the
controller, which has the full cycle validated by the guardian.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from .config import Config
from .data.representation import bspline_basis
from .events import EventContext
from .models.features import Norm, check_intensity, emotion_vector, motion_input
from .models.nets import CheckpointError, is_cvae, load_model
from .trajectory import Trajectory

Emotion = Union[str, Dict[str, float]]
TIMINGS = ("hold", "stretch")


def model_k(model, default: int) -> int:
    """Control points per joint of a trained model: its own (stored at training, the setting can
    differ between runs), else model.yaml for models older than that field."""
    rep = (getattr(model, "meta", None) or {}).get("representation") or {}
    return int(rep.get("n_control_points") or default)


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
    durations: Dict[str, float] = field(default_factory=dict)   # own duration of each generated group

    def to_dict(self):
        return {"inputs": self.inputs, "duration_s": self.duration, "notes": self.notes,
                "group_durations_s": self.durations, "trajectory": self.combined.to_dict()}


class MotionEngine:
    def __init__(self, cfg: Config, arm_model=None, head_model=None, rate_hz: float = 50.0, hands_model=None):
        self.cfg = cfg
        self.arm_model = arm_model
        self.head_model = head_model
        self.hands_model = hands_model
        self.rate = rate_hz
        self.K = int(cfg.model["representation"]["n_control_points"])  # groups without a model
        self.arm_norm = Norm.from_config(cfg, cfg.arm_joints)
        self.head_norm = Norm.from_config(cfg, cfg.head_joints)
        self.hand_norm = Norm.from_config(cfg, cfg.hand_joints) if cfg.hand_joints else None
        self.timing = str((cfg.model.get("combine") or {}).get("timing", "hold"))
        if self.timing not in TIMINGS:
            raise ValueError(f"combine.timing inconnu : {self.timing} (attendu {TIMINGS})")

    @classmethod
    def from_paths(cls, cfg, arm_path=None, head_path=None, hands_path=None):
        arm = load_model(arm_path, cfg, "arms") if arm_path else None
        head = load_model(head_path, cfg, "head") if head_path else None
        hands = load_model(hands_path, cfg, "hands") if hands_path and cfg.hand_joints else None
        return cls(cfg, arm, head, hands_model=hands)

    @property
    def has_model(self) -> bool:
        return any(m is not None for m in (self.arm_model, self.head_model, self.hands_model))

    def default_temperature(self) -> float:
        return float((self.cfg.model.get("cvae") or {}).get("temperature", 1.0))

    def latent_for(self, model, temperature: Optional[float] = None, seed: Optional[int] = None,
                   z=None) -> Optional[np.ndarray]:
        """z of a CVAE model: the given one, else drawn from N(0, T²). None for a plain MLP."""
        if not is_cvae(model):
            return None
        d = int(model.arch["latent_dim"])
        if z is not None:
            z = np.asarray(z, dtype=np.float64).ravel()
            if z.shape != (d,) or not np.isfinite(z).all():
                raise ValueError(f"z attendu de dimension {d}")
            return z
        t = self.default_temperature() if temperature is None else float(temperature)
        if not np.isfinite(t) or t < 0:
            raise ValueError("Température invalide (>= 0)")
        # own stream, distinct from the smooth-noise one (same seed => same gesture)
        rng = np.random.default_rng(None if seed is None else [int(seed), 7])
        return t * rng.normal(size=d)

    def _plan(self, model, emotion: Emotion, intensity: float, event: Optional[EventContext],
              variation: float = 0.0, seed: Optional[int] = None, z=None):
        """Normalised control points (K×J) and duration (s) of a group. `z`: latent code
        (CVAE models only)."""
        x = motion_input(self.cfg.emotions, emotion, intensity, event)
        out = (model.predict(x) if z is None else model.predict(x, z))[0]
        ctrl_n = out[:-1].reshape(model_k(model, self.K), -1)
        if variation > 0:
            ctrl_n = ctrl_n + self._smooth_noise(ctrl_n.shape, variation, seed)
        g = self.cfg.guardian
        dur = float(np.clip(np.exp(out[-1]), g.min_duration_s, g.max_duration_s))
        return ctrl_n, dur

    def plan_arms(self, emotion: Emotion, intensity: float, variation: float = 0.0, seed: Optional[int] = None,
                  event: Optional[EventContext] = None):
        if self.arm_model is None:
            raise CheckpointError("Aucun modèle des bras chargé")
        return self._plan(self.arm_model, emotion, intensity, event, variation, seed)

    def plan_head(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                  variation: float = 0.0, seed: Optional[int] = None):
        if self.head_model is None:
            raise CheckpointError("Aucun modèle de la tête chargé")
        return self._plan(self.head_model, emotion, intensity, event, variation, seed)

    def plan_hands(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                   variation: float = 0.0, seed: Optional[int] = None):
        if self.hands_model is None or not self.cfg.hand_joints:
            raise CheckpointError("Aucun modèle des mains chargé")
        return self._plan(self.hands_model, emotion, intensity, event, variation, seed)

    def _smooth_noise(self, shape, sigma, seed):
        """Non-learned variation: smooth noise on the inner control points.

        This is a first simple way to produce variants; a learned
        variant (latent) is provided in the models (latent_dim)."""
        rng = np.random.default_rng(seed)
        n = rng.normal(0, sigma, size=shape)
        n = (n + np.roll(n, 1, axis=0) + np.roll(n, -1, axis=0)) / 3
        n[0] = n[-1] = 0.0
        return n

    def model_of(self, group: str):
        return {"arms": self.arm_model, "head": self.head_model, "hands": self.hands_model}.get(group)

    def generate(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                 head_current: Optional[Dict[str, float]] = None, variation: float = 0.0,
                 seed: Optional[int] = None, hands_current: Optional[Dict[str, float]] = None,
                 project_zone: bool = False, temperature: Optional[float] = None,
                 latent: Optional[Dict[str, list]] = None) -> Generated:
        """Combined arms + head + hands gesture. `head_current` and `hands_current` are no longer
        model inputs (kept for call compatibility): the transitions from the
        measured posture are planned by the controller."""
        check_intensity(intensity)
        emotion_vector(self.cfg.emotions, emotion)
        if not self.has_model:
            raise CheckpointError("Aucun modèle entraîné (bras, tête ou mains)")
        event = event or EventContext.none()
        cfg, notes = self.cfg, []
        sub = (lambda k: None if seed is None else int(seed) + k)  # one seed per group
        plans, zs = {}, {}
        latent = latent or {}
        for k, group in enumerate(("arms", "head", "hands")):
            model = self.model_of(group)
            if model is None or (group == "hands" and not cfg.hand_joints):
                continue
            z = self.latent_for(model, temperature, sub(k), latent.get(group))
            if z is not None:
                zs[group] = [round(float(v), 4) for v in z]
            plans[group] = self._plan(model, emotion, intensity, event, variation, sub(k), z)
        T = max(d for _, d in plans.values())
        n = max(4, int(np.ceil(T * self.rate)) + 1)
        tt = np.linspace(0.0, T, n)

        def build(group, joints, norm, ctrl_n, dur, lo=None, hi=None):
            ctrl = norm.inv(ctrl_n)
            if project_zone:
                # control points outside the guardian zone -> nearest point inside (convex hull
                # property: the decoded curve stays in the zone for these position constraints)
                from .guardian.project import project_control_points
                ctrl, moved, n_moved = project_control_points(cfg, ctrl, joints)
                if n_moved:
                    notes.append(f"{group}: {n_moved} control point(s) moved into the zone (max {moved:.3f} rad)")
            if lo is not None:
                # control points clamped into the range: by the convex hull property
                # of B-splines, the curve stays inside too, without a velocity break
                ctrl = np.clip(ctrl, lo, hi)
            d = T if self.timing == "stretch" else dur
            # beyond its duration, the group holds its final pose (phase capped at 1)
            u = np.clip(tt / d, 0.0, 1.0)
            return Trajectory(joints, tt, bspline_basis(ctrl.shape[0], u) @ ctrl), ctrl

        def held(joints):
            q0 = np.array(cfg.pose("start", joints))
            return Trajectory(joints, tt, np.tile(q0, (n, 1))), np.tile(q0, (self.K, 1))

        if "arms" in plans:
            arms, arm_ctrl = build("arms", cfg.arm_joints, self.arm_norm, *plans["arms"])
        else:
            arms, arm_ctrl = held(cfg.arm_joints)
            notes.append("Aucun modèle des bras : bras maintenus en position de départ")
        if "head" in plans:
            head, head_ctrl = build("head", cfg.head_joints, self.head_norm, *plans["head"])
        else:
            head, head_ctrl = held(cfg.head_joints)
            notes.append("Aucun modèle tête : tête maintenue en position de départ")
        combined = Trajectory.merge(arms, head)
        hands = hand_ctrl = None
        if "hands" in plans:
            hj, g = cfg.hand_joints, cfg.guardian
            lo = np.array([cfg.limits[j][0] + g.margin(j) for j in hj])
            hi = np.array([cfg.limits[j][1] - g.margin(j) for j in hj])
            hands, hand_ctrl = build("hands", hj, self.hand_norm, *plans["hands"], lo=lo, hi=hi)
            combined = Trajectory.merge(combined, hands)
        elif cfg.hand_joints:
            notes.append("Aucun modèle des mains : mains non commandées (laissées telles quelles)")
        for m in (self.arm_model, self.head_model, self.hands_model):
            notes += list(getattr(m, "warnings", []) or [])
        durations = {k: round(d, 3) for k, (_, d) in plans.items()}
        return Generated(arms, head, combined, arm_ctrl, head_ctrl, T,
                         {"emotion": emotion, "intensity": intensity, "event": event.to_dict(),
                          "variation": variation, "seed": seed, "timing": self.timing,
                          "project_zone": bool(project_zone),
                          **({"temperature": self.default_temperature() if temperature is None else float(temperature),
                              "latent": zs} if zs else {})}, notes, hands, hand_ctrl,
                         durations)
