"""SYNTHETIC demonstrators for the simulation.

They replace the operator's hand to check that the pipeline works technically
(collection, data, training, generation, guardian). They say NOTHING about real
expressiveness: the shapes below are arbitrary. The takes they produce are marked
`synthetic: true`.

"surprise" and "anger" (anger) point towards the event when there is one (arm on
the event's side, more extended the closer the event), so that the pipeline can
learn a gesture oriented by angle and distance.

Some emotions deliberately have two gesture variants under the same label, to
illustrate that a deterministic regression model tends to produce their average.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from ..config import Config
from ..events import EventContext

# amplitude (rad) per "generic" joint on the left side; the right side is mirrored
# keys: sp=ShoulderPitch, sr=ShoulderRoll, ey=ElbowYaw, er=ElbowRoll, wy=WristYaw
STYLES = {
    "joy": [dict(sp=-0.8, sr=0.30, er=0.25, freq=1.2, osc=0.05, dur=3.2),
             dict(sp=-0.6, sr=0.15, ey=0.4, er=-0.3, freq=1.2, osc=0.08, dur=3.4, one_arm=True)],
    "sadness": [dict(sp=0.25, sr=-0.10, er=0.35, freq=0.5, osc=0.0, dur=3.8)],
    "anger": [dict(sp=-0.5, sr=0.05, er=0.5, freq=1.5, osc=0.04, dur=2.6)],
    "fear": [dict(sp=-0.4, sr=-0.10, er=-0.4, freq=2.0, osc=0.015, dur=2.8)],
    "surprise": [dict(sp=-0.7, sr=0.40, er=0.35, freq=1.0, osc=0.0, dur=2.6)],
    "calm": [dict(sp=-0.25, sr=0.20, ey=0.25, freq=0.8, osc=0.03, dur=3.2)],
}
_SIDE_JOINT = {"sp": "ShoulderPitch", "sr": "ShoulderRoll", "ey": "ElbowYaw", "er": "ElbowRoll", "wy": "WristYaw"}
# signs to get the right-side mirror (NAOqi convention)
_MIRROR = {"sp": 1, "sr": -1, "ey": -1, "er": -1, "wy": -1}


def _env(u):
    return np.sin(np.pi * np.clip(u, 0, 1)) ** 2


POINTING = ("surprise", "anger")


def pointing_pose(cfg: Config, event: EventContext) -> Dict[str, float]:
    """Approximate pose of the arm on the event's side, extended towards it (NAOqi convention:
    ShoulderPitch = -elevation, ShoulderRoll = azimuth, elbow almost straight)."""
    side = "L" if event.azimuth_rad >= 0 else "R"
    m = cfg.guardian.limit_margin_rad + 0.02
    want = {"ShoulderPitch": -event.elevation_rad, "ShoulderRoll": event.azimuth_rad,
            "ElbowYaw": 0.0, "ElbowRoll": -0.1 if side == "L" else 0.1}
    out = {}
    for name, v in want.items():
        j = side + name
        if j in cfg.limits:
            lo, hi = cfg.limits[j]
            out[j] = float(np.clip(v, lo + m, hi - m))
    return out


def arm_gesture(cfg: Config, emotion: str, intensity: float, rng: np.random.Generator,
                style: Optional[int] = None, event: Optional[EventContext] = None):
    """Returns (function t -> {joint: rad}, duration) for an arm gesture."""
    if event is not None and event.present and emotion in POINTING:
        return _pointing_gesture(cfg, emotion, intensity, rng, event)
    styles = STYLES.get(emotion, STYLES["calm"])
    s = dict(styles[rng.integers(len(styles)) if style is None else style % len(styles)])
    amp = (0.35 + 0.65 * intensity) * rng.uniform(0.85, 1.15)
    dur = s["dur"] * (1.2 - 0.25 * intensity) * rng.uniform(0.9, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    start = dict(zip(cfg.arm_joints, cfg.pose("start", cfg.arm_joints)))
    m = cfg.guardian.limit_margin_rad + 0.02

    def fn(t: float) -> Dict[str, float]:
        u = t / dur
        e = float(_env(u))
        osc = s.get("osc", 0.0) * math.sin(2 * math.pi * s["freq"] * t + phase)
        out = {}
        for side, sign_side in (("L", 1), ("R", -1)):
            active = not (s.get("one_arm") and side == "R")
            for key, name in _SIDE_JOINT.items():
                j = side + name
                if j not in start:
                    continue
                a = s.get(key, 0.0)
                if key == "sp":
                    a = a + osc
                sign = 1 if side == "L" else _MIRROR[key]
                val = start[j] + (sign * amp * a * e if active else 0.0)
                if key == "er" and active:
                    val += sign * amp * osc * e * 0.5
                lo, hi = cfg.limits[j]
                out[j] = float(np.clip(val, lo + m, hi - m))
        return out

    return fn, float(dur)


def _pointing_gesture(cfg, emotion, intensity, rng, event: EventContext):
    """Points towards the event: out, hold, back. The closer the event, the further
    the arm goes towards the pointing pose (unknown proximity: average value)."""
    start = dict(zip(cfg.arm_joints, cfg.pose("start", cfg.arm_joints)))
    target = pointing_pose(cfg, event)
    close = 1.0 / (1.0 + event.distance_m) if event.distance_m else 0.4
    w = float(np.clip((0.45 + 0.55 * intensity) * (0.55 + 0.9 * close), 0.3, 1.0)) * rng.uniform(0.92, 1.05)
    dur = (2.8 if emotion == "surprise" else 2.4) * (1.15 - 0.2 * intensity) * rng.uniform(0.9, 1.1)
    # sin² envelope: peak velocity = pi * amplitude / duration; kept under ~80 % of the limit
    vmax = 0.8 * cfg.guardian.max_velocity_rad_s
    amp = max(abs(v - start[j]) for j, v in target.items()) * min(w, 1.0)
    dur = max(dur, math.pi * amp / vmax)

    def fn(t: float) -> Dict[str, float]:
        e = float(_env(t / dur))
        out = dict(start)
        for j, v in target.items():
            out[j] = float(start[j] + min(w, 1.0) * e * (v - start[j]))
        return out

    return fn, float(dur)


def head_behaviour(cfg: Config, emotion: str, intensity: float, event: EventContext,
                   arm_fn: Callable[[float], Dict[str, float]], rng: np.random.Generator):
    """Synthetic gaze policy: look at the event, follow a hand with
    a delay, or look away, depending on the emotion. A close event gives a
    faster reaction and a stronger avoidance. Arbitrary, for tests."""
    start = dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints)))
    lag = rng.uniform(0.1, 0.35)
    mode = {"joy": "event", "surprise": "event", "anger": "event",
            "sadness": "avert", "fear": "avert", "calm": "hands"}.get(emotion, "event")
    if mode == "event" and not event.present:
        mode = "hands" if emotion == "joy" else "front"
    m = cfg.guardian.limit_margin_rad + 0.02
    close = 1.0 / (1.0 + event.distance_m) if event.present and event.distance_m else 0.4  # proximity
    T = 1.6 - 1.2 * close  # about 1.2 s at 1 m; faster if close
    lim = cfg.limits
    # target amplitude -> minimum duration to stay under ~70 % of the guardian limits
    # (quintic ramp: max velocity 1.875 A/T, max acceleration 5.77 A/T²); the head
    # is no longer tied to the duration of an arm gesture, it has its own tempo
    if mode == "event":
        amp = max(abs(np.clip(event.azimuth_rad, -1.2, 1.2) - start["HeadYaw"]),
                  abs(-event.elevation_rad - start["HeadPitch"]))
    elif mode == "avert":
        amp = 0.4 + 0.75 * close
    else:
        amp = 0.4
    g = cfg.guardian
    vmax = min(g.velocity_limit(j) for j in cfg.head_joints)
    amax = min(g.acceleration_limit(j) for j in cfg.head_joints)
    T = max(T, 1.875 * amp / (0.7 * vmax), np.sqrt(5.77 * amp / (0.7 * amax)))

    def clipped(y, p):
        return {"HeadYaw": float(np.clip(y, lim["HeadYaw"][0] + m, lim["HeadYaw"][1] - m)),
                "HeadPitch": float(np.clip(p, lim["HeadPitch"][0] + m, lim["HeadPitch"][1] - m))}

    def ramp(t, T):
        x = np.clip(t / T, 0, 1)
        return 10 * x**3 - 15 * x**4 + 6 * x**5

    def fn(t: float) -> Dict[str, float]:
        r = float(ramp(t, T))
        y0, p0 = start["HeadYaw"], start["HeadPitch"]
        if mode == "event":
            ty, tp = np.clip(event.azimuth_rad, -1.2, 1.2), -event.elevation_rad
            return clipped(y0 + r * (ty - y0) * (0.6 + 0.4 * intensity), p0 + r * (tp - p0))
        if mode == "avert":
            away = (-np.sign(event.azimuth_rad or 0.3) * (0.3 + 0.75 * close) * intensity if event.present
                    else 0.4 * intensity)
            return clipped(y0 + r * away, p0 + r * 0.3 * intensity)
        if mode == "hands":
            a = arm_fn(max(0.0, t - lag))
            # looks towards the left hand, very roughly
            y = 0.35 * (a.get("LShoulderRoll", 0.2) - 0.2) * 2
            p = 0.15 + 0.2 * (a.get("LShoulderPitch", 1.1) - 1.1)
            return clipped(y0 + r * (y - y0), p0 + r * (p - p0))
        return clipped(y0, p0)

    return fn


# hand opening (0 closed, 1 open) targeted at the core of the gesture, per emotion
HAND_OPENING = {"joy": 0.85, "surprise": 1.0, "anger": 0.05, "fear": 0.6, "sadness": 0.15, "calm": 0.5}


def hand_behaviour(cfg: Config, emotion: str, intensity: float, event: EventContext, duration: float,
                   rng: np.random.Generator):
    """Simulated operator for the hands pass: drives the hand opening, arms and head stiff
    (function of time u in s -> {LHand, RHand}). Hands open for
    joy or surprise, fists clenched for anger; with an event, the hand on the event's
    side opens more (gesture reaching towards it). Arbitrary, for tests."""
    hj = list(cfg.hand_joints)
    start = dict(zip(hj, cfg.pose("start", hj)))
    peak = HAND_OPENING.get(emotion, 0.5)
    lag = rng.uniform(0.0, 0.25)
    side = None
    if event is not None and event.present:
        side = "L" if event.azimuth_rad >= 0 else "R"

    def fn(u: float) -> Dict[str, float]:
        e = float(_env((u - lag) / max(duration, 1e-3)))
        out = {}
        for j in hj:
            p = peak
            if side is not None:
                p = min(1.0, peak + 0.3) if j.startswith(side) else peak * 0.6
            v = start[j] + intensity * e * (p - start[j])
            lo, hi = cfg.limits[j]
            m = cfg.guardian.margin(j)
            out[j] = float(np.clip(v, lo + m, hi - m))
        return out

    return fn
