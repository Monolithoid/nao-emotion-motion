"""Recording demonstrations: one pass per group, INDEPENDENT.

Each pass records a single group for a prompt (emotion, intensity, event); none
requires an already recorded arm gesture nor a model.

Arms pass: arms GUIDED (relaxed, read), head held stiff at its measured posture.

Head pass: head GUIDED, arms held stiff at their measured posture.
No motorised movement: the preparation only relaxes the head.

Hands pass: nothing is relaxed, arms and head stay stiff in place and the operator
DRIVES the hand opening from the interface (sliders, buttons, keyboard): a NAO hand
cannot be guided by hand. The operator's target is filtered online (guardian bounds,
velocity, acceleration) before being sent, and kept next to the measurements.

Each recording lasts until "Arrêter" (stop) is pressed (or the maximum duration); the
useful segment is suggested automatically when saving.

No joint is both relaxed and commanded.

Bridge drop during a pass: the loop stops (never an orphan "recording" state), the
controller goes into fault and is disarmed, and the measurements already received are
kept as an INCOMPLETE take, which the operator can accept, reject or discard like the
others. On the robot side, the bridge stops and holds the posture (watchdog).
"""
from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..adapters.base import ConnectionLost
from ..config import Config
from ..data.representation import encode_trajectory, decode
from ..data.store import Take
from ..events import EventContext
from ..trajectory import Trajectory
from .controller import Controller, ControllerError, State


class RecState(str, enum.Enum):
    IDLE = "idle"
    PREPARED = "prepared"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class RecordingResult:
    traj: Trajectory
    segment: tuple
    guided: List[str]
    driven: List[str]
    head_initial: Optional[List[float]]
    complete: bool
    message: str = ""
    hands_initial: Optional[List[float]] = None
    reference: Optional[Trajectory] = None   # targets sent (hands pass: hand opening)


def replay_trajectory(take: Take, joints, ctrl_per_s: float = 6.0, rate_hz: float = 50.0) -> Trajectory:
    """Smoothed, faithful version of an arms take for motorised replay.

    The raw measurements contain sensor noise that would produce
    artificial accelerations; they are projected onto a fairly fine B-spline
    (6 control points per second by default). The raw data stays
    intact on disk.
    """
    seg = take.segment_trajectory(joints)
    k = max(8, int(np.ceil(seg.duration * ctrl_per_s)) + 4)
    ctrl, dur = encode_trajectory(seg, k, max(100, len(seg.t)))
    return decode(ctrl, dur, joints, rate_hz)


class RecordingSession:
    """Shared recording loop; can run in a thread (UI) or synchronously (tests)."""

    def __init__(self, cfg: Config, controller: Controller):
        self.cfg = cfg
        self.ctrl = controller
        self.adapter = controller.adapter
        self.state = RecState.IDLE
        self.rate = float(cfg.robot.get("recording", {}).get("rate_hz", 50))
        self._stop = threading.Event()
        self.countdown_left = 0.0
        self.elapsed = 0.0
        self.result: Optional[RecordingResult] = None
        self.error: Optional[str] = None
        self.guided: List[str] = []
        self.driven: List[str] = []

    def request_stop(self):
        self._stop.set()

    def _interrupted(self, e: Exception, lost: bool):
        """Abnormal end of the loop (bridge drop or other error): consistent state,
        controller in fault and disarmed, cause visible in the interface."""
        what = "Connexion au pont perdue" if lost else "Erreur"
        self.error = f"{what} pendant l'enregistrement : {e}"
        ctrl = self.ctrl
        ctrl.armed = False
        ctrl.state = State.FAULT
        ctrl.last_error = self.error
        ctrl._log(self.error)

    @staticmethod
    def _partial(ts) -> bool:
        """Enough measurements to keep an incomplete take?"""
        return len(ts) >= 2 and ts[-1] > ts[0]

    def _countdown(self, seconds: float):
        self.state = RecState.COUNTDOWN
        a = self.adapter
        t0 = a.now()
        while a.now() - t0 < seconds:
            if self._stop.is_set():
                return False
            self.countdown_left = seconds - (a.now() - t0)
            a.heartbeat()
            a.sleep(min(0.05, seconds))
        self.countdown_left = 0.0
        return True


class GuidedRecording(RecordingSession):
    """One group GUIDED by hand (relaxed, read) while the other is held stiff."""
    kind = "arms"

    def _groups(self):
        return self.cfg.arm_joints, self.cfg.head_joints

    def prepare(self, confirm: bool = False):
        """Relaxes the guided group (it drops if nobody holds it) and holds the other."""
        if getattr(self.adapter, "is_real", False) and not confirm:
            what = {"arms": "les bras sont tenus", "head": "la tête est tenue"}[self.kind]
            raise ControllerError(f"Robot réel : confirmer que {what} avant de relâcher")
        if self.ctrl.state not in (State.IDLE, State.HOLDING):
            raise ControllerError(f"Préparation impossible dans l'état {self.ctrl.state.value}")
        self.guided, self.driven = self._groups()
        self.adapter.set_roles(guided=self.guided, driven=self.driven)
        self._stop.clear()
        self.state = RecState.PREPARED

    def run(self, countdown_s: float = 3.0, max_duration_s: float = 15.0) -> RecordingResult:
        if self.state != RecState.PREPARED:
            raise ControllerError("Préparer l'enregistrement d'abord")
        a = self.adapter
        joints = self.guided
        ts, qs = [], []
        try:
            if not self._countdown(countdown_s):
                self.state = RecState.ABORTED
                return None
            self.state = RecState.RECORDING
            t0 = a.now()
            k = 0
            while True:
                t = a.now() - t0
                q = a.read_angles(joints)  # real measurements
                ts.append(t)
                qs.append(q)
                self.elapsed = t
                if self._stop.is_set() or t >= max_duration_s:
                    break
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        except Exception as e:
            # never an orphan "recording" state: otherwise the interface refuses any
            # reconnection ("recording in progress") until the server restarts
            self._interrupted(e, isinstance(e, ConnectionLost))
            if not self._partial(ts):
                self.state = RecState.ABORTED
                raise
            return self._finish(ts, qs, complete=False,
                                message=f"Incomplète : {self.error} (après {ts[-1]:.1f} s)")
        return self._finish(ts, qs, complete=True)

    def _finish(self, ts, qs, complete: bool, message: str = "") -> RecordingResult:
        traj = Trajectory(self.guided, np.array(ts), np.array(qs))
        init = traj.q[0].tolist() if self.kind == "head" else None
        self.result = RecordingResult(traj, (0.0, float(ts[-1])), self.guided, self.driven, init, complete,
                                      message)
        self.state = RecState.DONE
        return self.result


class ArmRecording(GuidedRecording):
    """Arms pass: arms guided, head stiff in place."""
    kind = "arms"


class HeadRecording(GuidedRecording):
    """Head pass: head guided, arms stiff in place (no arm gesture required)."""
    kind = "head"

    def _groups(self):
        return self.cfg.head_joints, self.cfg.arm_joints


class HandRecording(RecordingSession):
    """Hands pass: arms and head stiff in place, hands DRIVEN live by
    the operator (`set_target`). Nothing is relaxed.

    The target received from the interface is bounded (guardian limits and margin) and followed
    with limited velocity and acceleration: it is this filtered target that
    is sent and kept (`reference`), next to the measurements."""
    kind = "hands"

    def __init__(self, cfg: Config, controller: Controller):
        super().__init__(cfg, controller)
        if not cfg.hand_joints:
            raise ControllerError("Aucune main configurée (joints.yaml, section hands)")
        hj = list(cfg.hand_joints)
        self.hand_joints = hj
        g = cfg.guardian
        self._lo = np.array([cfg.limits[j][0] + g.margin(j) for j in hj])
        self._hi = np.array([cfg.limits[j][1] - g.margin(j) for j in hj])
        self._vmax = np.array([0.8 * g.velocity_limit(j) for j in hj])
        self._amax = np.array([0.8 * g.acceleration_limit(j) for j in hj])
        self.hand_target: Optional[np.ndarray] = None
        self.target_fn: Optional[Callable[[float], dict]] = None  # simulated demonstrator (t -> target)
        self.command: Optional[np.ndarray] = None
        self._vel = np.zeros(len(hj))
        self._last_t = 0.0
        self._ref_rows: List[np.ndarray] = []

    def prepare(self, confirm: bool = False):
        """Stiffens arms, head and hands at their measured posture (no movement, nothing relaxed)."""
        if self.ctrl.state not in (State.IDLE, State.HOLDING):
            raise ControllerError(f"Préparation impossible dans l'état {self.ctrl.state.value}")
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : cocher « Mouvements autorisés » pour piloter les mains")
        self.guided = []
        self.driven = self.cfg.arm_joints + self.cfg.head_joints + self.hand_joints
        self.adapter.set_roles(guided=self.guided, driven=self.driven)
        self._stop.clear()
        self.state = RecState.PREPARED

    def set_target(self, targets: dict):
        """New opening target {LHand|RHand: 0..1}; hands not listed keep theirs."""
        hj = self.hand_joints
        unknown = [j for j in targets if j not in hj]
        if unknown:
            raise ControllerError(f"Mains attendues {hj}, reçu {unknown}")
        base = self.hand_target if self.hand_target is not None else (
            self.command if self.command is not None else self.adapter.read_angles(hj))
        tgt = np.array(base, dtype=float)
        for i, j in enumerate(hj):
            if j in targets:
                v = float(targets[j])
                if not np.isfinite(v):
                    raise ControllerError(f"Ouverture invalide pour {j}")
                tgt[i] = v
        self.hand_target = np.clip(tgt, self._lo, self._hi)

    def _follow(self, t: float):
        hj = self.hand_joints
        if self.target_fn is not None:
            d = self.target_fn(t)
            self.set_target({j: d[j] for j in hj if j in d})
        dt = max(0.0, t - self._last_t)
        self._last_t = t
        q, v = self.command, self._vel
        err = self.hand_target - q
        # target velocity: bounded, and compatible with stopping on the target (braking)
        v_des = np.sign(err) * np.minimum(self._vmax, np.sqrt(2 * self._amax * np.abs(err)))
        v = v + np.clip(v_des - v, -self._amax * dt, self._amax * dt)
        q = np.clip(q + v * dt, self._lo, self._hi)
        self.command, self._vel = q, v
        self.adapter.send_angles(hj, q, self.ctrl.speed)
        self._ref_rows.append(q.copy())

    def run(self, countdown_s: float = 3.0, max_duration_s: float = 15.0) -> RecordingResult:
        if self.state != RecState.PREPARED:
            raise ControllerError("Préparer l'enregistrement d'abord")
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : pilotage des mains impossible")
        a = self.adapter
        hj = self.hand_joints
        self.ctrl.state = State.EXECUTING
        ts, qs = [], []
        try:
            if not self._countdown(countdown_s):
                self.state = RecState.ABORTED
                self.ctrl.state = State.IDLE
                return None
            self.state = RecState.RECORDING
            self.command = np.clip(a.read_angles(hj), self._lo, self._hi)
            if self.hand_target is None:
                self.hand_target = self.command.copy()
            self._vel = np.zeros(len(hj))
            self._last_t = 0.0
            self._ref_rows = []
            t0 = a.now()
            k = 0
            while True:
                t = a.now() - t0
                a.heartbeat()
                self._follow(t)
                q = a.read_angles(hj)
                ts.append(t)
                qs.append(q)
                self.elapsed = t
                if self._stop.is_set() or t >= max_duration_s:
                    break
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        except Exception as e:
            lost = isinstance(e, ConnectionLost)
            if not lost:
                try:
                    a.stop_motion()
                except Exception:
                    pass
            self._interrupted(e, lost)
            if not self._partial(ts):
                self.state = RecState.ABORTED
                raise
            return self._finish(ts, qs, complete=False,
                                message=f"Incomplète : {self.error} (après {ts[-1]:.1f} s)")
        if self.ctrl.state == State.EXECUTING:  # a global stop already put it on hold
            self.ctrl.state = State.IDLE
        return self._finish(ts, qs, complete=True)

    def _finish(self, ts, qs, complete: bool, message: str = "") -> RecordingResult:
        hj = self.hand_joints
        traj = Trajectory(hj, np.array(ts), np.array(qs))
        # target sent with each measurement (same clock); a drop can happen in between
        ref = Trajectory(hj, traj.t.copy(), np.array(self._ref_rows[:len(ts)]))
        self.result = RecordingResult(traj, (0.0, float(ts[-1])), [], self.driven, None, complete, message,
                                      hands_initial=traj.q[0].tolist(), reference=ref)
        self.state = RecState.DONE
        return self.result
