"""Enregistrement des démonstrations en deux passages.

Passage 1 (bras) : bras GUIDÉS (relâchés, lus), tête maintenue raide sur sa
posture mesurée. On enregistre les angles mesurés et leurs horodatages.

Passage 2 (tête) : bras PILOTÉS qui rejouent une prise de bras, tête GUIDÉE.
La même boucle échantillonne les capteurs et envoie les consignes des bras,
sur la même horloge : la fenêtre de relecture (segment) est donc connue
exactement dans le temps de l'enregistrement. On conserve les bras MESURÉS
pendant la relecture, c'est-à-dire le mouvement réellement exécuté.

Aucune articulation n'est à la fois relâchée et commandée.
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


def replay_trajectory(take: Take, joints, ctrl_per_s: float = 6.0, rate_hz: float = 50.0) -> Trajectory:
    """Version lissée et fidèle d'une prise de bras pour la relecture motorisée.

    Les mesures brutes contiennent du bruit capteur qui produirait des
    accélérations artificielles ; on les projette sur une B-spline assez fine
    (6 points de contrôle par seconde par défaut). Les données brutes restent
    intactes sur disque.
    """
    seg = take.segment_trajectory(joints)
    k = max(8, int(np.ceil(seg.duration * ctrl_per_s)) + 4)
    ctrl, dur = encode_trajectory(seg, k, max(100, len(seg.t)))
    return decode(ctrl, dur, joints, rate_hz)


class RecordingSession:
    """Boucle d'enregistrement commune ; exécutable dans un thread (UI) ou en synchrone (tests)."""

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


class ArmRecording(RecordingSession):
    def prepare(self, confirm: bool = False):
        """Relâche les bras (ils tombent si personne ne les tient) et maintient la tête."""
        if getattr(self.adapter, "is_real", False) and not confirm:
            raise ControllerError("Robot réel : confirmer que les bras sont tenus avant de les relâcher")
        if self.ctrl.state not in (State.IDLE, State.HOLDING):
            raise ControllerError(f"Préparation impossible dans l'état {self.ctrl.state.value}")
        self.guided = self.cfg.arm_joints
        self.driven = self.cfg.head_joints
        self.adapter.set_roles(guided=self.guided, driven=self.driven)
        self._stop.clear()
        self.state = RecState.PREPARED

    def run(self, countdown_s: float = 3.0, max_duration_s: float = 15.0) -> RecordingResult:
        if self.state != RecState.PREPARED:
            raise ControllerError("Préparer l'enregistrement d'abord")
        a = self.adapter
        joints = self.guided
        try:
            if not self._countdown(countdown_s):
                self.state = RecState.ABORTED
                return None
            self.state = RecState.RECORDING
            ts, qs = [], []
            t0 = a.now()
            k = 0
            while True:
                t = a.now() - t0
                q = a.read_angles(joints)  # mesures réelles
                ts.append(t)
                qs.append(q)
                self.elapsed = t
                if self._stop.is_set() or t >= max_duration_s:
                    break
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        except ConnectionLost as e:
            self.error = f"Connexion perdue pendant l'enregistrement : {e}"
            self.state = RecState.ABORTED
            raise
        traj = Trajectory(joints, np.array(ts), np.array(qs))
        self.result = RecordingResult(traj, (0.0, float(ts[-1])), self.guided, self.driven, None, True)
        self.state = RecState.DONE
        return self.result


class HeadRecording(RecordingSession):
    """Passage 2. La trajectoire des bras exécutée pendant que l'opérateur guide la
    tête provient soit d'une démonstration enregistrée (`from_take`), soit du
    MODÈLE DES BRAS existant pour une émotion/intensité (`from_generated`).
    Dans les deux cas, ce sont les bras MESURÉS pendant l'exécution qui sont
    conservés et servent à l'apprentissage de la tête."""

    def __init__(self, cfg, controller, replay: Trajectory, event: EventContext, source: dict,
                 pre_roll_s: float = 0.5, post_roll_s: float = 0.5):
        super().__init__(cfg, controller)
        if replay.joints != cfg.arm_joints:
            raise ControllerError("La trajectoire des bras doit couvrir exactement les articulations des bras")
        self.source = source  # {"arm_source": "take"|"generated", ...}
        self.event = event
        self.pre_roll = pre_roll_s
        self.post_roll = post_roll_s
        self.replay = replay.shifted(-replay.t[0])
        self.guided = cfg.head_joints
        self.driven = cfg.arm_joints
        self.approach_plan = None
        self.replay_report = None

    @classmethod
    def from_take(cls, cfg, controller, arm_take: Take, event: EventContext, **kw) -> "HeadRecording":
        if arm_take.meta.kind != "arms":
            raise ControllerError("La relecture exige une prise de bras")
        rp = replay_trajectory(arm_take, cfg.arm_joints)
        return cls(cfg, controller, rp, event, {"arm_source": "take", "arm_take_id": arm_take.meta.take_id,
                                                "arm_gesture_id": arm_take.meta.take_id}, **kw)

    @classmethod
    def from_generated(cls, cfg, controller, engine, emotion: str, intensity: float, event: EventContext,
                       variation: float = 0.0, seed=None, **kw) -> "HeadRecording":
        """Le modèle des bras (figé) produit le geste ; il est validé puis exécuté."""
        from ..data.representation import decode
        import hashlib
        arm_n, dur = engine.plan_arms(emotion, intensity, variation, seed)
        ctrl = engine.arm_norm.inv(arm_n)
        traj = decode(ctrl, dur, cfg.arm_joints, 50.0)
        digest = getattr(engine.arm_model, "digest", None)
        gid = "gen-" + hashlib.sha256(np.round(ctrl, 6).tobytes() + str(dur).encode()).hexdigest()[:12]
        src = {"arm_source": "generated", "arm_take_id": None, "arm_gesture_id": gid,
               "arm_generation": {"arm_checkpoint_digest": digest, "emotion": emotion, "intensity": float(intensity),
                                  "variation": float(variation), "seed": seed, "duration_s": float(dur)}}
        return cls(cfg, controller, traj, event, src, **kw)

    def plan_approach(self):
        """Valide la relecture et planifie l'approche repos/courant -> départ -> 1er point de la relecture."""
        c = self.ctrl
        head_now = c.measured(self.cfg.head_joints)
        self.replay_report = c.guardian.validate(self.replay, context_pose=head_now)
        if not self.replay_report.ok:
            from ..guardian import GuardianRejection
            raise GuardianRejection(self.replay_report)
        arms_now = c.measured(self.cfg.arm_joints)
        from .transitions import min_jerk
        g = self.cfg.guardian
        start = np.array(self.cfg.pose("start", self.cfg.arm_joints))
        q0 = np.array([arms_now[j] for j in self.cfg.arm_joints])
        approach = Trajectory.concat([
            min_jerk(q0, start, self.cfg.arm_joints, g.transition_velocity_rad_s, g.max_acceleration_rad_s2 / 2),
            min_jerk(start, self.replay.q[0], self.cfg.arm_joints, g.transition_velocity_rad_s,
                     g.max_acceleration_rad_s2 / 2)])
        rep = c.guardian.validate(approach, start_state=arms_now, check_duration=False, context_pose=head_now)
        from .controller import Plan
        self.approach_plan = Plan(approach, [("approche", 0.0, approach.duration)], rep)
        return self.approach_plan

    def prepare(self, confirm: bool = False):
        """Amène les bras (motorisés) au début de la relecture, puis relâche la tête."""
        if self.approach_plan is None:
            self.plan_approach()
        if getattr(self.adapter, "is_real", False) and not confirm:
            raise ControllerError("Robot réel : confirmer l'approche motorisée des bras et la tenue de la tête")
        self.ctrl.execute(self.approach_plan)  # exige un contrôleur armé
        self.adapter.set_roles(guided=self.guided, driven=self.driven)
        self._stop.clear()
        self.state = RecState.PREPARED

    def run(self, countdown_s: float = 3.0) -> RecordingResult:
        if self.state != RecState.PREPARED:
            raise ControllerError("Préparer l'enregistrement d'abord")
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : relecture impossible")
        a = self.adapter
        all_j = self.cfg.arm_joints + self.cfg.head_joints
        arm_j = self.cfg.arm_joints
        D = self.replay.duration
        speed = self.ctrl.speed
        complete = True
        msg = ""
        self.ctrl.state = State.EXECUTING
        try:
            if not self._countdown(countdown_s):
                self.state = RecState.ABORTED
                self.ctrl.state = State.IDLE
                return None
            self.state = RecState.RECORDING
            ts, qs = [], []
            seg_start = seg_end = None
            t0 = a.now()
            k = 0
            while True:
                t = a.now() - t0
                u = t - self.pre_roll
                if self._stop.is_set():
                    a.stop_motion()
                    complete = False
                    msg = "Arrêt demandé pendant la relecture : bras maintenus"
                    break
                if 0.0 <= u <= D + 1e-9 or (seg_start is not None and seg_end is None):
                    uc = min(max(u, 0.0), D)
                    a.heartbeat()
                    a.send_angles(arm_j, self.replay.sample(np.array([uc]))[0], speed)
                    if seg_start is None:
                        seg_start = t - uc  # instant correspondant à la phase 0 de la relecture
                    if u >= D:
                        seg_end = seg_start + D
                q = a.read_angles(all_j)
                ts.append(t)
                qs.append(q)
                self.elapsed = t
                if seg_end is not None and t >= seg_end + self.post_roll:
                    break
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        except ConnectionLost as e:
            self.error = f"Connexion perdue pendant la relecture : {e}"
            self.state = RecState.ABORTED
            self.ctrl.state = State.FAULT
            self.ctrl.armed = False
            raise
        except Exception as e:
            self.error = str(e)
            try:
                a.stop_motion()
            except Exception:
                pass
            self.state = RecState.ABORTED
            self.ctrl.state = State.FAULT
            self.ctrl.armed = False
            raise
        self.ctrl.state = State.HOLDING if not complete else State.IDLE
        traj = Trajectory(all_j, np.array(ts), np.array(qs))
        if seg_start is None or seg_end is None:
            complete = False
            seg = (0.0, float(ts[-1]))
        else:
            seg = (float(seg_start), float(min(seg_end, ts[-1])))
        hi = traj.sample(np.array([seg[0]]))[0][len(arm_j):].tolist()
        self.result = RecordingResult(traj, seg, self.guided, self.driven, hi, complete, msg)
        self.state = RecState.DONE
        return self.result
