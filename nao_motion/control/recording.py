"""Enregistrement des démonstrations : un passage par groupe, INDÉPENDANTS.

Chaque passage enregistre un seul groupe pour une consigne (émotion, intensité,
événement) ; aucun ne demande de geste des bras déjà enregistré ni de modèle.

Passage bras : bras GUIDÉS (relâchés, lus), tête maintenue raide sur sa posture
mesurée.

Passage tête : tête GUIDÉE, bras maintenus raides sur leur posture mesurée.
Aucun mouvement motorisé : la préparation ne fait que relâcher la tête.

Passage mains : rien n'est relâché, bras et tête restent raides sur place et
l'opérateur PILOTE l'ouverture des mains depuis l'interface (curseurs, boutons,
clavier) : une main de NAO ne se guide pas à la main. La consigne de l'opérateur
est filtrée en ligne (bornes, vitesse, accélération du guardian) avant d'être
envoyée, et conservée à côté des mesures.

Chaque enregistrement dure jusqu'à « Arrêter » (ou la durée maximale) ; le
segment utile est proposé automatiquement à la sauvegarde.

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
    hands_initial: Optional[List[float]] = None
    reference: Optional[Trajectory] = None   # consignes envoyées (passage mains : ouverture des mains)


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


class GuidedRecording(RecordingSession):
    """Un groupe GUIDÉ à la main (relâché, lu) pendant que l'autre est maintenu raide."""
    kind = "arms"

    def _groups(self):
        return self.cfg.arm_joints, self.cfg.head_joints

    def prepare(self, confirm: bool = False):
        """Relâche le groupe guidé (il tombe si personne ne le tient) et maintient l'autre."""
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
        init = None
        if self.kind == "head":
            init = traj.q[0].tolist()
        self.result = RecordingResult(traj, (0.0, float(ts[-1])), self.guided, self.driven, init, True)
        self.state = RecState.DONE
        return self.result


class ArmRecording(GuidedRecording):
    """Passage bras : bras guidés, tête raide sur place."""
    kind = "arms"


class HeadRecording(GuidedRecording):
    """Passage tête : tête guidée, bras raides sur place (aucun geste des bras requis)."""
    kind = "head"

    def _groups(self):
        return self.cfg.head_joints, self.cfg.arm_joints


class HandRecording(RecordingSession):
    """Passage mains : bras et tête raides sur place, mains PILOTÉES en direct par
    l'opérateur (`set_target`). Rien n'est relâché.

    La cible reçue de l'interface est bornée (limites et marge du guardian) et suivie
    avec une vitesse et une accélération limitées : c'est cette consigne filtrée qui
    est envoyée et conservée (`reference`), à côté des mesures."""
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
        self.target_fn: Optional[Callable[[float], dict]] = None  # démonstrateur simulé (t -> cible)
        self.command: Optional[np.ndarray] = None
        self._vel = np.zeros(len(hj))
        self._last_t = 0.0
        self._ref_rows: List[np.ndarray] = []

    def prepare(self, confirm: bool = False):
        """Raidit bras, tête et mains sur leur posture mesurée (aucun déplacement, rien de relâché)."""
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
        """Nouvelle cible d'ouverture {LHand|RHand: 0..1} ; les mains absentes gardent la leur."""
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
        # vitesse visée : bornée, et compatible avec un arrêt sur la cible (freinage)
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
            ts, qs = [], []
            t0 = a.now()
            k = 0
            while True:
                t = a.now() - t0
                a.heartbeat()
                self._follow(t)
                ts.append(t)
                qs.append(a.read_angles(hj))
                self.elapsed = t
                if self._stop.is_set() or t >= max_duration_s:
                    break
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        except ConnectionLost as e:
            self.error = f"Connexion perdue pendant l'enregistrement : {e}"
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
        if self.ctrl.state == State.EXECUTING:  # un arrêt global l'a déjà mis en maintien
            self.ctrl.state = State.IDLE
        traj = Trajectory(hj, np.array(ts), np.array(qs))
        ref = Trajectory(hj, traj.t.copy(), np.array(self._ref_rows))
        self.result = RecordingResult(traj, (0.0, float(ts[-1])), [], self.driven, None, True,
                                      hands_initial=traj.q[0].tolist(), reference=ref)
        self.state = RecState.DONE
        return self.result
