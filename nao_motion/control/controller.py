"""Contrôleur d'exécution : planifie le cycle complet, le fait valider, l'exécute.

Principes :
  - Démarre DÉSARMÉ. Aucun mouvement n'est envoyé sans `arm(confirm=True)`
    suivi d'un appel explicite à une méthode d'exécution.
  - Le cycle nominal repos -> départ -> geste -> départ -> repos est construit
    ici (transitions à jerk minimal), puis validé EN ENTIER par le guardian
    avant le premier envoi.
  - Exécution en streaming : à chaque pas, vérification de l'arrêt demandé, de
    la connexion et de l'écart de suivi.
  - Sur arrêt ou erreur : arrêt et MAINTIEN de la posture mesurée quand c'est
    possible. Pas de retour motorisé au repos automatique ; le retour au repos
    (`return_to_rest`) ou le relâchement (`relax`) sont des actions explicites.
  - En cas de perte de connexion, plus rien ne peut être envoyé : c'est le
    chien de garde du pont, côté robot, qui arrête et maintient.
"""
from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..adapters.base import AdapterError, ConnectionLost
from ..config import Config
from ..guardian import Guardian, GuardianRejection, Report
from ..trajectory import Trajectory
from .transitions import boundary_derivatives, min_jerk, quintic_matched


class State(str, enum.Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    EXECUTING = "executing"
    HOLDING = "holding"      # arrêt demandé : posture maintenue
    FAULT = "fault"          # erreur : intervention de l'opérateur requise


class ControllerError(RuntimeError):
    pass


@dataclass
class Plan:
    traj: Trajectory
    segments: List[tuple]            # (nom, t_debut, t_fin)
    report: Report
    motion_report: Optional[Report] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"segments": self.segments, "report": self.report.to_dict(),
                "motion_report": self.motion_report.to_dict() if self.motion_report else None,
                "notes": self.notes, "duration_s": self.traj.duration}


class Controller:
    def __init__(self, cfg: Config, adapter, guardian: Optional[Guardian] = None):
        self.cfg = cfg
        self.adapter = adapter
        self.guardian = guardian or Guardian(cfg)
        self.state = State.DISCONNECTED
        self.armed = False
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        ex = cfg.robot.get("execution", {})
        self.rate = float(ex.get("stream_rate_hz", 25))
        self.speed = float(ex.get("max_fraction_speed", 0.3))
        self.tracking_error = float(ex.get("tracking_error_rad", 0.35))
        # mains : le moteur de la main suit plus lentement qu'une articulation du bras
        self.hand_tracking_error = float(ex.get("hand_tracking_error", 0.5))
        self.events: List[str] = []

    def _log(self, msg):
        self.events.append(msg)
        del self.events[:-200]

    # -- connexion / armement ------------------------------------------------
    def connect(self):
        """Connexion SANS mouvement ni changement de raideur."""
        self.adapter.connect()
        self.state = State.IDLE
        self._log("connecté (aucun mouvement envoyé)")

    def arm(self, confirm: bool = False):
        if not confirm:
            raise ControllerError("Armement refusé : confirmation explicite requise")
        if self.state in (State.DISCONNECTED, State.FAULT):
            raise ControllerError(f"Armement impossible dans l'état {self.state.value}")
        if getattr(self.adapter, "is_real", False):
            if not self.cfg.calibrated or not self.cfg.poses_calibrated:
                raise ControllerError("Robot réel : aucune calibration active (limites et poses mesurées). "
                                      "Lancer `nao-motion calibrate`. Armement refusé.")
            cal_name = (self.cfg.calibration.robot or {}).get("robot_name")
            live_name = self.adapter.capabilities().get("robot_name")
            conf_name = (self.cfg.robot.get("robot") or {}).get("name")
            if conf_name and live_name and conf_name != live_name:
                raise ControllerError(f"Robot configuré « {conf_name} » (.env / robot.yaml), robot connecté "
                                      f"« {live_name} ». Armement refusé.")
            if cal_name and live_name and cal_name != live_name:
                raise ControllerError(f"Calibration faite sur « {cal_name} », robot connecté « {live_name} ». "
                                      "Armement refusé.")
        self.armed = True
        self._log("armé")

    def disarm(self):
        self.armed = False
        self._log("désarmé")

    def stop(self):
        """Arrêt explicite : interrompt l'exécution, maintient la posture, désarme."""
        self._stop.set()
        self.armed = False
        try:
            self.adapter.stop_motion()
        except AdapterError as e:
            self.last_error = f"arrêt : {e}"
        if self.state == State.EXECUTING or self.state == State.IDLE:
            self.state = State.HOLDING
        self._log("ARRÊT demandé : posture maintenue, contrôleur désarmé")

    def acknowledge(self):
        """L'opérateur a constaté l'état du robot ; revient à IDLE si connecté (sans mouvement)."""
        if not self.adapter.is_connected():
            self.state = State.DISCONNECTED
            raise ControllerError("Toujours déconnecté")
        self.state = State.IDLE
        self.last_error = None
        self._stop.clear()

    # -- état mesuré -----------------------------------------------------------
    def measured(self, joints) -> Dict[str, float]:
        return dict(zip(joints, self.adapter.read_angles(joints).tolist()))

    def _near(self, measured: Dict[str, float], pose_name: str, tol: float) -> List[str]:
        ref = dict(zip(measured, self.cfg.pose(pose_name, list(measured))))
        return [j for j in measured if abs(measured[j] - ref[j]) > tol]

    def _transition(self, q0, q1, joints):
        g = self.cfg.guardian
        return min_jerk(q0, q1, joints, g.transition_velocity_rad_s, g.max_acceleration_rad_s2 * 0.5)

    def _rest_transition(self, q0, q1, joints):
        g = self.cfg.guardian
        return min_jerk(q0, q1, joints, g.rest_velocity_rad_s * 0.8, g.max_acceleration_rad_s2 * 0.25)

    def _validate_segments(self, segs, rest_names, start_state, ctx) -> Report:
        """Segments repos <-> départ : validate_rest_transition (le repos peut être hors
        de la zone de mouvement) ; tout le reste : validation complète sur le trajet
        continu qu'ils forment."""
        t, reports, main, main_t0 = 0.0, [], [], None
        for name, s in segs:
            if name in rest_names:
                reports.append(self.guardian.validate_rest_transition(
                    s, start_state=start_state if t == 0 else None, time_offset=t))
            else:
                main_t0 = t if main_t0 is None else main_t0
                main.append(s)
            t += s.duration
        if main:
            r = self.guardian.validate(Trajectory.concat(main), check_duration=False, context_pose=ctx,
                                       start_state=start_state if main_t0 == 0 else None)
            for x in r.violations:
                if x.time_s is not None:
                    x.time_s += main_t0
            reports.insert(0, r)
        rep = reports[0]
        for r in reports[1:]:
            rep.ok = rep.ok and r.ok
            rep.violations += r.violations
        return rep

    # -- planification -----------------------------------------------------
    def plan_cycle(self, motion: Trajectory, from_pose: str = "rest", to_pose: str = "rest",
                   current: Optional[Dict[str, float]] = None) -> Plan:
        """Construit et valide repos -> départ -> geste -> départ -> repos.

        from_pose="rest" exige que le robot soit mesuré au repos. "current" part
        de la posture mesurée (choix explicite de l'opérateur). to_pose peut
        valoir "start" pour s'arrêter en position de départ.
        """
        joints = motion.joints
        rate_hz = 50.0
        motion = motion.shifted(-motion.t[0]).resampled(rate_hz)
        cur = current if current is not None else self.measured(joints)
        q_cur = np.array([cur[j] for j in joints])
        notes = []
        if from_pose == "rest":
            # mains exclues : ouvertes ou fermées, elles rejoignent le départ avec le reste
            far = [j for j in self._near(cur, "rest", 2 * self.cfg.guardian.start_tolerance_rad)
                   if not self.cfg.is_hand(j)]
            if far:
                raise ControllerError(f"Le robot n'est pas mesuré au repos ({far}). "
                                      "Utiliser from_pose='current' en connaissance de cause.")
        elif from_pose != "current":
            raise ControllerError("from_pose : 'rest' ou 'current'")
        start = np.array(self.cfg.pose("start", joints))
        rest = np.array(self.cfg.pose("rest", joints))
        g = self.cfg.guardian
        v0, a0 = boundary_derivatives(motion)
        v1, a1 = boundary_derivatives(motion, at_end=True)
        z = np.zeros(len(joints))
        vlim, alim = g.max_velocity_rad_s * 0.8, g.max_acceleration_rad_s2 * 0.8
        if from_pose == "rest":
            # le repos peut être hors zone (contact) : segment vérifié à part
            segs = [("repos_vers_depart", self._rest_transition(q_cur, start, joints))]
        else:
            segs = [("vers_depart", self._transition(q_cur, start, joints))]
        segs += [("depart_vers_geste", quintic_matched(start, z, z, motion.q[0], v0, a0, joints, vlim, alim)),
                 ("geste", motion),
                 ("geste_vers_depart", quintic_matched(motion.q[-1], v1, a1, start, z, z, joints, vlim, alim))]
        if to_pose == "rest":
            segs.append(("depart_vers_repos", self._rest_transition(start, rest, joints)))
        segments, t = [], 0.0
        for name, s in segs:
            segments.append((name, round(t, 4), round(t + s.duration, 4)))
            t += s.duration
        full = Trajectory.concat([s for _, s in segs])
        others = [j for j in self.cfg.all_joints if j not in joints]
        ctx = self.measured(others) if others and current is None else {}
        motion_report = self.guardian.validate(motion, context_pose=ctx)
        rep = self._validate_segments(segs, {"repos_vers_depart", "depart_vers_repos"}, cur, ctx)
        notes.append("Segments repos <-> départ : limites d'usine et vitesse réduite seulement (le repos peut être "
                     "hors zone) ; leur absence de heurt relève de la calibration.")
        if not motion_report.ok:
            rep.ok = False
            rep.violations = motion_report.violations + rep.violations
        if others:
            notes.append(f"Articulations non commandées (maintenues) : {others}")
        return Plan(full, segments, rep, motion_report, notes)

    def plan_hands_move(self, targets: Dict[str, float]) -> Plan:
        """Ouverture des mains : posture mesurée -> cible {LHand|RHand: 0..1}, profil à jerk
        minimal, validé par le guardian (limites, vitesse, accélération). Rien n'est envoyé ici."""
        joints = [j for j in self.cfg.hand_joints if j in targets]
        unknown = [j for j in targets if not self.cfg.is_hand(j)]
        if unknown or not joints:
            raise ControllerError(f"Mains attendues {self.cfg.hand_joints}, reçu {sorted(targets)}")
        cur = self.measured(joints)
        q0 = np.array([cur[j] for j in joints])
        q1 = np.array([float(targets[j]) for j in joints])
        g = self.cfg.guardian
        v = 0.8 * min(g.velocity_limit(j) for j in joints)
        a = 0.5 * min(g.acceleration_limit(j) for j in joints)
        traj = min_jerk(q0, q1, joints, v, a, min_duration=0.3)
        rep = self.guardian.validate(traj, start_state=cur, check_duration=False)
        return Plan(traj, [("mains", 0.0, round(traj.duration, 4))], rep)

    def plan_return_to_rest(self, joints) -> Plan:
        cur = self.measured(joints)
        q = np.array([cur[j] for j in joints])
        start = np.array(self.cfg.pose("start", joints))
        rest = np.array(self.cfg.pose("rest", joints))
        segs = [("vers_depart", self._transition(q, start, joints)),
                ("depart_vers_repos", self._rest_transition(start, rest, joints))]
        full = Trajectory.concat([s for _, s in segs])
        rep = self._validate_segments(segs, {"depart_vers_repos"}, cur, {})
        segments, t = [], 0.0
        for name, s in segs:
            segments.append((name, round(t, 4), round(t + s.duration, 4)))
            t += s.duration
        return Plan(full, segments, rep)

    # -- exécution -----------------------------------------------------------
    def execute(self, plan: Plan, on_tick: Optional[Callable[[float, np.ndarray], None]] = None) -> str:
        """Exécute un plan validé. Retourne 'done', 'stopped' ou lève en cas de faute."""
        with self._lock:
            if not self.armed:
                raise ControllerError("Contrôleur désarmé : aucun mouvement envoyé")
            if self.state != State.IDLE:
                raise ControllerError(f"Exécution impossible dans l'état {self.state.value}")
            if not plan.report.ok:
                raise GuardianRejection(plan.report)
            self._stop.clear()
            self.state = State.EXECUTING
        joints = plan.traj.joints
        try:
            # revalidation contre l'état mesuré juste avant d'envoyer
            cur = self.measured(joints)
            self.guardian.require(plan.traj, start_state=cur, check_duration=False)
            self.adapter.set_roles(driven=joints)
            result = self._stream(plan.traj, on_tick)
        except ConnectionLost as e:
            self.state = State.FAULT
            self.armed = False
            self.last_error = f"Connexion perdue : {e}. Le pont arrête et maintient côté robot."
            self._log(self.last_error)
            raise
        except GuardianRejection as e:
            self.state = State.IDLE
            self.last_error = str(e)
            raise
        except Exception as e:
            self.last_error = f"Erreur pendant l'exécution : {e}"
            self._safe_hold()
            self.state = State.FAULT
            self.armed = False
            self._log(self.last_error)
            raise
        if result == "stopped":
            self.state = State.HOLDING
        else:
            self.state = State.IDLE
        return result

    def _safe_hold(self):
        try:
            self.adapter.stop_motion()
        except AdapterError:
            pass

    def _stream(self, traj: Trajectory, on_tick=None) -> str:
        a = self.adapter
        dt = 1.0 / self.rate
        t0 = a.now()
        end = traj.t[-1]
        check_every = max(1, int(self.rate / 10))
        tol = np.array([self.hand_tracking_error if self.cfg.is_hand(j) else self.tracking_error
                        for j in traj.joints])
        k = 0
        while True:
            if self._stop.is_set():
                self._safe_hold()
                return "stopped"
            t = a.now() - t0
            tc = min(t, end)
            q = traj.sample(np.array([tc]))[0]
            a.heartbeat()
            a.send_angles(traj.joints, q, self.speed)
            if on_tick is not None:
                on_tick(t, q)
            if k % check_every == 0:
                meas = a.read_angles(traj.joints)
                # comparaison avec la consigne un pas plus tôt (latence de suivi)
                ref = traj.sample(np.array([max(0.0, tc - 2 * dt)]))[0]
                dev = np.abs(meas - ref)
                if np.any(dev > tol):
                    i = int(np.argmax(dev - tol))
                    raise ControllerError(f"Écart de suivi {dev[i]:.2f} sur {traj.joints[i]} > {tol[i]}")
            k += 1
            if t >= end:
                return "done"
            a.sleep(max(0.0, t0 + k * dt - a.now()))

    def return_to_rest(self, confirm: bool = False) -> str:
        """Retour au repos EXPLICITE depuis la posture mesurée (via la position de départ)."""
        if not confirm:
            raise ControllerError("Retour au repos : confirmation explicite requise")
        if self.state == State.HOLDING:
            self.acknowledge()
        joints = [j for j in self.cfg.all_joints if self.adapter.roles.get(j) != "guided"]
        if len(joints) != len(self.cfg.all_joints):
            raise ControllerError("Des articulations sont relâchées : les raidir d'abord (hold)")
        return self.execute(self.plan_return_to_rest(joints))

    def hold(self, joints):
        """Raidit des articulations sur leur posture mesurée (aucun déplacement commandé)."""
        self.adapter.set_roles(driven=list(joints))

    def relax(self, joints, confirm: bool = False):
        if not confirm:
            raise ControllerError("Relâchement : confirmation explicite requise (les bras peuvent tomber)")
        self.adapter.relax(list(joints))
