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
from ..guardian import Guardian, GuardianRejection, Report, Violation
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


class TrackingFault(ControllerError):
    """The measured posture stayed far from the command (blocked or pushed joint)."""


class NothingSent(ControllerError):
    """Refusal before the first command: nothing was sent, so the controller stays IDLE
    (a fault would block every later play until it is acknowledged)."""


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
        # per-joint thresholds (e.g. light, fast wrists) and the lag the robot is allowed: the
        # measured angle is compared with the command over the last tracking_lag_s, not with a
        # single past instant (bridge + setAngles latency made fast gestures trip the stop)
        self.tracking_error_per_joint = {str(k): float(v) for k, v in
                                         (ex.get("tracking_error_per_joint") or {}).items()}
        self.tracking_lag = float(ex.get("tracking_lag_s", 0.3))
        self.tracking_factor_guardian_off = float(ex.get("tracking_error_factor_guardian_off", 1.5))
        # NAO tracking is noisy: above the threshold the play only WARNS; it stops when the gap
        # stays above it for tracking_persist_s, or exceeds tracking_stop_rad at any time
        self.tracking_persist = float(ex.get("tracking_persist_s", 0.5))
        self.tracking_stop = float(ex.get("tracking_stop_rad", 0.8))
        self.tracking_warnings: List[str] = []
        # what a tracking fault does: "relax" cuts the motors of the played joints (operator's
        # choice 2026-11-17: safest when a limb is blocked or pushed; arms may drop), "hold"
        # keeps the measured posture stiff
        self.tracking_stop_action = str(ex.get("tracking_stop_action", "relax"))
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
        # per-joint caps (the gesture's own limits: factory speeds or the speeds of the takes),
        # not the global default of guardian.yaml (1.5 rad/s), far below a real gesture
        vlim = np.array([g.velocity_limit(j) for j in joints]) * 0.8
        alim = np.array([g.acceleration_limit(j) for j in joints]) * 0.8
        if from_pose == "rest":
            # le repos peut être hors zone (contact) : segment vérifié à part
            segs = [("repos_vers_depart", self._rest_transition(q_cur, start, joints))]
        else:
            # The measured posture may lie outside the calibrated zone (arms placed by hand,
            # resting on the legs...): like rest -> start, this approach segment is checked on
            # the factory limits at reduced speed, not on the zone (else refused at t = 0).
            segs = [("vers_depart", self._rest_transition(q_cur, start, joints))]
        others = [j for j in self.cfg.all_joints if j not in joints]
        ctx = self.measured(others) if others and current is None else {}

        def in_zone(seg):  # the transition must stay inside the guardian zone (limits, coupled)
            r = self.guardian.validate(seg, check_duration=False, context_pose=ctx)
            return not any(x.code in ("limit", "coupled") for x in r.violations)
        segs += [("depart_vers_geste", quintic_matched(start, z, z, motion.q[0], v0, a0, joints, vlim, alim,
                                                       accept=in_zone)),
                 ("geste", motion),
                 ("geste_vers_depart", quintic_matched(motion.q[-1], v1, a1, start, z, z, joints, vlim, alim,
                                                       accept=in_zone))]
        if to_pose == "rest":
            segs.append(("depart_vers_repos", self._rest_transition(start, rest, joints)))
        segments, t = [], 0.0
        for name, s in segs:
            segments.append((name, round(t, 4), round(t + s.duration, 4)))
            t += s.duration
        full = Trajectory.concat([s for _, s in segs])
        motion_report = self.guardian.validate(motion, context_pose=ctx)
        rep = self._validate_segments(segs, {"repos_vers_depart", "vers_depart", "depart_vers_repos"}, cur, ctx)
        notes.append("Segments posture mesurée / repos <-> départ : limites d'usine et vitesse réduite seulement "
                     "(posture et repos peuvent être hors zone) ; leur absence de heurt relève de la calibration.")
        if not motion_report.ok:
            g0 = next(a for n, a, _ in segments if n == "geste")
            for x in motion_report.violations:
                x.segment = "geste"
                if x.time_s is not None:
                    x.time_s += g0
            rep.ok = False
            rep.violations = motion_report.violations + rep.violations
        for x in rep.violations:
            if x.segment is None and x.time_s is not None:
                x.segment = next((n for n, a, b in segments if a - 1e-6 <= x.time_s <= b + 1e-6), None)
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
        segs = [("vers_depart", self._rest_transition(q, start, joints)),
                ("depart_vers_repos", self._rest_transition(start, rest, joints))]
        full = Trajectory.concat([s for _, s in segs])
        rep = self._validate_segments(segs, {"vers_depart", "depart_vers_repos"}, cur, {})
        segments, t = [], 0.0
        for name, s in segs:
            segments.append((name, round(t, 4), round(t + s.duration, 4)))
            t += s.duration
        return Plan(full, segments, rep)

    # -- exécution -----------------------------------------------------------
    def factory_ceiling(self, traj: Trajectory):
        """Factory limits (rest_limits: factory limits read at calibration when known, else the
        configured limits) = absolute ceiling, kept even with the guardian off. Returns the
        trajectory clipped into them and the list of clipped joints."""
        lo = np.array([self.cfg.rest_limits[j][0] for j in traj.joints])
        hi = np.array([self.cfg.rest_limits[j][1] for j in traj.joints])
        q = np.clip(traj.q, lo, hi)
        clipped = [j for k, j in enumerate(traj.joints) if np.any(q[:, k] != traj.q[:, k])]
        return Trajectory(traj.joints, traj.t.copy(), q), clipped

    def execute(self, plan: Plan, on_tick: Optional[Callable[[float, np.ndarray], None]] = None,
                guardian: bool = True) -> str:
        """Exécute un plan validé. Retourne 'done', 'stopped' ou lève en cas de faute.

        guardian=False (test choisi par l'opérateur, à ses risques) : ni le verdict du guardian
        ni sa revalidation ne sont appliqués ; seules les limites d'usine restent un plafond
        absolu (refus si un point les dépasse) ; l'arrêt sur écart de suivi reste actif."""
        with self._lock:
            if not self.armed:
                raise ControllerError("Contrôleur désarmé : aucun mouvement envoyé")
            if self.state != State.IDLE:
                raise ControllerError(f"Exécution impossible dans l'état {self.state.value}")
            if not guardian:
                _, out = self.factory_ceiling(plan.traj)
                if out:
                    raise ControllerError(f"Hors limites d'usine (plafond absolu) : {out}. Rien n'a été envoyé.")
            elif not plan.report.ok:
                raise GuardianRejection(plan.report)
            self._stop.clear()
            self.state = State.EXECUTING
        joints = plan.traj.joints
        try:
            # revalidation contre l'état mesuré juste avant d'envoyer
            # The plan was validated whole, segment by segment, when it was built (posture /
            # rest <-> start on the factory limits, the rest on the calibrated zone). Checking the
            # concatenated cycle again against the zone here refused every cycle whose rest or
            # measured posture lies outside it. What can have changed since is the robot's
            # posture: its first point must still match the measured state.
            cur = self.measured(joints)
            tol = self.cfg.guardian.start_tolerance_rad * (1 if guardian else 2)
            far = {j: round(abs(cur[j] - float(plan.traj.q[0, k])), 3) for k, j in enumerate(joints)
                   if abs(cur[j] - float(plan.traj.q[0, k])) > tol}
            if far:  # no jump at the first command, guardian or not
                if guardian:
                    raise GuardianRejection(Report(False, [
                        Violation("start_mismatch", "Premier point éloigné de l'état mesuré", joint=j, value=d,
                                  limit=tol, segment="départ") for j, d in far.items()]))
                raise NothingSent(f"Le robot a bougé depuis la planification ({list(far)}) : rejouer. "
                                  "Rien n'a été envoyé.")
            self.adapter.set_roles(driven=joints)
            result = self._stream(plan.traj, on_tick, factor=1.0 if guardian else self.tracking_factor_guardian_off)
        except ConnectionLost as e:
            self.state = State.FAULT
            self.armed = False
            self.last_error = f"Connexion perdue : {e}. Le pont arrête et maintient côté robot."
            self._log(self.last_error)
            raise
        except TrackingFault as e:
            self._safe_hold()
            cut = ""
            if self.tracking_stop_action == "relax":
                try:
                    self.adapter.relax(list(joints))
                    cut = " Moteurs coupés (raideur nulle) sur les articulations jouées."
                except AdapterError as e2:
                    cut = f" Coupure des moteurs impossible ({e2}) : posture maintenue."
            self.last_error = f"{e}.{cut}"
            self.state = State.FAULT
            self.armed = False
            self._log(self.last_error)
            raise
        except (GuardianRejection, NothingSent) as e:
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

    def tracking_tolerance(self, joints, factor: float = 1.0) -> np.ndarray:
        return factor * np.array([self.tracking_error_per_joint.get(
            j, self.hand_tracking_error if self.cfg.is_hand(j) else self.tracking_error) for j in joints])

    def _stream(self, traj: Trajectory, on_tick=None, factor: float = 1.0) -> str:
        a = self.adapter
        dt = 1.0 / self.rate
        t0 = a.now()
        end = traj.t[-1]
        check_every = max(1, int(self.rate / 10))
        tol = self.tracking_tolerance(traj.joints, factor)
        hard = np.maximum(tol, self.tracking_stop * factor)
        lags = np.linspace(0.0, max(self.tracking_lag, 2 * dt), 13)
        since = np.full(len(traj.joints), np.nan)  # time each joint went above its threshold
        worst = np.zeros(len(traj.joints))
        self.tracking_warnings = []
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
                # distance to the commanded path over the last tracking_lag_s (latency of the
                # bridge and of setAngles): a joint that is merely late is not stopped, a blocked
                # or pushed one is
                refs = traj.sample(np.maximum(0.0, tc - lags))
                dev = np.abs(refs - meas[None, :]).min(axis=0)
                over = dev > tol
                since = np.where(over, np.where(np.isnan(since), t, since), np.nan)
                for i in np.nonzero(over & (dev > worst))[0]:
                    worst[i] = dev[i]
                held = np.nan_to_num(t - since, nan=0.0)
                bad = np.nonzero((dev > hard) | (over & (held >= self.tracking_persist)))[0]
                if bad.size:
                    i = int(bad[np.argmax(dev[bad] - tol[bad])])
                    why = (f"> {hard[i]:.2f} (arrêt immédiat)" if dev[i] > hard[i]
                           else f"> {tol[i]:.2f} pendant {held[i]:.1f} s")
                    raise TrackingFault(
                        f"Écart de suivi {dev[i]:.2f} sur {traj.joints[i]} {why} (robot.yaml execution : "
                        f"tracking_error_rad, tracking_error_per_joint, tracking_persist_s, tracking_stop_rad)")
            k += 1
            if t >= end:
                self.tracking_warnings = [f"{j} : écart jusqu'à {w:.2f} rad (seuil {tol[i]:.2f}, toléré)"
                                          for i, (j, w) in enumerate(zip(traj.joints, worst)) if w > 0]
                for w in self.tracking_warnings:
                    self._log("avertissement suivi : " + w)
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
