"""Couche de service commune à l'interface, à la CLI et à la démonstration.

Regroupe configuration, stockage, adaptateur, contrôleur, guardian, moteur
et état de collecte. Rien ici ne déclenche de mouvement sans appel explicite.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .adapters import make_adapter
from .config import Config, load_config
from .control.controller import Controller, ControllerError, State
from .control.recording import ArmRecording, HandRecording, HeadRecording, RecState
from .data.representation import suggest_segment
from .data.store import TakeMeta, TakeStore, new_id
from .engine import MotionEngine
from .events import EventContext
from .guardian import Guardian
from .models.nets import load_model
from .sim.synthetic import arm_gesture, hand_behaviour, head_behaviour
from . import training


class Workbench:
    def __init__(self, data_dir="data", runs_dir="runs", cfg: Optional[Config] = None, adapter=None,
                 adapter_kind: Optional[str] = None, session_id: Optional[str] = None, **adapter_kw):
        self.cfg = cfg or load_config()
        self.store = TakeStore(data_dir, self.cfg)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter or make_adapter(self.cfg, adapter_kind, **adapter_kw)
        self.guardian = Guardian(self.cfg)
        self.ctrl = Controller(self.cfg, self.adapter, self.guardian)
        self.session_id = session_id or new_id("session")
        self.event = EventContext.none()
        self.event_ttl_s: Optional[float] = None
        self.recording = None
        self.rec_info: dict = {}
        self.rec_thread: Optional[threading.Thread] = None
        self.jobs: Dict[str, dict] = {}
        self.engine: Optional[MotionEngine] = None
        self.plans: Dict[str, dict] = {}
        self.sim_demonstration = self.adapter.name == "sim"
        self.rng = np.random.default_rng(0)
        self.lock = threading.RLock()
        self.calibration = None  # CalibrationWizard ouvert depuis l'interface
        self._hands_pending: Optional[dict] = None  # dernière cible d'ouverture des mains (hors enregistrement)
        self._hands_thread: Optional[threading.Thread] = None

    # -- connexion au robot (adresse saisie dans l'interface) -------------------
    def _recording_active(self) -> bool:
        """Enregistrement réellement en cours. Un état « enregistrement » sans boucle vivante
        (fil terminé sur une erreur) est soldé ici : il bloquerait sinon toute reconnexion."""
        rec = self.recording
        if rec is None or rec.state not in (RecState.COUNTDOWN, RecState.RECORDING):
            return False
        th = self.rec_thread
        if th is not None and not th.is_alive() and rec.state in (RecState.COUNTDOWN, RecState.RECORDING):
            rec.state = RecState.DONE if rec.result is not None else RecState.ABORTED
            rec.error = rec.error or "Boucle d'enregistrement interrompue"
            return False
        return True

    def _busy(self) -> bool:
        cal = self.calibration
        return self.ctrl.state == State.EXECUTING or (cal is not None and cal.sweeping) or self._recording_active()

    def _wind_down_after_loss(self):
        """Pont perdu : la boucle d'enregistrement va échouer au plus tard après le délai du
        pont ; on lui demande de s'arrêter et on l'attend (borné) avant de reconnecter."""
        ad = self.adapter
        if ad.name != "bridge" or ad.is_connected():
            return
        if self.recording is not None:
            self.recording.request_stop()
        th = self.rec_thread
        if th is not None and th.is_alive():
            th.join(timeout=float(getattr(ad, "timeout_s", 3.0)) + 2.0)
        self._recording_active()

    def bridge_state(self) -> dict:
        """État de la liaison avec le pont, affiché par l'interface."""
        ad = self.adapter
        if ad.name != "bridge":
            return {"used": False}
        return {"used": True, "address": f"{ad.host}:{ad.port}", "connected": ad.is_connected(),
                "lost": ad.lost_reason, "lost_at": ad.lost_at}

    def use_adapter(self, adapter):
        """Remplace l'adaptateur (simulation <-> pont). Refusé pendant un mouvement ou un enregistrement."""
        self._wind_down_after_loss()
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        try:
            if self.adapter.is_connected():
                self.adapter.disconnect()
        except Exception:
            pass
        self.adapter = adapter
        self.ctrl = Controller(self.cfg, adapter, self.guardian)
        self.calibration = None  # liée à l'ancien adaptateur
        rec = self.recording
        if rec is None or rec.result is None:  # une prise terminée (même incomplète) reste à sauvegarder
            self.recording = None
        self.engine = None
        self.sim_demonstration = adapter.name == "sim"

    def connect_robot(self, robot_host: str, robot_port: int = 9559, bridge_host: Optional[str] = None,
                      bridge_port: Optional[int] = None, bridge_on_robot: Optional[bool] = None) -> dict:
        """Connexion au robot via le pont, à l'adresse (IP ou nom DNS) fournie par l'interface.
        Pont sur le robot (défaut, `bridge.location: robot`) : l'interface joint le pont à
        l'adresse du robot, et le pont joint NAOqi en local. Ni mouvement, ni changement de
        raideur ; le contrôleur reste désarmé."""
        from .adapters.bridge_client import BridgeAdapter
        from .config.hosts import validate_host
        host = validate_host(robot_host)
        if bridge_on_robot is None:
            bridge_on_robot = (self.cfg.robot.get("bridge") or {}).get("location", "robot") == "robot"
        if bridge_on_robot:
            bridge_host, naoqi_host = host, "127.0.0.1"
        else:
            naoqi_host = host
        if bridge_host:
            validate_host(bridge_host)
        ad = BridgeAdapter(self.cfg, host=bridge_host, port=bridge_port)
        self._wind_down_after_loss()
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        old = self.adapter
        if old.name == "bridge" and (old.host, old.port) == (ad.host, ad.port):
            # même pont (ex. après une coupure) : reconnexion en place, la calibration
            # ouverte et ses explorations sont conservées
            return self._reconnect_bridge(old, naoqi_host, robot_port, bridge_on_robot, host)
        if self.calibration is not None:
            raise ControllerError("Calibration ouverte : la fermer avant de changer de robot ou de pont")
        if old.name == "bridge":
            old.disconnect()  # toujours : le pont n'accepte qu'un client à la fois
        ad.connect()
        try:
            # un pont lancé avec --on-robot est déjà connecté à NAOqi : rien à refaire
            info = ad.info if (bridge_on_robot and ad.robot_connected) else ad.connect_robot(naoqi_host, int(robot_port or 9559))
        except Exception:
            ad.disconnect()
            raise
        self.use_adapter(ad)
        self.ctrl.state = State.IDLE
        where = "sur le robot" if bridge_on_robot else "sur ce PC"
        self.ctrl._log(f"connecté au robot {host} via le pont {where} (aucun mouvement envoyé)")
        return info

    def _reconnect_bridge(self, ad, naoqi_host, robot_port, bridge_on_robot, host) -> dict:
        ad.connect()  # ferme l'ancienne socket d'abord
        info = ad.info if (bridge_on_robot and ad.robot_connected) else ad.connect_robot(naoqi_host, int(robot_port or 9559))
        if self.ctrl.state.value in ("disconnected", "fault"):
            self.ctrl.state = State.IDLE
            self.ctrl.last_error = None  # la coupure reste dans le journal
        self.ctrl._log(f"reconnecté au robot {host} via le même pont (aucun mouvement envoyé)")
        return info

    def use_simulation(self):
        from .adapters.sim import SimAdapter
        self.use_adapter(SimAdapter(self.cfg))
        self.ctrl.connect()

    # -- calibration du guardian (onglet Calibration) ---------------------------
    def open_calibration(self):
        """Ouvre une calibration guidée. Aucun mouvement, aucune raideur modifiée."""
        from .calibration_wizard import CalibrationWizard
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        if not self.adapter.is_connected():
            raise ControllerError("Robot non connecté (onglet Robot)")
        if self.ctrl.armed:
            self.ctrl.disarm()  # aucun mouvement motorisé pendant une calibration
        if self.calibration is not None:
            self.calibration.stop()
        self.calibration = CalibrationWizard(self)
        return self.calibration.state()

    def close_calibration(self):
        """Ferme sans rien enregistrer ; les articulations restent dans leur état (pas de retour au repos)."""
        if self.calibration is not None:
            self.calibration.stop()
        self.calibration = None

    def check_no_calibration(self):
        if self.calibration is not None:
            raise ControllerError("Calibration ouverte : la fermer avant toute collecte ou exécution")

    def require_calibration(self):
        if self.calibration is None:
            raise ControllerError("Aucune calibration ouverte")
        return self.calibration

    def list_calibrations(self) -> list:
        from .calibration import list_calibrations
        return list_calibrations(self.cfg.root, allow_simulated=self._sim_adapter())

    def _sim_adapter(self) -> bool:
        # une calibration simulée ne s'active qu'en simulation (essais de l'interface)
        return not getattr(self.adapter, "is_real", False)

    def activate_calibration(self, file) -> dict:
        """Active un fichier de configs/calibration/ (enregistré lors d'une session précédente
        ou à l'instant) puis recharge la configuration. Aucun mouvement ; si le rechargement
        échoue, l'activation précédente est rétablie."""
        from .calibration import activate
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        marker = self.cfg.root / "calibration" / "ACTIVE"
        previous = marker.read_text(encoding="utf-8") if marker.exists() else None
        activate(self.cfg.root, file, allow_simulated=self._sim_adapter())
        try:
            self.reload_config()
        except Exception:
            if previous is None:
                marker.unlink(missing_ok=True)
            else:
                marker.write_text(previous, encoding="utf-8")
            raise
        return {"active": self.cfg.calibration_version, "file": Path(str(file)).name}

    def reload_config(self):
        """Recharge configs/ (après activation d'une calibration). Contrôleur désarmé, aucun mouvement."""
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        cfg = load_config(self.cfg.root)
        state = self.ctrl.state
        self.cfg = cfg
        self.store.cfg = cfg
        self.guardian = Guardian(cfg)
        self.ctrl = Controller(cfg, self.adapter, self.guardian)
        self.ctrl.state = state if state != State.EXECUTING else State.HOLDING
        self.ctrl._log(f"configuration rechargée : calibration {cfg.calibration_version} (aucun mouvement)")
        self.engine = None
        self.plans = {}

    # -- événement extérieur ---------------------------------------------------
    def set_event(self, d: dict, ttl_s: Optional[float] = None) -> EventContext:
        """Reçoit un événement (composant caméra/micros, ou test manuel)."""
        if d.get("frame") == "head" and (d.get("head_yaw_rad") is None or d.get("head_pitch_rad") is None):
            try:
                hy, hp = self.adapter.read_angles(self.cfg.head_joints)
                d = dict(d, head_yaw_rad=float(hy), head_pitch_rad=float(hp))
            except Exception:
                pass
        ev = EventContext.from_source(present=bool(d.get("present")), azimuth=d.get("azimuth"),
                                      elevation=d.get("elevation"), frame=d.get("frame", "torso"),
                                      unit=d.get("unit", "rad"), head_yaw_rad=d.get("head_yaw_rad"),
                                      head_pitch_rad=d.get("head_pitch_rad"), source=d.get("source", "external"),
                                      confidence=d.get("confidence"), distance=d.get("distance_m", d.get("distance")))
        self.event, self.event_ttl_s = ev, ttl_s
        return ev

    def current_event(self) -> EventContext:
        ev = self.event
        if ev.present and self.event_ttl_s is not None and time.time() - ev.timestamp > self.event_ttl_s:
            return EventContext.none(source="expired")
        return ev

    # -- collecte ---------------------------------------------------------------
    def prepare_take(self, kind, emotion, intensity, intention=None, confirm=False):
        """Prépare un passage INDÉPENDANT : kind = "arms" (bras guidés), "head" (tête guidée)
        ou "hands" (mains pilotées). Consigne : émotion, intensité, événement courant.
        Aucun geste des bras préalable, aucun modèle requis, aucun mouvement motorisé."""
        self.check_no_calibration()
        if self._recording_active():
            raise ControllerError("Un enregistrement est déjà en cours")
        if kind not in ("arms", "head", "hands"):
            raise ControllerError(f"Passage inconnu : {kind}")
        self.cfg.emotion_index(emotion)
        intensity = float(intensity)
        if not 0.0 <= intensity <= 1.0:
            raise ControllerError("Intensité hors de [0, 1]")
        if intention is not None and intention not in self.cfg.intentions:
            raise ControllerError(f"Intention inconnue : {intention}")
        rec = {"arms": ArmRecording, "head": HeadRecording, "hands": HandRecording}[kind](self.cfg, self.ctrl)
        rec.prepare(confirm=confirm)
        self.recording = rec
        self.rec_info = {"kind": kind, "emotion": emotion, "intensity": intensity, "intention": intention,
                         "event": self.current_event().to_dict()}
        return dict(self.rec_info)

    def prepare_arm_take(self, emotion, intensity, intention=None, confirm=False):
        """Passage bras : bras relâchés (guidés), tête raide sur place."""
        return self.prepare_take("arms", emotion, intensity, intention, confirm)

    def prepare_head_take(self, emotion, intensity, intention=None, confirm=False):
        """Passage tête : tête relâchée (guidée), bras raides sur place."""
        return self.prepare_take("head", emotion, intensity, intention, confirm)

    def prepare_hands_take(self, emotion, intensity, intention=None, confirm=False):
        """Passage mains : tout raide sur place, mains pilotées depuis l'interface (contrôleur armé)."""
        return self.prepare_take("hands", emotion, intensity, intention, confirm)

    # -- ouverture des mains ------------------------------------------------------
    def set_hands(self, targets: dict, background: bool = True) -> dict:
        """Ouverture des mains {LHand|RHand: 0 (fermée) .. 1 (ouverte)}.

        Pendant le passage mains (compte à rebours ou enregistrement), la cible est suivie en
        direct par la boucle d'enregistrement. Sinon, c'est un mouvement motorisé court :
        validé par le guardian et exécuté par le contrôleur, qui doit être armé
        (« Mouvements autorisés »). Une nouvelle cible reçue pendant le mouvement est
        jouée juste après (la dernière reçue seulement)."""
        targets = {str(j): float(v) for j, v in (targets or {}).items()}
        unknown = [j for j in targets if not self.cfg.is_hand(j)]
        if unknown or not targets:
            raise ControllerError(f"Mains attendues {self.cfg.hand_joints}, reçu {sorted(targets) or 'rien'}")
        rec = self.recording
        if isinstance(rec, HandRecording) and rec.state in (RecState.COUNTDOWN, RecState.RECORDING):
            rec.set_target(targets)
            return {"live": True}
        self.check_no_calibration()
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : cocher « Mouvements autorisés » pour bouger les mains")
        if rec is not None and rec.state in (RecState.COUNTDOWN, RecState.RECORDING):
            raise ControllerError("Enregistrement en cours : les mains ne sont pilotables que pendant le passage mains")
        with self.lock:
            self._hands_pending = dict(self._hands_pending or {}, **targets)
            if self._hands_thread is not None:  # le mouvement en cours enchaînera sur cette cible
                return {"queued": True}
            if self.ctrl.state == State.EXECUTING:
                self._hands_pending = None
                raise ControllerError("Mouvement en cours : attendre la fin ou arrêter")
            if background:
                self._hands_thread = threading.Thread(target=self._hands_worker, daemon=True)
                self._hands_thread.start()
                return {"started": True}
            self._hands_thread = threading.current_thread()
        self._hands_worker(raise_errors=True)
        return {"done": True}

    def _hands_worker(self, raise_errors: bool = False):
        while True:
            with self.lock:
                targets, self._hands_pending = self._hands_pending, None
                if not targets:
                    self._hands_thread = None
                    return
            try:
                plan = self.ctrl.plan_hands_move(targets)
                self.ctrl.execute(plan)  # refus du guardian -> GuardianRejection, rien n'est envoyé
            except Exception as e:
                self.ctrl.last_error = f"Mains : {e}"
                with self.lock:
                    self._hands_pending = None
                    self._hands_thread = None
                if raise_errors:
                    raise
                return

    def hands_state(self) -> dict:
        """Ouverture mesurée et, pendant le passage mains, cible et consigne en cours. Aucune commande."""
        hj = self.cfg.hand_joints
        out = {"joints": hj, "limits": {j: list(self.cfg.limits[j]) for j in hj}, "measured": None}
        rec = self.recording
        if isinstance(rec, HandRecording) and rec.state in (RecState.COUNTDOWN, RecState.RECORDING):
            for k, v in (("target", rec.hand_target), ("command", rec.command)):
                out[k] = None if v is None else dict(zip(hj, np.round(v, 3).tolist()))
            return out  # pas de lecture supplémentaire pendant la boucle d'enregistrement
        if hj and self.adapter.is_connected() and not self._recording_active():
            # (pendant un passage bras ou tête, la boucle a la priorité sur la liaison)
            try:
                out["measured"] = dict(zip(hj, np.round(self.adapter.read_angles(hj), 3).tolist()))
            except Exception:
                pass
        return out

    def _sim_demonstrator(self, countdown_s):
        """En simulation, un démonstrateur synthétique remplace la main de l'opérateur.
        Retourne la durée du geste simulé (l'enregistrement s'arrête peu après)."""
        if not self.sim_demonstration:
            return None
        info, rec = self.rec_info, self.recording
        ev = EventContext.from_dict(info["event"])
        if info["kind"] == "arms":
            fn, dur = arm_gesture(self.cfg, info["emotion"], info["intensity"], self.rng, event=ev)
            self.adapter.set_demonstrator(lambda t: fn(t - countdown_s))
            return dur
        if info["kind"] == "hands":
            # l'opérateur simulé pilote les mains (aucune articulation guidée)
            dur = float(self.rng.uniform(2.4, 3.4))
            rec.target_fn = hand_behaviour(self.cfg, info["emotion"], info["intensity"], ev, dur, self.rng)
            return dur
        # tête : bras raides en position de départ, la tête suit sa propre politique
        start = dict(zip(self.cfg.arm_joints, self.cfg.pose("start", self.cfg.arm_joints)))
        hfn = head_behaviour(self.cfg, info["emotion"], info["intensity"], ev, lambda t: start, self.rng)
        self.adapter.set_demonstrator(lambda t: hfn(t - countdown_s))
        return float(self.rng.uniform(2.0, 3.0))

    def start_recording(self, countdown_s=None, max_duration_s=15.0, background=True):
        rec = self.recording
        if rec is None or rec.state != RecState.PREPARED:
            raise ControllerError("Préparer l'enregistrement d'abord")
        cd = float(self.cfg.robot.get("recording", {}).get("countdown_s", 3) if countdown_s is None else countdown_s)
        sim_dur = self._sim_demonstrator(cd)
        if sim_dur is not None:
            max_duration_s = min(max_duration_s, sim_dur + 0.4)

        def work():
            try:
                rec.run(cd, max_duration_s)
            except Exception as e:
                rec.error = rec.error or f"{e}"
                traceback.print_exc()
            finally:
                if self.sim_demonstration:
                    self.adapter.set_demonstrator(None)
                    if isinstance(rec, HandRecording):
                        rec.target_fn = None

        if background:
            self.rec_thread = threading.Thread(target=work, daemon=True)
            self.rec_thread.start()
        else:
            work()
        return {"countdown_s": cd}

    def stop_recording(self):
        if self.recording is not None:
            self.recording.request_stop()
        if self.rec_thread is not None:
            self.rec_thread.join(timeout=float(getattr(self.adapter, "timeout_s", 3.0)) + 2.0)
        self._recording_active()  # solde un état orphelin

    def save_recording(self, status="pending", notes="", synthetic=None) -> str:
        rec = self.recording
        if rec is None or rec.result is None:
            raise ControllerError("Aucun enregistrement terminé à sauvegarder")
        r, info = rec.result, self.rec_info
        # segment utile proposé : là où le groupe bouge (mains : d'après la consigne pilotée)
        seg = suggest_segment(r.reference if r.reference is not None else r.traj)
        take_id = new_id(info["kind"])
        meta = TakeMeta(
            take_id=take_id, session_id=self.session_id, kind=info["kind"], emotion=info["emotion"],
            intensity=float(info["intensity"]), joints=r.traj.joints, guided_joints=r.guided,
            driven_joints=r.driven, event=info["event"], calibration_version=self.cfg.calibration_version,
            config_fingerprint=self.cfg.fingerprint(), adapter=self.adapter.name,
            sample_rate_hz=rec.rate, segment=seg, head_initial=r.head_initial, hands_initial=r.hands_initial,
            intention=info.get("intention"),
            synthetic=self.sim_demonstration if synthetic is None else synthetic, status=status,
            annotations={"notes": notes or ("" if r.complete else r.message), "tags": [] if r.complete else ["incomplete"]},
        )
        self.store.save(meta, r.traj, reference=r.reference)  # consigne des mains, même horloge
        self.recording = None
        return take_id

    def recording_status(self) -> dict:
        rec = self.recording
        if rec is None:
            return {"state": "none"}
        self._recording_active()  # un état orphelin n'est jamais affiché comme « en cours »
        d = {"state": rec.state.value, "countdown_left": round(rec.countdown_left, 2),
             "elapsed": round(rec.elapsed, 2), "error": rec.error, **self.rec_info}
        if rec.result is not None:
            d["complete"] = rec.result.complete
            d["message"] = rec.result.message
        return d

    # -- entraînement -----------------------------------------------------------
    def train(self, component, epochs=None, include_pending=False, background=True):
        """Entraîne UN groupe (bras, tête ou mains), indépendamment des deux autres."""
        statuses = ("accepted", "pending") if include_pending else ("accepted",)
        job_id = new_id(f"train-{component}")
        job = {"id": job_id, "component": component, "state": "running", "progress": [], "run": None,
               "error": None, "started": time.time()}
        self.jobs[job_id] = job

        def log(ep, l, lv):
            job["progress"].append({"epoch": ep, "train_loss": l, "val_loss": None if np.isnan(lv) else lv})
            del job["progress"][:-100]

        def work():
            try:
                run = training.train_component(component, self.store, self.cfg, self.runs_dir, statuses,
                                               epochs, log)
                job["run"] = str(run)
                job["state"] = "done"
                self.engine = None  # rechargé à la prochaine génération ; AUCUNE exécution
            except Exception as e:
                job["state"] = "error"
                job["error"] = str(e)

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return job

    def runs(self):
        import json
        out = []
        for p in sorted(self.runs_dir.glob("*/metrics.json"), reverse=True):
            d = json.loads(p.read_text())
            out.append({"run": p.parent.name, "path": str(p.parent), "metrics": d["metrics"]})
        return out

    # -- génération ---------------------------------------------------------------
    def get_engine(self) -> MotionEngine:
        if self.engine is None:
            inf = self.cfg.robot.get("inference", {})
            if inf.get("location", "local") == "remote":
                from .inference.client import RemoteEngine
                self.engine = RemoteEngine(self.cfg, inf.get("remote_url"))
                return self.engine
            backend = inf.get("backend", "torch")
            a = training.latest(self.runs_dir, "arms")
            h = training.latest(self.runs_dir, "head")
            m = training.latest(self.runs_dir, "hands")
            if backend == "numpy":
                a = a and a.with_name("model_numpy.npz")
                h = h and h.with_name("model_numpy.npz")
                m = m and m.with_name("model_numpy.npz")
            self.engine = MotionEngine.from_paths(self.cfg, a, h, m)
        return self.engine

    def generate(self, emotion, intensity, use_event=True, variation=0.0, seed=None, plan_cycle=True) -> dict:
        """Génère et valide ; ne bouge rien. Retourne un plan identifié pour exécution explicite."""
        eng = self.get_engine()
        ev = self.current_event() if use_event else EventContext.none()
        head_now = hands_now = None
        try:
            if self.adapter.is_connected():
                head_now = self.ctrl.measured(self.cfg.head_joints)
                hands_now = self.ctrl.measured(self.cfg.hand_joints) if self.cfg.hand_joints else None
        except Exception:
            head_now = hands_now = None
        gen = eng.generate(emotion, intensity, ev, head_now, variation, seed, hands_current=hands_now)
        out = {"generated": gen.to_dict(), "motion_report": self.guardian.validate(gen.combined).to_dict()}
        plan_id = new_id("plan")
        plan = None
        if plan_cycle and self.adapter.is_connected():
            try:
                plan = self.ctrl.plan_cycle(gen.combined, from_pose="rest")
                out["cycle"] = plan.to_dict()
            except ControllerError as e:
                out["cycle_error"] = str(e)
        self.plans[plan_id] = {"gen": gen, "plan": plan}
        out["plan_id"] = plan_id
        return out

    def execute_plan(self, plan_id, background=True):
        self.check_no_calibration()
        entry = self.plans.get(plan_id)
        if entry is None or entry["plan"] is None:
            raise ControllerError("Plan inconnu ou non planifiable")
        plan = entry["plan"]
        if not plan.report.ok:
            from .guardian import GuardianRejection
            raise GuardianRejection(plan.report)
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : aucun mouvement envoyé")

        def work():
            try:
                self.ctrl.execute(plan)
            except Exception as e:
                self.ctrl.last_error = str(e)
                if not background:
                    raise

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return {"started": True, "duration_s": plan.traj.duration}

    def stop(self):
        """Arrêt explicite global : enregistrement, relecture et exécution."""
        if self.recording is not None:
            self.recording.request_stop()
        if self.calibration is not None and self.calibration.sweeping:
            self.calibration.sweep_stop()
        self.ctrl.stop()

    def status(self) -> dict:
        info = getattr(self.adapter, "info", {}) or {}
        return {"adapter": self.adapter.name, "real_robot": bool(getattr(self.adapter, "is_real", False)),
                "robot_address": info.get("robot_address"), "robot_name": info.get("robot_name"),
                "naoqi_version": info.get("naoqi_version"),
                "connected": self.adapter.is_connected(), "bridge": self.bridge_state(),
                "controller": self.ctrl.state.value,
                "armed": self.ctrl.armed, "last_error": self.ctrl.last_error,
                "recording": self.recording_status(), "event": self.current_event().to_dict(),
                "roles": self.adapter.roles, "calibrated": self.cfg.calibrated,
                "poses_calibrated": self.cfg.poses_calibrated, "log": self.ctrl.events[-10:],
                "sim_demonstration": self.sim_demonstration}
