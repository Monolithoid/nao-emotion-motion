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
from .control.recording import ArmRecording, HeadRecording, RecState
from .data.representation import suggest_segment
from .data.store import TakeMeta, TakeStore, new_id
from .engine import MotionEngine
from .events import EventContext
from .guardian import Guardian
from .models.nets import load_model
from .sim.synthetic import arm_gesture, head_behaviour
from .trajectory import Trajectory
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
                                      confidence=d.get("confidence"))
        self.event, self.event_ttl_s = ev, ttl_s
        return ev

    def current_event(self) -> EventContext:
        ev = self.event
        if ev.present and self.event_ttl_s is not None and time.time() - ev.timestamp > self.event_ttl_s:
            return EventContext.none(source="expired")
        return ev

    # -- collecte ---------------------------------------------------------------
    def prepare_arm_take(self, emotion, intensity, intention=None, confirm=False):
        if self.recording is not None and self.recording.state in (RecState.COUNTDOWN, RecState.RECORDING):
            raise ControllerError("Un enregistrement est déjà en cours")
        self.cfg.emotion_index(emotion)
        rec = ArmRecording(self.cfg, self.ctrl)
        rec.prepare(confirm=confirm)
        self.recording = rec
        self.rec_info = {"kind": "arms", "emotion": emotion, "intensity": float(intensity), "intention": intention,
                         "event": self.current_event().to_dict()}
        return self.rec_info

    def prepare_head_take(self, arm_take_id=None, confirm=False, generated: Optional[dict] = None):
        """Passage 2. `generated={"emotion", "intensity", "variation"?, "seed"?}` : le modèle
        des bras existant (figé) produit le geste ; sinon `arm_take_id` : démonstration rejouée."""
        ev = self.current_event()
        if generated is not None:
            g = dict(generated)
            self.cfg.emotion_index(g["emotion"])
            eng = self.get_engine()
            if getattr(eng, "arm_model", None) is None:
                raise ControllerError("Aucun modèle des bras entraîné : entraîner les bras d'abord")
            rec = HeadRecording.from_generated(self.cfg, self.ctrl, eng, g["emotion"], float(g["intensity"]), ev,
                                               float(g.get("variation", 0.0)), g.get("seed"))
            emotion, intensity, intention = g["emotion"], float(g["intensity"]), g.get("intention")
        else:
            arm = self.store.load(arm_take_id)
            rec = HeadRecording.from_take(self.cfg, self.ctrl, arm, ev)
            emotion, intensity, intention = arm.meta.emotion, arm.meta.intensity, arm.meta.intention
        rec.prepare(confirm=confirm)  # approche motorisée validée, contrôleur armé requis
        self.recording = rec
        self.rec_info = {"kind": "head", "emotion": emotion, "intensity": intensity, "intention": intention,
                         "event": ev.to_dict(), **rec.source}
        return {k: v for k, v in self.rec_info.items()}

    def _sim_demonstrator(self, countdown_s):
        """En simulation, un démonstrateur synthétique remplace la main de l'opérateur."""
        if not self.sim_demonstration:
            return None
        info, rec = self.rec_info, self.recording
        if info["kind"] == "arms":
            fn, dur = arm_gesture(self.cfg, info["emotion"], info["intensity"], self.rng)
            self.adapter.set_demonstrator(lambda t: fn(t - countdown_s))
            return dur
        replay = rec.replay
        arm_fn = lambda t: dict(zip(replay.joints, replay.sample(np.array([min(max(t, 0), replay.duration)]))[0]))
        hfn = head_behaviour(self.cfg, info["emotion"], info["intensity"], rec.event, arm_fn, self.rng)
        off = countdown_s + rec.pre_roll
        self.adapter.set_demonstrator(lambda t: hfn(t - off))
        return None

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
                if isinstance(rec, ArmRecording):
                    rec.run(cd, max_duration_s)
                else:
                    rec.run(cd)
            except Exception as e:
                rec.error = f"{e}"
                traceback.print_exc()
            finally:
                if self.sim_demonstration:
                    self.adapter.set_demonstrator(None)

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
            self.rec_thread.join(timeout=5)

    def save_recording(self, status="pending", notes="", synthetic=None) -> str:
        rec = self.recording
        if rec is None or rec.result is None:
            raise ControllerError("Aucun enregistrement terminé à sauvegarder")
        r, info = rec.result, self.rec_info
        if info["kind"] == "arms":
            seg = suggest_segment(r.traj)
            take_id = new_id("arms")
        else:
            seg = r.segment
            take_id = new_id("head")
        meta = TakeMeta(
            take_id=take_id, session_id=self.session_id, kind=info["kind"], emotion=info["emotion"],
            intensity=float(info["intensity"]), joints=r.traj.joints, guided_joints=r.guided,
            driven_joints=r.driven, event=info["event"], calibration_version=self.cfg.calibration_version,
            config_fingerprint=self.cfg.fingerprint(), adapter=self.adapter.name,
            sample_rate_hz=rec.rate, segment=seg, head_initial=r.head_initial,
            arm_take_id=info.get("arm_take_id"), intention=info.get("intention"),
            arm_source=info.get("arm_source"), arm_gesture_id=info.get("arm_gesture_id"),
            arm_generation=info.get("arm_generation"),
            synthetic=self.sim_demonstration if synthetic is None else synthetic, status=status,
            annotations={"notes": notes or ("" if r.complete else r.message), "tags": [] if r.complete else ["incomplete"]},
        )
        ref = rec.replay.shifted(r.segment[0]) if info["kind"] == "head" else None  # consigne, même horloge
        self.store.save(meta, r.traj, reference=ref)
        self.recording = None
        return take_id

    def recording_status(self) -> dict:
        rec = self.recording
        if rec is None:
            return {"state": "none"}
        d = {"state": rec.state.value, "countdown_left": round(rec.countdown_left, 2),
             "elapsed": round(rec.elapsed, 2), "error": rec.error, **self.rec_info}
        if rec.result is not None:
            d["complete"] = rec.result.complete
        return d

    # -- entraînement -----------------------------------------------------------
    def train(self, component, epochs=None, include_pending=False, background=True, arm_checkpoint=None):
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
                if component == "arms":
                    run = training.train_arms(self.store, self.cfg, self.runs_dir, statuses, epochs, log)
                elif component == "head":
                    ac = arm_checkpoint or training.latest(self.runs_dir, "arms")
                    if ac is None:
                        raise training.TrainingError("Entraîner d'abord le modèle des bras")
                    run = training.train_head(self.store, self.cfg, self.runs_dir, ac, statuses, epochs, log)
                else:
                    raise ValueError(component)
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
            if backend == "numpy":
                a = a and a.with_name("model_numpy.npz")
                h = h and h.with_name("model_numpy.npz")
            self.engine = MotionEngine.from_paths(self.cfg, a, h)
        return self.engine

    def generate(self, emotion, intensity, use_event=True, variation=0.0, seed=None, plan_cycle=True) -> dict:
        """Génère et valide ; ne bouge rien. Retourne un plan identifié pour exécution explicite."""
        eng = self.get_engine()
        ev = self.current_event() if use_event else EventContext.none()
        head_now = None
        try:
            head_now = self.ctrl.measured(self.cfg.head_joints) if self.adapter.is_connected() else None
        except Exception:
            head_now = None
        gen = eng.generate(emotion, intensity, ev, head_now, variation, seed)
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
        self.ctrl.stop()

    def status(self) -> dict:
        return {"adapter": self.adapter.name, "real_robot": bool(getattr(self.adapter, "is_real", False)),
                "connected": self.adapter.is_connected(), "controller": self.ctrl.state.value,
                "armed": self.ctrl.armed, "last_error": self.ctrl.last_error,
                "recording": self.recording_status(), "event": self.current_event().to_dict(),
                "roles": self.adapter.roles, "calibrated": self.cfg.calibrated,
                "poses_calibrated": self.cfg.poses_calibrated, "log": self.ctrl.events[-10:],
                "sim_demonstration": self.sim_demonstration}
