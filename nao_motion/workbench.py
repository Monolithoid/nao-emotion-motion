"""Service layer shared by the interface, the CLI and the demo.

Groups configuration, storage, adapter, controller, guardian, engine
and collection state. Nothing here triggers a movement without an explicit call.
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
        self.play_info: Optional[dict] = None  # last generated gesture played (Generation tab)
        self.sim_demonstration = self.adapter.name == "sim"
        self.rng = np.random.default_rng(0)
        self.lock = threading.RLock()
        self.calibration = None  # CalibrationWizard ouvert depuis l'interface
        self._hands_pending: Optional[dict] = None  # last hand-opening target (outside recording)
        self._hands_thread: Optional[threading.Thread] = None

    # -- robot connection (address entered in the interface) -------------------
    def _recording_active(self) -> bool:
        """Recording actually in progress. A "recording" state without a live loop
        (thread ended on an error) is cleared here: otherwise it would block any reconnection."""
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
        """Bridge lost: the recording loop will fail at the latest after the bridge
        timeout; we ask it to stop and wait for it (bounded) before reconnecting."""
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
        """State of the link with the bridge, shown by the interface."""
        ad = self.adapter
        if ad.name != "bridge":
            return {"used": False}
        return {"used": True, "address": f"{ad.host}:{ad.port}", "connected": ad.is_connected(),
                "lost": ad.lost_reason, "lost_at": ad.lost_at}

    def use_adapter(self, adapter):
        """Replaces the adapter (simulation <-> bridge). Refused during a movement or a recording."""
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
        self.calibration = None  # tied to the old adapter
        rec = self.recording
        if rec is None or rec.result is None:  # a finished take (even incomplete) is still to be saved
            self.recording = None
        self.engine = None
        self.sim_demonstration = adapter.name == "sim"

    def connect_robot(self, robot_host: str, robot_port: int = 9559, bridge_host: Optional[str] = None,
                      bridge_port: Optional[int] = None, bridge_on_robot: Optional[bool] = None) -> dict:
        """Connects to the robot through the bridge, at the address (IP or DNS name) given by the interface.
        Bridge on the robot (default, `bridge.location: robot`): the interface reaches the bridge at
        the robot's address, and the bridge reaches NAOqi locally. No movement, no stiffness
        change; the controller stays disarmed."""
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
            # same bridge (e.g. after a drop): reconnect in place, the open calibration
            # and its explorations are kept
            return self._reconnect_bridge(old, naoqi_host, robot_port, bridge_on_robot, host)
        if self.calibration is not None:
            raise ControllerError("Calibration ouverte : la fermer avant de changer de robot ou de pont")
        if old.name == "bridge":
            old.disconnect()  # always: the bridge accepts only one client at a time
        ad.connect()
        try:
            # a bridge started with --on-robot is already connected to NAOqi: nothing to redo
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
            self.ctrl.last_error = None  # the drop stays in the log
        self.ctrl._log(f"reconnecté au robot {host} via le même pont (aucun mouvement envoyé)")
        return info

    def use_simulation(self):
        from .adapters.sim import SimAdapter
        self.use_adapter(SimAdapter(self.cfg))
        self.ctrl.connect()

    # -- guardian calibration (Calibration tab) ---------------------------------
    def open_calibration(self):
        """Opens a guided calibration. No movement, no stiffness change."""
        from .calibration_wizard import CalibrationWizard
        if self._busy():
            raise ControllerError("Mouvement ou enregistrement en cours : arrêter d'abord")
        if not self.adapter.is_connected():
            raise ControllerError("Robot non connecté (onglet Robot)")
        if self.ctrl.armed:
            self.ctrl.disarm()  # no motorised movement during a calibration
        if self.calibration is not None:
            self.calibration.stop()
        self.calibration = CalibrationWizard(self)
        return self.calibration.state()

    def close_calibration(self):
        """Closes without saving anything; the joints stay as they are (no return to rest)."""
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
        # a simulated calibration can only be activated in simulation (interface trials)
        return not getattr(self.adapter, "is_real", False)

    def extend_calibration(self, file=None, speeds=False, save=False, include_pending=False) -> dict:
        """Calibration extended with the recorded takes (see calibration_extend.py). Preview by
        default; `save` writes a NEW file in configs/calibration/ (never activated here)."""
        from .calibration import CalibrationError
        from .calibration_extend import extend_with_takes
        d = self.cfg.root / "calibration"
        if not file:
            if self.cfg.calibration is None or self.cfg.calibration.path is None:
                raise CalibrationError("No active calibration: choose the calibration to extend")
            path = self.cfg.calibration.path
        else:
            path = d / Path(str(file)).name
        if not path.exists():
            raise CalibrationError(f"Calibration not found: {path.name}")
        statuses = ("accepted", "pending") if include_pending else ("accepted",)
        cal, preview = extend_with_takes(self.cfg, self.store, path, statuses, bool(speeds))
        if save:
            preview["saved"] = cal.save(d / f"{cal.version}.yaml").name
        return preview

    def activate_calibration(self, file) -> dict:
        """Activates a file from configs/calibration/ (saved during a previous session
        or just now), then reloads the configuration. No movement; if reloading
        fails, the previous activation is restored."""
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
        """Reloads configs/ (after activating a calibration). Controller disarmed, no movement."""
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

    # -- external event --------------------------------------------------------
    def set_event(self, d: dict, ttl_s: Optional[float] = None) -> EventContext:
        """Receives an event (camera/microphone component, or manual test)."""
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
        """Prepares an INDEPENDENT pass: kind = "arms" (guided arms), "head" (guided head)
        or "hands" (driven hands). Prompt: emotion, intensity, current event.
        No prior arm gesture, no model required, no motorised movement."""
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
        """Arms pass: arms relaxed (guided), head stiff in place."""
        return self.prepare_take("arms", emotion, intensity, intention, confirm)

    def prepare_head_take(self, emotion, intensity, intention=None, confirm=False):
        """Head pass: head relaxed (guided), arms stiff in place."""
        return self.prepare_take("head", emotion, intensity, intention, confirm)

    def prepare_hands_take(self, emotion, intensity, intention=None, confirm=False):
        """Hands pass: everything stiff in place, hands driven from the interface (controller armed)."""
        return self.prepare_take("hands", emotion, intensity, intention, confirm)

    # -- hand opening -------------------------------------------------------------
    def set_hands(self, targets: dict, background: bool = True) -> dict:
        """Hand opening {LHand|RHand: 0 (closed) .. 1 (open)}.

        During the hands pass (countdown or recording), the target is followed
        live by the recording loop. Otherwise, it is a short motorised movement:
        validated by the guardian and executed by the controller, which must be armed
        ("Mouvements autorisés"). A new target received during the movement is
        played right after (only the last one received)."""
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
            if self._hands_thread is not None:  # the current movement will continue with this target
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
                self.ctrl.execute(plan)  # guardian refusal -> GuardianRejection, nothing is sent
            except Exception as e:
                self.ctrl.last_error = f"Mains : {e}"
                with self.lock:
                    self._hands_pending = None
                    self._hands_thread = None
                if raise_errors:
                    raise
                return

    def hands_state(self) -> dict:
        """Measured opening and, during the hands pass, current target and prompt. No command."""
        hj = self.cfg.hand_joints
        out = {"joints": hj, "limits": {j: list(self.cfg.limits[j]) for j in hj}, "measured": None}
        rec = self.recording
        if isinstance(rec, HandRecording) and rec.state in (RecState.COUNTDOWN, RecState.RECORDING):
            for k, v in (("target", rec.hand_target), ("command", rec.command)):
                out[k] = None if v is None else dict(zip(hj, np.round(v, 3).tolist()))
            return out  # no extra read during the recording loop
        if hj and self.adapter.is_connected() and not self._recording_active():
            # (during an arms or head pass, the loop has priority over the link)
            try:
                out["measured"] = dict(zip(hj, np.round(self.adapter.read_angles(hj), 3).tolist()))
            except Exception:
                pass
        return out

    def _sim_demonstrator(self, countdown_s):
        """In simulation, a synthetic demonstrator replaces the operator's hand.
        Returns the duration of the simulated gesture (recording stops shortly after)."""
        if not self.sim_demonstration:
            return None
        info, rec = self.rec_info, self.recording
        ev = EventContext.from_dict(info["event"])
        if info["kind"] == "arms":
            fn, dur = arm_gesture(self.cfg, info["emotion"], info["intensity"], self.rng, event=ev)
            self.adapter.set_demonstrator(lambda t: fn(t - countdown_s))
            return dur
        if info["kind"] == "hands":
            # the simulated operator drives the hands (no guided joint)
            dur = float(self.rng.uniform(2.4, 3.4))
            rec.target_fn = hand_behaviour(self.cfg, info["emotion"], info["intensity"], ev, dur, self.rng)
            return dur
        # head: arms stiff in the start position, the head follows its own policy
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
        self._recording_active()  # clears an orphan state

    def save_recording(self, status="pending", notes="", synthetic=None) -> str:
        rec = self.recording
        if rec is None or rec.result is None:
            raise ControllerError("Aucun enregistrement terminé à sauvegarder")
        r, info = rec.result, self.rec_info
        # suggested useful segment: where the group moves (hands: from the driven target)
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
        self.store.save(meta, r.traj, reference=r.reference)  # hand target, same clock
        self.recording = None
        return take_id

    def recording_status(self) -> dict:
        rec = self.recording
        if rec is None:
            return {"state": "none"}
        self._recording_active()  # an orphan state is never shown as "in progress"
        d = {"state": rec.state.value, "countdown_left": round(rec.countdown_left, 2),
             "elapsed": round(rec.elapsed, 2), "error": rec.error, **self.rec_info}
        if rec.result is not None:
            d["complete"] = rec.result.complete
            d["message"] = rec.result.message
        return d

    # -- training ---------------------------------------------------------------
    def train(self, component, epochs=None, include_pending=False, background=True, overrides=None):
        """Trains ONE group (arms, head or hands), independently of the other two.
        `overrides`: settings for this training run (see training.OVERRIDABLE)."""
        overrides = training.check_overrides(overrides)  # immediate error if a setting is invalid
        statuses = ("accepted", "pending") if include_pending else ("accepted",)
        job_id = new_id(f"train-{component}")
        job = {"id": job_id, "component": component, "state": "running", "progress": [], "run": None,
               "error": None, "started": time.time(), "overrides": overrides,
               # planned epochs per stage (shown with the training in progress)
               "epochs": int(overrides.get("epochs") or epochs or self.cfg.model["training"]["epochs"])}
        self.jobs[job_id] = job

        def log(ep, l, lv, stage=None):
            job["progress"].append({"epoch": ep, "train_loss": l, "val_loss": None if np.isnan(lv) else lv,
                                    "stage": stage})
            del job["progress"][:-100]

        def work():
            try:
                run = training.train_component(component, self.store, self.cfg, self.runs_dir, statuses,
                                               epochs, log, overrides)
                job["run"] = str(run)
                job["state"] = "done"
                self.engine = None  # reloaded on the next generation; NO execution
            except Exception as e:
                job["state"] = "error"
                job["error"] = str(e)

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return job

    def runs(self):
        """Training runs in chronological order (oldest first), whatever the group.
        Run folders are named <group>-<YYYYmmdd-HHMMSS-us>: sorting by name grouped them by group."""
        import json
        from .data.trash import active_runs
        active = set(active_runs(self.runs_dir).values())
        out = []
        for p in self.runs_dir.glob("*/metrics.json"):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            name = p.parent.name
            comp, _, stamp = name.partition("-")
            out.append({"run": name, "path": str(p.parent), "component": comp, "stamp": stamp,
                        "active": name in active, "metrics": d.get("metrics", {})})
        out.sort(key=lambda r: (r["stamp"], r["run"]))
        return out

    def run_history(self, run: str, full: bool = False):
        """Loss curve (train/val every 10 epochs) of one run, read from its metrics.json.
        `full`: dict with also the curves of each cross-validation fold and of the final model."""
        import json
        p = (self.runs_dir / run / "metrics.json")
        if "/" in run or "\\" in run or run.startswith(".") or not p.resolve().is_relative_to(self.runs_dir.resolve()) \
                or not p.exists():
            raise ValueError(f"Unknown run: {run}")
        d = json.loads(p.read_text())
        if not full:
            return d.get("history", [])
        return {k: d.get(k, [] if k != "final_history" else None) for k in ("history", "fold_histories", "final_history")}

    # -- generation ---------------------------------------------------------------
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

    def generate(self, emotion, intensity, use_event=True, variation=0.0, seed=None, plan_cycle=True,
                 project_zone=None, temperature=None, latent=None, event=None) -> dict:
        """Generates and validates; moves nothing. Returns an identified plan for explicit execution.
        CVAE: `temperature` (0 = central gesture), `latent` {group: z}; `event` (dict) replaces
        the current event (e.g. replaying a take from the z map with its own event)."""
        eng = self.get_engine()
        if event is not None:
            ev = EventContext.from_dict(event)
        else:
            ev = self.current_event() if use_event else EventContext.none()
        head_now = hands_now = None
        try:
            if self.adapter.is_connected():
                head_now = self.ctrl.measured(self.cfg.head_joints)
                hands_now = self.ctrl.measured(self.cfg.hand_joints) if self.cfg.hand_joints else None
        except Exception:
            head_now = hands_now = None
        if project_zone is None:
            project_zone = bool(self.cfg.robot.get("execution", {}).get("project_control_points", True))
        kw = {}
        if temperature is not None or latent:
            kw = {"temperature": None if temperature is None else float(temperature), "latent": latent or None}
        gen = eng.generate(emotion, intensity, ev, head_now, variation, seed, hands_current=hands_now,
                           project_zone=project_zone, **kw)
        played, adapt = self.playable(gen.combined)
        out = {"generated": gen.to_dict(), "motion_report": self.guardian.validate(played).to_dict()}
        if adapt is not None:
            # the raw model output, its own verdict, and what was changed to make it playable
            out["raw_report"] = self.guardian.validate(gen.combined).to_dict()
            out["adaptation"] = adapt.to_dict()
            out["played"] = played.to_dict()
        plan_id = new_id("plan")
        plan = None
        if plan_cycle and self.adapter.is_connected():
            # preview of the cycle that "Play" would send now; it is planned again from the
            # measured posture when the operator presses Play (the robot may have moved since)
            try:
                plan, start = self.plan_generated(played, "auto", "rest")
                out["cycle"] = dict(plan.to_dict(), start=start, end="rest")
            except ControllerError as e:
                out["cycle_error"] = str(e)
        self.plans[plan_id] = {"gen": gen, "played": played, "plan": plan}
        del_keys = list(self.plans)[:-20]  # keep the last generated gestures only
        for k in del_keys:
            self.plans.pop(k, None)
        out["plan_id"] = plan_id
        return out

    def plan_generated(self, played, start="auto", end="rest"):
        """Full cycle around a generated gesture, from the MEASURED posture, validated as a whole
        by the guardian (factory limits are the absolute ceiling). Nothing is sent here.

        start: "rest" (the robot must be measured at rest), "current" (from the measured
        posture, via the start pose), or "auto" (rest when measured there, else current).
        end: "rest" or "start" (stop in the start pose)."""
        if start not in ("auto", "rest", "current"):
            raise ControllerError("start : 'auto', 'rest' ou 'current'")
        if end not in ("rest", "start"):
            raise ControllerError("end : 'rest' ou 'start'")
        cur = self.ctrl.measured(played.joints)
        if start == "auto":
            far = [j for j in self.ctrl._near(cur, "rest", 2 * self.cfg.guardian.start_tolerance_rad)
                   if not self.cfg.is_hand(j)]
            start = "current" if far else "rest"
        plan = self.ctrl.plan_cycle(played, from_pose=start, to_pose=end)
        return plan, start

    def playable(self, traj):
        """Gesture actually proposed for playback: adapted (slowed down, brought back inside the
        limits) unless robot.yaml execution.adapt_generated is false. The guardian stays the judge."""
        if not bool(self.cfg.robot.get("execution", {}).get("adapt_generated", True)):
            return traj, None
        from .guardian.playable import make_playable
        a = make_playable(self.cfg, traj)
        return a.traj, a

    def execute_plan(self, plan_id, background=True, start=None, end="rest", guardian=True, relax_end=False):
        """Plays a generated gesture on the robot (simulated or real). With start=None the plan
        prepared at generation time is played as is (old behaviour); otherwise the cycle is
        planned again now from the measured posture (see plan_generated). The whole cycle must
        pass the guardian: on a refusal NOTHING is sent.

        The joints of the cycle are stiffened in place just before the first command (the
        other motors are left as they are). relax_end=True cuts them again once the cycle has
        ended normally (not after a stop, which holds the posture). Relaxing during the play is
        always possible (Controller.relax): the play ends there, as "relaxed"."""
        self.check_no_calibration()
        entry = self.plans.get(plan_id)
        if entry is None:
            raise ControllerError("Geste inconnu : générer d'abord")
        if self._busy():
            raise ControllerError("Robot occupé (exécution, enregistrement ou balayage en cours)")
        if not self.adapter.is_connected():
            raise ControllerError("Robot non connecté (onglet Robot)")
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : cocher « moteurs armés » ; aucun mouvement envoyé")
        if self.ctrl.state == State.HOLDING:
            self.ctrl.acknowledge()  # posture held after a stop: the operator asks to move again
        notes = []
        if not guardian:
            # operator's test, at their own risk: the RAW model gesture (no adaptation), the
            # guardian verdict is reported but not applied; factory limits stay the ceiling
            raw = entry["gen"].combined if entry.get("gen") is not None else entry["played"]
            plan, used = self.plan_generated(raw, start or "auto", end)
            plan.traj, clipped = self.ctrl.factory_ceiling(plan.traj)
            if clipped:
                notes.append(f"clipped to the factory limits: {clipped}")
            would = plan.report.to_dict()
            info = {"state": "playing", "plan_id": plan_id, "start": used, "end": end, "guardian": False,
                    "relax_end": bool(relax_end),
                    "guardian_would": {"ok": would["ok"], "violations": len(would["violations"])},
                    "notes": notes, "duration_s": plan.traj.duration, "segments": plan.segments, "at": time.time()}
            return self._run_play(plan, info, background, guardian=False)
        if start is None:
            plan, used = entry["plan"], "rest"
            if plan is None:
                raise ControllerError("Plan inconnu ou non planifiable")
        else:
            plan, used = self.plan_generated(entry["played"], start, end)
        if not plan.report.ok:
            from .guardian import GuardianRejection
            self.play_info = {"state": "refused", "plan_id": plan_id, "start": used, "end": end,
                              "error": "Guardian : cycle refusé, rien n'a été envoyé", "at": time.time()}
            raise GuardianRejection(plan.report)
        info = {"state": "playing", "plan_id": plan_id, "start": used, "end": end, "guardian": True,
                "relax_end": bool(relax_end), "duration_s": plan.traj.duration, "segments": plan.segments, "at": time.time()}
        return self._run_play(plan, info, background)

    def _run_play(self, plan, info, background, guardian=True):
        self.play_info = info

        def work():
            try:
                info["result"] = self.ctrl.execute(plan, guardian=guardian)
                info["warnings"] = list(self.ctrl.tracking_warnings)
                info["state"] = {"stopped": "stopped", "relaxed": "relaxed"}.get(info["result"], "done")
                if info["state"] == "done" and info.get("relax_end"):
                    self.ctrl.relax(plan.traj.joints, confirm=True)
                    info["relaxed_end"] = True
            except Exception as e:
                self.ctrl.last_error = str(e)
                info["state"], info["error"] = "error", str(e)
                if not background:
                    raise
            finally:
                info["ended_at"] = time.time()

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return {"started": True, "duration_s": plan.traj.duration, "start": info["start"], "end": info["end"],
                "guardian": guardian, "notes": info.get("notes", []), "cycle": plan.to_dict()}

    def rest_and_relax(self, background=True) -> dict:
        """Generation tab (key 3): motorised return to rest from the measured posture, then all
        the motors relaxed (arms, head, hands). Joints relaxed before are first stiffened in place
        (no commanded motion) so that the return starts where they are. Requires armed motors.
        Relaxing during the return is possible (Space): it ends there."""
        self.check_no_calibration()
        if self._busy():
            raise ControllerError("Robot occupé (exécution, enregistrement ou balayage en cours)")
        if not self.adapter.is_connected():
            raise ControllerError("Robot non connecté (onglet Robot)")
        if not self.ctrl.armed:
            raise ControllerError("Contrôleur désarmé : cocher « moteurs armés » ; aucun mouvement envoyé")
        if self.ctrl.state == State.HOLDING:
            self.ctrl.acknowledge()
        if self.ctrl.state != State.IDLE:
            raise ControllerError(f"Retour au repos impossible dans l'état {self.ctrl.state.value} (acquitter d'abord)")
        joints = list(self.cfg.all_joints)
        guided = [j for j in joints if self.adapter.roles.get(j) == "guided"]
        if guided:
            self.ctrl.hold(guided)
        plan = self.ctrl.plan_return_to_rest(joints)
        if not plan.report.ok:
            from .guardian import GuardianRejection
            self.play_info = {"state": "refused", "kind": "rest", "plan_id": None,
                              "error": "Guardian : retour au repos refusé, rien n'a été envoyé", "at": time.time()}
            raise GuardianRejection(plan.report)
        info = {"state": "playing", "kind": "rest", "plan_id": None, "relax_end": True,
                "duration_s": plan.traj.duration, "segments": plan.segments, "at": time.time()}
        self.play_info = info
        relax = list(self.cfg.controlled_joints)

        def work():
            try:
                info["result"] = self.ctrl.execute(plan)
                info["state"] = {"stopped": "stopped", "relaxed": "relaxed"}.get(info["result"], "done")
                if info["state"] == "done":
                    self.ctrl.relax(relax, confirm=True)
                    info["relaxed_end"] = True
            except Exception as e:
                self.ctrl.last_error = str(e)
                info["state"], info["error"] = "error", str(e)
                if not background:
                    raise
            finally:
                info["ended_at"] = time.time()

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return {"started": True, "duration_s": plan.traj.duration, "segments": plan.segments}

    def stop(self):
        """Explicit global stop: recording, replay and execution."""
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
                "sim_demonstration": self.sim_demonstration, "play": self.play_info}
