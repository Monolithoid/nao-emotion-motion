"""Interface locale de collecte (HTTP + page web statique, bibliothèque standard).

Écoute par défaut sur 127.0.0.1 uniquement. Aucune route ne déclenche de
mouvement sans action explicite de l'opérateur (et `confirm: true`).
"""
from __future__ import annotations

import json
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..control.recording import replay_trajectory
from ..data.representation import suggest_segment

STATIC = Path(__file__).parent / "static"


def _downsample(t, q, n=400):
    if len(t) <= n:
        return t.tolist(), q.tolist()
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    return t[idx].tolist(), q[idx].tolist()


class Api:
    def __init__(self, wb):
        self.wb = wb
        if wb.adapter.name == "sim":
            wb.ctrl.connect()  # simulation : connexion sans risque, aucun mouvement

    # GET ---------------------------------------------------------------------
    def get_config(self, q):
        c = self.wb.cfg
        return {"emotions": [{"id": e, "label": c.emotion_labels[e]} for e in c.emotions],
                "intentions": c.intentions, "groups": c.groups, "limits": c.limits, "hands": c.hand_joints,
                "calibration_version": c.calibration_version, "calibrated": c.calibrated,
                "calibration_robot": (c.calibration.robot if c.calibration else None),
                "bridge": c.robot.get("bridge"),
                "poses_calibrated": c.poses_calibrated, "robot": c.robot.get("robot"),
                "events": {"distance_m": [0.3, 3.0], "azimuth_deg": [-90, 90], "elevation_deg": [-30, 45],
                           **(c.robot.get("events") or {})},
                "countdown_s": c.robot.get("recording", {}).get("countdown_s", 3),
                "model": {k: c.model.get(k) for k in ("arm_model", "head_model", "hands_model", "training", "representation")},
                # network tab: what goes in and out of each group's MLP
                "net": {"joints": {"arms": c.arm_joints, "head": c.head_joints, "hands": c.hand_joints},
                        "event_features": ["présent", "sin az", "cos az", "élévation", "distance connue", "proximité"]}}

    def get_status(self, q):
        return self.wb.status()

    def get_takes(self, q):
        """Takes, each with `in_export`: true when `export-dataset` (default filters) would publish it."""
        from ..data.dataset import select_takes
        kind = q.get("kind")
        exported = {m.take_id for m in select_takes(self.wb.store.root)}
        return [{**m.to_dict(), "in_export": m.take_id in exported} for m in self.wb.store.list(kind=kind)]

    def get_take(self, take_id):
        st = self.wb.store
        take = st.load(take_id)
        t, qq = _downsample(take.traj.t, take.traj.q)
        out = {"meta": take.meta.to_dict(), "t": t, "q": qq,
               "suggested_segment": list(suggest_segment(take.traj)),
               "head_takes": [m.to_dict() for m in st.head_takes_for(take_id)] if take.meta.kind == "arms" else [],
               "hand_takes": [m.to_dict() for m in st.head_takes_for(take_id, "hands")] if take.meta.kind == "arms" else []}
        if take.meta.kind == "arms":
            rp = replay_trajectory(take, self.wb.cfg.arm_joints)
            ctx = dict(zip(self.wb.cfg.head_joints, self.wb.cfg.pose("start", self.wb.cfg.head_joints)))
            out["replay"] = {"t": rp.t.tolist(), "q": rp.q.tolist(), "joints": rp.joints,
                             "offset": take.meta.segment[0],
                             "guardian": self.wb.guardian.validate(rp, context_pose=ctx).to_dict()}
        return out

    def get_record_draft(self, q):
        rec = self.wb.recording
        if rec is None or rec.result is None:
            raise ValueError("Aucune prise terminée")
        r = rec.result
        t, qq = _downsample(r.traj.t, r.traj.q)
        seg = r.segment if self.wb.rec_info.get("kind") in ("head", "hands") else suggest_segment(r.traj)
        return {"t": t, "q": qq, "joints": r.traj.joints, "segment": list(seg), "complete": r.complete,
                "message": r.message}

    def get_stats(self, q):
        return self.wb.store.stats()

    def get_progress(self, q):
        """Avancement de la collecte (prises acceptées par groupe, paliers, couverture)."""
        from ..data.progress import progress
        return progress(self.wb.cfg, self.wb.store.list())

    def get_coverage(self, q):
        """Coverage map (emotion x intensity) of a group's accepted takes, with the next take's cell."""
        from ..data.balance import coverage_grid
        kind = q.get("kind")
        if kind not in ("arms", "head", "hands"):
            raise ValueError(f"Unknown pass: {kind}")
        inten = q.get("intensity")
        return coverage_grid(self.wb.cfg, self.wb.store.list(kind=kind), kind, emotion=q.get("emotion"),
                             intensity=float(inten) if inten not in (None, "") else None)

    def get_train(self, q):
        return {"jobs": list(self.wb.jobs.values())[-10:], "runs": self.wb.runs()[-200:]}

    def get_fidelity(self, q):
        """Network tab: how much detail K control points lose on the accepted takes of a group
        (read only, no model, no movement)."""
        from ..data.fidelity import DEFAULT_KS, fidelity
        comp = q.get("component", "arms")
        if comp not in ("arms", "head", "hands"):
            raise ValueError(f"groupe inconnu : {comp}")
        ks = [int(k) for k in q["ks"].split(",")] if q.get("ks") else DEFAULT_KS
        return fidelity(self.wb.store, self.wb.cfg, comp, ks)

    def get_train_history(self, q):
        """Loss curve of one run (for the training results chart)."""
        from urllib.parse import unquote
        return {"run": unquote(q.get("run", "")), **self.wb.run_history(unquote(q.get("run", "")), full=True)}

    # trash of takes and training runs (moving is reversible, emptying is not) ------------------
    def get_trash(self, q):
        from ..data import trash
        takes = trash.list_trashed_takes(self.wb.store.root)
        runs = trash.list_trashed_runs(self.wb.runs_dir)
        return {"takes": takes, "runs": runs, "takes_bytes": sum(t["bytes"] for t in takes),
                "runs_bytes": sum(r["bytes"] for r in runs)}

    def post_trash_takes(self, d):
        """{"ids": [...]}: move takes to data/trash/takes/ (out of training, progress and export)."""
        from ..data import trash
        return trash.trash_takes(self.wb.store.root, d.get("ids") or [])

    def post_trash_runs(self, d):
        """{"runs": [...]}: move trainings to runs/trash/. Trashing an active model makes the
        newest remaining training of its group the active one (`active_changed`)."""
        from ..data import trash
        if any(j["state"] == "running" for j in self.wb.jobs.values()):
            raise ValueError("A training is running: wait for it to end")
        out = trash.trash_runs(self.wb.runs_dir, d.get("runs") or [])
        if out["active_changed"]:
            self.wb.engine = None  # reloaded at the next generation
        return out

    def post_trash_restore(self, d):
        """{"takes": [...], "runs": [...]}: put back what was trashed."""
        from ..data import trash
        out = {}
        if d.get("takes"):
            out.update(trash.restore_takes(self.wb.store.root, d["takes"]))
        if d.get("runs"):
            r = trash.restore_runs(self.wb.runs_dir, d["runs"])
            out["restored_runs"], out["active_changed"] = r["restored"], r["active_changed"]
            if r["active_changed"]:
                self.wb.engine = None
        return out

    def post_trash_empty(self, d):
        """{"what": "takes" | "runs", "items": [...] (optional), "confirm": true}: deletes FOR GOOD."""
        from ..data import trash
        if not d.get("confirm"):
            raise ValueError("Explicit confirmation required: emptying the trash cannot be undone")
        what = d.get("what")
        if what not in ("takes", "runs"):
            raise ValueError(f"Unknown trash: {what}")
        return trash.empty(self.wb.store.root if what == "takes" else None,
                           self.wb.runs_dir if what == "runs" else None, d.get("items"))

    def get_event(self, q):
        return self.wb.current_event().to_dict()

    def get_hands(self, q):
        """Ouverture mesurée des mains (et, pendant le passage mains, cible et consigne). Aucune commande."""
        return self.wb.hands_state()

    def get_calibration(self, q):
        cal = self.wb.calibration
        return cal.state() if cal is not None else {"active": False}

    # POST --------------------------------------------------------------------
    def post_connect(self, d):
        """{"mode": "sim"} ou {"mode": "robot", "robot_host": "nao.local" | "192.168.1.10",
        "robot_port": 9559, "bridge_host"?: ..., "bridge_port"?: ...}. Aucun mouvement."""
        mode = d.get("mode")
        if self.wb.calibration is not None and mode == "sim":
            raise ValueError("Calibration ouverte : la fermer avant de passer en simulation")
        if mode == "robot":
            self.wb.connect_robot(d.get("robot_host"), int(d.get("robot_port") or 9559),
                                  d.get("bridge_host") or None, d.get("bridge_port") or None,
                                  d.get("bridge_on_robot"))
        elif mode == "sim":
            self.wb.use_simulation()
        else:
            self.wb.ctrl.connect()
        return self.wb.status()

    def post_arm(self, d):
        if d.get("armed"):
            self.wb.check_no_calibration()
            self.wb.ctrl.arm(confirm=bool(d.get("confirm")))
        else:
            self.wb.ctrl.disarm()
        return self.wb.status()

    def post_stop(self, d):
        self.wb.stop()
        return self.wb.status()

    def post_acknowledge(self, d):
        self.wb.ctrl.acknowledge()
        return self.wb.status()

    def post_hold(self, d):
        """Raidit sur place (aucun déplacement). Groupes de joints.yaml, plus « hands » pour les mains ;
        sans groupe : tout, mains comprises."""
        c = self.wb.cfg
        groups = d.get("groups") or list(c.groups) + (["hands"] if c.hand_joints else [])
        self.wb.ctrl.hold([j for g in groups for j in (c.hand_joints if g == "hands" else c.groups[g])])
        return self.wb.status()

    def post_relax(self, d):
        groups = d.get("groups") or []
        self.wb.ctrl.relax([j for g in groups for j in self.wb.cfg.groups[g]], confirm=bool(d.get("confirm")))
        return self.wb.status()

    def post_rest(self, d):
        self.wb.ctrl.return_to_rest(confirm=bool(d.get("confirm")))
        return self.wb.status()

    def post_record_prepare(self, d):
        """{"kind": "arms"|"head"|"hands", "emotion", "intensity", "intention"?} : passages indépendants."""
        return self.wb.prepare_take(d.get("kind", "arms"), d["emotion"], float(d["intensity"]),
                                    d.get("intention") or None, confirm=bool(d.get("confirm")))

    def post_record_start(self, d):
        return self.wb.start_recording(d.get("countdown_s"), float(d.get("max_duration_s", 15)))

    def post_record_stop(self, d):
        self.wb.stop_recording()
        return self.wb.recording_status()

    def post_record_save(self, d):
        return {"take_id": self.wb.save_recording(status=d.get("status", "pending"), notes=d.get("notes", ""))}

    def post_record_discard(self, d):
        self.wb.recording = None
        return {"discarded": True}

    def post_hands(self, d):
        """{"targets": {"LHand": 0..1, "RHand": 0..1}} : suivi en direct pendant le passage mains,
        sinon mouvement court validé par le guardian (contrôleur armé requis)."""
        return self.wb.set_hands(d.get("targets") or {})

    def post_take(self, take_id, d):
        m = self.wb.store.update(take_id, status=d.get("status"), notes=d.get("notes"), tags=d.get("tags"),
                                 segment=d.get("segment"), intention=d.get("intention", "__keep__"))
        return m.to_dict()

    def post_draw(self, d):
        """Tirage aléatoire équilibré : propose une consigne, ne commande rien."""
        from ..data.balance import balanced_draw
        kind = d.get("kind")
        if kind not in ("arms", "head", "hands", "gen", None):
            raise ValueError(f"Passage inconnu : {kind}")
        kind = None if kind == "gen" else kind
        metas = self.wb.store.list(kind=kind) if kind else []
        return balanced_draw(self.wb.cfg, metas, kind, allow_none=bool(d.get("allow_none")),
                             event=bool(d.get("event", True)))

    def post_event(self, d):
        ev = self.wb.set_event(d, d.get("ttl_s"))
        return ev.to_dict()

    def post_train(self, d):
        return self.wb.train(d["component"], d.get("epochs"), bool(d.get("include_pending")),
                             overrides=d.get("overrides"))

    def post_generate(self, d):
        emotion = d.get("mix") or d["emotion"]
        out = self.wb.generate(emotion, float(d["intensity"]), bool(d.get("use_event", True)),
                               float(d.get("variation", 0)), d.get("seed"))
        return out

    def post_request(self, d):
        """Point d'entrée pour un LLM extérieur : émotion + intensité -> proposition
        validée, JAMAIS exécutée automatiquement."""
        out = self.post_generate(d)
        out["note"] = "Proposition seulement : l'exécution exige une action explicite de l'opérateur."
        return out

    def post_execute(self, d):
        if not d.get("confirm"):
            raise ValueError("Confirmation explicite requise")
        return self.wb.execute_plan(d["plan_id"])


    # calibration du guardian (aucune route ne commande de mouvement) ----------
    def post_calibration_open(self, d):
        return self.wb.open_calibration()

    def post_calibration_close(self, d):
        self.wb.close_calibration()
        return {"active": False}

    def _cal(self):
        return self.wb.require_calibration()

    def post_calibration_release(self, d):
        self._cal().release(d["group"], confirm=bool(d.get("confirm")))
        return self._cal().state()

    def post_calibration_hold(self, d):
        self._cal().hold(d["group"])
        return self._cal().state()

    def post_calibration_sweep_start(self, d):
        self._cal().sweep_start(d["group"], confirm=bool(d.get("confirm")))
        return self._cal().state()

    def post_calibration_sweep_clear(self, d):
        self._cal().sweep_clear(d["group"])
        return self._cal().state()

    def post_calibration_sweep_stop(self, d):
        self._cal().sweep_stop()
        return self._cal().state()

    def post_calibration_capture(self, d):
        self._cal().capture(d["pose"], d["group"])
        return self._cal().state()

    def post_calibration_compute(self, d):
        self._cal().compute(d.get("version") or None)
        return self._cal().state()

    def post_calibration_save(self, d):
        self._cal().save()
        return self._cal().state()

    def post_calibration_activate(self, d):
        if not d.get("confirm"):
            raise ValueError("Confirmation explicite requise")
        cal = self._cal()
        cal.activate()  # recharge la configuration (Workbench.activate_calibration)
        return cal.state()

    def post_calibration_import(self, d):
        """Reprend un groupe d'une calibration enregistrée dans la calibration ouverte."""
        self._cal().import_group(d["file"], d["group"])
        return self._cal().state()

    def post_calibration_drop_import(self, d):
        self._cal().drop_import(d["group"])
        return self._cal().state()

    def post_calibration_discard(self, d):
        self._cal().discard_draft()
        return self._cal().state()

    # calibrations enregistrées (configs/calibration/) : liste et activation -----
    def get_calibrations(self, q):
        return {"files": self.wb.list_calibrations(), "active": self.wb.cfg.calibration_version,
                "calibrated": self.wb.cfg.calibrated}

    def post_calibrations_extend(self, d):
        """Preview (or save, with save=true) a calibration extended with the recorded takes.
        Nothing is activated and nothing moves."""
        return self.wb.extend_calibration(d.get("file"), bool(d.get("speeds")), bool(d.get("save")))

    def post_calibrations_activate(self, d):
        """Active un fichier déjà enregistré (aucun mouvement, contrôleur laissé tel quel)."""
        self.wb.activate_calibration(d["file"])
        return self.get_calibrations({})


def make_handler(api: Api):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, ctype="application/json; charset=utf-8"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _route(self, method):
            path, _, qs = self.path.partition("?")
            q = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            try:
                if method == "GET" and not path.startswith("/api/"):
                    f = STATIC / ("index.html" if path in ("/", "") else path.lstrip("/"))
                    if not f.resolve().is_relative_to(STATIC.resolve()) or not f.exists():
                        return self._send(404, {"error": "introuvable"})
                    binary = {".png": "image/png", ".ico": "image/x-icon"}
                    if f.suffix in binary:
                        return self._send(200, f.read_bytes(), binary[f.suffix])
                    ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css",
                             "svg": "image/svg+xml"}.get(f.suffix[1:], "text/plain")
                    return self._send(200, f.read_bytes(), ctype + "; charset=utf-8")
                name = path[len("/api/"):]
                d = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length", 0))
                    d = json.loads(self.rfile.read(n) or b"{}") if n else {}
                m = re.fullmatch(r"takes/([\w\-]+)", name)
                if m:
                    res = api.get_take(m.group(1)) if method == "GET" else api.post_take(m.group(1), d)
                else:
                    fn = getattr(api, f"{method.lower()}_{name.replace('/', '_')}", None)
                    if fn is None:
                        return self._send(404, {"error": f"route inconnue {name}"})
                    res = fn(q if method == "GET" else d)
                self._send(200, res)
            except Exception as e:
                traceback.print_exc()
                payload = {"error": str(e)}
                rep = getattr(e, "report", None)
                if rep is not None:
                    payload["report"] = rep.to_dict()
                self._send(400, payload)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")
    return H


def serve(wb, host="127.0.0.1", port=8765):
    api = Api(wb)
    srv = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"Interface : http://{host}:{port}  (adaptateur : {wb.adapter.name}). Aucun mouvement au démarrage.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            wb.stop()
        except Exception:
            pass
