"""Ligne de commande `nao-motion`.

Aucune commande ne met le robot en mouvement sans option explicite
(`--execute` + `--confirm`), et jamais à la fin d'un entraînement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _wb(a, **kw):
    from .workbench import Workbench
    from .config import load_config
    cfg = load_config(a.config) if a.config else None
    return Workbench(a.data, a.runs, cfg=cfg, adapter_kind=kw.pop("adapter", None) or getattr(a, "adapter", None),
                     **kw)


def _connect(ad, cfg, a):
    """Connecte l'adaptateur ; pour le pont, transmet l'adresse du robot (IP ou DNS)."""
    ad.connect()
    _connect_robot(ad, cfg, a)


def _connect_robot(ad, cfg, a):
    if ad.name == "bridge" and not ad.info.get("robot_connected"):
        host = getattr(a, "robot", None) or (cfg.robot.get("robot") or {}).get("ip")
        if not host:
            raise SystemExit("Adresse du robot inconnue : --robot <IP ou nom DNS> (ou NAO_IP dans .env)")
        ad.connect_robot(host, int((cfg.robot.get("robot") or {}).get("port") or 9559))


def cmd_check_config(a):
    from .config import load_config
    from .guardian import Guardian
    from .trajectory import Trajectory
    cfg = load_config(a.config)
    g = Guardian(cfg)
    print(f"Configuration : {cfg.root}")
    print(f"  calibration : {cfg.calibration_version} (active : {cfg.calibrated}) ; poses calibrées : {cfg.poses_calibrated}"
          + (f" ; robot {cfg.calibration.robot.get('robot_name')}" if cfg.calibration else ""))
    print(f"  articulations : {cfg.all_joints}")
    print(f"  émotions : {cfg.emotions} ; intentions : {cfg.intentions}")
    ok = True
    for name in ("rest", "start"):
        q = np.array(cfg.pose(name, cfg.all_joints))
        rep = g.validate(Trajectory(cfg.all_joints, [0, 1], [q, q]), check_duration=False)
        print(f"  pose {name} : {rep.summary()}")
        for v in rep.violations:
            print("    -", v.to_dict())
        ok &= rep.ok
    if not cfg.calibrated or not cfg.poses_calibrated:
        print("  ATTENTION : aucune calibration active (nao-motion calibrate) ; l'exécution sur robot réel sera refusée.")
    r = cfg.robot["robot"]
    if cfg.robot.get("_env_sources"):
        print(f"  surchargé par .env / environnement : {cfg.robot['_env_sources']}")
    print(f"  robot : {r.get('name')} {r.get('model')} NAOqi {r.get('naoqi_version')} @ {r.get('ip')}:{r.get('port')}")
    missing = [k for k in ("model", "naoqi_version", "ip") if not r.get(k)]
    if missing:
        print(f"  Robot réel : informations à renseigner dans robot.yaml : {missing}")
    return 0 if ok else 1


def cmd_ui(a):
    from .ui.server import serve
    serve(_wb(a), a.host, a.port)


def cmd_demo(a):
    from .demo import run_demo
    res = run_demo(a.workdir, epochs=a.epochs)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("metrics")}, indent=2, ensure_ascii=False))


def cmd_synth(a):
    from .demo import collect_synthetic
    wb = _wb(a, adapter="sim", virtual_time=True, sensor_noise_rad=0.002)
    wb.ctrl.connect()
    collect_synthetic(wb, reps=a.reps, intensities=tuple(a.intensities))
    print("Données SYNTHÉTIQUES : utiles pour vérifier la chaîne technique uniquement.")


def cmd_takes(a):
    from .data.store import TakeStore
    from .config import load_config
    st = TakeStore(a.data, load_config(a.config))
    for m in st.list(kind=a.kind):
        ev = m.event_ctx
        evs = "aucun" if not ev.present else f"az={np.degrees(ev.azimuth_rad):.0f}° el={np.degrees(ev.elevation_rad):.0f}°"
        print(f"{m.take_id}  {m.kind:4s} {m.emotion:10s} {m.intensity:.2f} {m.status:8s} {m.duration_s:5.2f}s "
              f"évt:{evs} {'bras:' + m.arm_take_id if m.arm_take_id else ''} {'[synth]' if m.synthetic else ''}")
    if a.stats:
        print(json.dumps(st.stats(), indent=1, ensure_ascii=False))


def cmd_train(a):
    wb = _wb(a, adapter="sim")
    job = wb.train(a.component, epochs=a.epochs, include_pending=a.include_pending, background=False,
                   arm_checkpoint=a.arm_checkpoint)
    if job["state"] != "done":
        print("Échec :", job["error"], file=sys.stderr)
        return 1
    m = json.loads((Path(job["run"]) / "metrics.json").read_text())["metrics"]
    print(f"Run : {job['run']}")
    print(json.dumps(m, indent=1, ensure_ascii=False))
    print("Aucun mouvement n'est déclenché par l'entraînement.")


def _event_from_args(a):
    if a.event_az is None:
        return {"present": False, "source": "cli"}
    return {"present": True, "azimuth": a.event_az, "elevation": a.event_el or 0.0, "unit": "deg",
            "frame": a.event_frame, "source": "cli"}


def cmd_generate(a):
    from .events import EventContext
    wb = _wb(a, adapter="sim", virtual_time=True)
    wb.ctrl.connect()
    wb.set_event(_event_from_args(a))
    emotion = json.loads(a.mix) if a.mix else a.emotion
    out = wb.generate(emotion, a.intensity, variation=a.variation, seed=a.seed)
    g = out["generated"]
    print(f"Geste : {g['duration_s']:.2f} s ; guardian (geste) : {'accepté' if out['motion_report']['ok'] else 'refusé'}")
    for v in out["motion_report"]["violations"]:
        print("  -", v)
    if "cycle" in out:
        print(f"Cycle repos->départ->geste->départ->repos : {'accepté' if out['cycle']['report']['ok'] else 'refusé'}")
    for n in g["notes"]:
        print("  note :", n)
    if a.out:
        Path(a.out).write_text(json.dumps(out["generated"], indent=1, ensure_ascii=False))
        print(f"Trajectoire écrite dans {a.out}")
    if a.simulate:
        if not out.get("cycle", {}).get("report", {}).get("ok"):
            print("Exécution simulée non lancée : cycle refusé par le guardian.")
            return 1
        wb.ctrl.arm(confirm=True)
        wb.execute_plan(out["plan_id"], background=False)
        print(f"Exécution SIMULÉE terminée : {wb.ctrl.state.value}")


def cmd_execute(a):
    """Exécute une trajectoire sauvegardée : seul chemin vers un robot réel en CLI."""
    from .trajectory import Trajectory
    if not a.confirm:
        print("Refus : ajouter --confirm pour autoriser un mouvement.", file=sys.stderr)
        return 2
    wb = _wb(a, adapter=a.adapter)
    traj = Trajectory.from_dict(json.loads(Path(a.plan).read_text())["trajectory"])
    wb.ctrl.connect()
    _connect_robot(wb.adapter, wb.cfg, a)
    plan = wb.ctrl.plan_cycle(traj, from_pose="rest")
    print(plan.report.summary())
    if not plan.report.ok:
        for v in plan.report.violations[:20]:
            print("  -", v.to_dict())
        return 1
    wb.ctrl.arm(confirm=True)
    try:
        print("Exécution… Ctrl+C pour arrêter (posture maintenue, pas de retour au repos automatique).")
        wb.ctrl.execute(plan)
    except KeyboardInterrupt:
        wb.ctrl.stop()
        print("Arrêt : posture maintenue.")
    print("État :", wb.ctrl.state.value)


def cmd_capture_pose(a):
    """Lit la posture mesurée (aucune commande) et affiche un extrait YAML pour poses.yaml."""
    wb = _wb(a, adapter=a.adapter)
    _connect(wb.adapter, wb.cfg, a)
    js = wb.cfg.groups[a.group]
    q = wb.adapter.read_angles(js)
    print(f"# {a.name}/{a.group} mesuré ({wb.adapter.name})")
    print(f"  {a.group}: {{" + ", ".join(f"{j}: {v:.4f}" for j, v in zip(js, q)) + "}")


def cmd_robot_info(a):
    """Identité et limites d'usine du robot via le pont : lecture seule, aucun mouvement."""
    from .adapters import make_adapter
    from .config import load_config
    cfg = load_config(a.config)
    ad = make_adapter(cfg, a.adapter)
    _connect(ad, cfg, a)
    caps = ad.capabilities()
    print(json.dumps({k: v for k, v in caps.items()}, indent=1, ensure_ascii=False, default=str))
    fl = getattr(ad, "factory_limits", None)
    if fl:
        print("Limites d'usine (ALMotion.getLimits) : [min, max, vitesse max, couple max]")
        for j, l in fl(cfg.all_joints).items():
            print(f"  {j:15s} {l}")
    q = ad.read_angles(cfg.all_joints)
    print("Posture mesurée :", {j: round(float(v), 3) for j, v in zip(cfg.all_joints, q)})
    ad.disconnect()


def _ask(msg):
    return input(msg).strip().lower()


def cmd_calibrate(a):
    """Procédure de calibration du guardian (voir nao_motion/calibration.py)."""
    import threading
    from .adapters import make_adapter
    from .calibration import CalibrationSession, activate, load_procedure
    from .config import load_config
    cfg = load_config(a.config, use_calibration=False)  # on mesure sans s'appuyer sur une calibration existante
    proc = load_procedure(cfg.root)
    if a.adapter == "sim":
        from .demo import simulate_calibration
        from .workbench import Workbench
        import tempfile
        wb = Workbench(tempfile.mkdtemp(), tempfile.mkdtemp(), cfg=cfg, adapter_kind="sim", virtual_time=True)
        wb.ctrl.connect()
        cal = simulate_calibration(wb)
        out = Path(a.out or "calibration-sim.yaml")
        cal.save(out)
        print(f"Calibration SIMULÉE écrite dans {out} (aucune valeur réelle ; non activée).")
        return 0
    ad = make_adapter(cfg, a.adapter)
    _connect(ad, cfg, a)
    sess = CalibrationSession(cfg, ad, proc)
    print("Robot :", sess.robot_info())
    print("Rien ne sera commandé : chaque groupe exploré est relâché, les autres sont raidis sur place.")
    print("Placez le robot dans une posture stable (assis/accroupi) avant de commencer.")
    for g in cfg.groups:
        if _ask(f"\n[{g}] Tenez le groupe. Entrée pour le relâcher et commencer l'exploration (q pour quitter) ") == "q":
            return 1
        th = threading.Thread(target=sess.sweep, args=(g,), kwargs={"max_duration_s": 600, "confirm": True}, daemon=True)
        th.start()
        print(f"  Amenez lentement chaque articulation de {g} jusqu'à ses butées utiles, "
              "et parcourez les combinaisons (ex. tête tournée ET baissée).")
        input("  Entrée pour terminer l'exploration de ce groupe ")
        sess.request_stop()
        th.join()
        q = sess.sweeps[g]["q"]
        for i, j in enumerate(cfg.groups[g]):
            print(f"  {j:15s} min {q[:, i].min():+.3f}  max {q[:, i].max():+.3f} rad ({len(q)} mesures)")
        ad.set_roles(driven=cfg.groups[g])  # raidi sur place, sans déplacement
    for name in ("rest", "start"):
        for g in cfg.groups:
            input(f"\n[{name}/{g}] Entrée pour relâcher {g} et le placer en position « {name} » ")
            ad.set_roles(guided=cfg.groups[g])
            input("  Placé ? Entrée pour capturer (le groupe sera raidi sur place) ")
            print("  ", sess.capture_pose(name, g))
            ad.set_roles(driven=cfg.groups[g])
    cal = sess.result(version=a.version)
    print(f"\nCalibration {cal.version} : {len(cal.limits)} articulations, {len(cal.coupled)} contraintes couplées")
    for j, (lo, hi) in cal.limits.items():
        m = cal.measured[j]
        print(f"  {j:15s} mesuré [{m['min']:+.3f}, {m['max']:+.3f}] usine {m['factory']} -> guardian [{lo:+.3f}, {hi:+.3f}]")
    if _ask("Enregistrer cette calibration ? [o/N] ") != "o":
        print("Non enregistrée.")
        return 1
    path = cal.save(cfg.root / "calibration" / f"{cal.version}.yaml")
    print(f"Écrite : {path}")
    if _ask("L'activer pour le guardian ? [o/N] ") == "o":
        print("Active :", activate(cfg.root, path))
    return 0


def cmd_calibration_activate(a):
    from .calibration import activate
    from .config import load_config
    cfg = load_config(a.config, use_calibration=False)
    print("Active :", activate(cfg.root, a.file))
    load_config(a.config)  # vérifie que poses et limites sont cohérentes


def cmd_serve_inference(a):
    from .inference.server import serve
    serve(_wb(a, adapter="sim"), a.host, a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="nao-motion", description="Mouvements expressifs NAO (émotion + intensité)")
    p.add_argument("--config", default=None, help="dossier de configuration (défaut : configs/)")
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-config"); s.set_defaults(fn=cmd_check_config)
    s = sub.add_parser("ui", help="interface locale de collecte")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--adapter", default=None, choices=["sim", "bridge"]); s.set_defaults(fn=cmd_ui)
    s = sub.add_parser("demo", help="parcours complet en simulation (données synthétiques)")
    s.add_argument("--workdir", default="demo_sim"); s.add_argument("--epochs", type=int, default=None)
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("synth", help="remplir data/ avec des prises SYNTHÉTIQUES (simulation)")
    s.add_argument("--reps", type=int, default=2)
    s.add_argument("--intensities", type=float, nargs="+", default=[0.3, 0.7, 1.0]); s.set_defaults(fn=cmd_synth)
    s = sub.add_parser("takes", help="lister les prises")
    s.add_argument("--kind", choices=["arms", "head"]); s.add_argument("--stats", action="store_true")
    s.set_defaults(fn=cmd_takes)
    s = sub.add_parser("train", help="entraîner arms ou head (sans aucun mouvement)")
    s.add_argument("component", choices=["arms", "head"]); s.add_argument("--epochs", type=int)
    s.add_argument("--include-pending", action="store_true"); s.add_argument("--arm-checkpoint")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("generate", help="générer et valider (simulation uniquement)")
    s.add_argument("--emotion", default="joie"); s.add_argument("--mix", help='mélange JSON ex. {"joie":0.5,"surprise":0.5}')
    s.add_argument("--intensity", type=float, default=0.7)
    s.add_argument("--event-az", type=float, help="azimut en degrés (gauche positif) ; absent = pas d'événement")
    s.add_argument("--event-el", type=float, default=0.0); s.add_argument("--event-frame", default="torso",
                                                                         choices=["torso", "head"])
    s.add_argument("--variation", type=float, default=0.0); s.add_argument("--seed", type=int)
    s.add_argument("--out"); s.add_argument("--simulate", action="store_true", help="exécuter sur le robot simulé")
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("execute", help="exécuter une trajectoire sauvegardée (robot réel possible)")
    s.add_argument("plan"); s.add_argument("--adapter", default="sim", choices=["sim", "bridge"])
    s.add_argument("--robot", help="adresse du robot : IP ou nom DNS")
    s.add_argument("--confirm", action="store_true"); s.set_defaults(fn=cmd_execute)
    s = sub.add_parser("capture-pose", help="lire une posture mesurée (aucune commande)")
    s.add_argument("--group", required=True, choices=["left_arm", "right_arm", "head"])
    s.add_argument("--name", default="start", choices=["rest", "start"])
    s.add_argument("--adapter", default="sim", choices=["sim", "bridge"])
    s.add_argument("--robot", help="adresse du robot : IP ou nom DNS"); s.set_defaults(fn=cmd_capture_pose)
    s = sub.add_parser("robot-info", help="identité, version NAOqi, limites d'usine (lecture seule)")
    s.add_argument("--adapter", default="bridge", choices=["sim", "bridge"])
    s.add_argument("--robot", help="adresse du robot : IP ou nom DNS (défaut : NAO_IP)"); s.set_defaults(fn=cmd_robot_info)
    s = sub.add_parser("calibrate", help="procédure de calibration du guardian (exploration guidée)")
    s.add_argument("--adapter", default="bridge", choices=["sim", "bridge"])
    s.add_argument("--robot", help="adresse du robot : IP ou nom DNS (défaut : NAO_IP)")
    s.add_argument("--version", help="nom de la calibration (défaut : <robot>-<date>)")
    s.add_argument("--out", help="sim uniquement : fichier de sortie"); s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("calibration-activate", help="activer un fichier de configs/calibration/")
    s.add_argument("file"); s.set_defaults(fn=cmd_calibration_activate)
    s = sub.add_parser("serve-inference", help="serveur HTTP d'inférence (génération sans exécution)")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8766)
    s.set_defaults(fn=cmd_serve_inference)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
