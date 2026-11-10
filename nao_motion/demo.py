"""Parcours complet en simulation (temps virtuel), données SYNTHÉTIQUES.

0. calibration simulée (exploration guidée -> limites et enveloppes) ;
1. collecter et sauvegarder des démonstrations des bras ;
2. les consulter et les rejouer virtuellement (validation guardian) ;
3. entraîner le modèle des bras ;
4. générer et valider une trajectoire, l'exécuter sur le robot simulé ;
5. collecte tête, INDÉPENDANTE des bras : l'opérateur (ici simulé) guide la
   tête pour une émotion, une intensité et un événement, bras raides sur place ;
6. entraîner la tête seule et exécuter le geste combiné bras + tête ;
7. collecte mains, indépendante : l'opérateur (simulé) pilote l'ouverture des
   mains ; entraîner les mains seules, geste combiné bras + tête + mains.

Les données synthétiques vérifient le fonctionnement technique, pas l'expressivité.
"""
from __future__ import annotations

import json
from pathlib import Path

from .control.recording import replay_trajectory
from .events import EventContext
from .workbench import Workbench

DEFAULT_CONTEXTS = [EventContext.none(), EventContext(True, 0.7, 0.1, source="test", distance_m=2.5),
                    EventContext(True, -0.6, 0.0, source="test", distance_m=0.5),
                    EventContext(True, 0.2, 0.2, source="test")]  # distance inconnue


ARM_CONTEXTS = [EventContext.none(), EventContext(True, 0.7, 0.2, source="test", distance_m=0.5),
                EventContext(True, -0.6, 0.0, source="test", distance_m=2.5)]


def collect_arms(wb: Workbench, reps=2, intensities=(0.3, 0.7, 1.0), contexts=None, log=print):
    """Une prise par (émotion, intensité, répétition) ; l'événement change d'une
    prise à l'autre (sans, à gauche proche, à droite loin...)."""
    contexts = contexts or ARM_CONTEXTS
    arm_ids = []
    for e in wb.cfg.emotions:
        for i in intensities:
            for r in range(reps):
                wb.event = contexts[len(arm_ids) % len(contexts)]
                wb.prepare_arm_take(e, i)
                wb.start_recording(countdown_s=0.3, background=False)
                arm_ids.append(wb.save_recording(status="accepted", notes="synthétique"))
    wb.event = EventContext.none()
    log(f"  {len(arm_ids)} prises de bras enregistrées")
    return arm_ids


def collect_group(wb: Workbench, kind: str, intensities=(0.3, 0.7, 1.0), reps=1, contexts=None, log=print):
    """Passage tête ou mains INDÉPENDANT : une prise par (émotion, intensité, contexte,
    répétition), sans aucun geste des bras ni modèle."""
    cfg = wb.cfg
    contexts = contexts or DEFAULT_CONTEXTS
    wb.ctrl.hold(cfg.controlled_joints if kind == "hands" else cfg.all_joints)
    if kind == "hands":
        wb.ctrl.arm(confirm=True)  # les mains sont pilotées (motorisées)
    ids = []
    for e in cfg.emotions:
        for i in intensities:
            for ev in contexts:
                for _ in range(reps):
                    wb.event = ev
                    wb.prepare_take(kind, e, i)
                    wb.start_recording(countdown_s=0.3, background=False)
                    ids.append(wb.save_recording(status="accepted", notes="synthétique"))
                    if kind == "head":
                        wb.ctrl.hold(cfg.head_joints)  # tête raidie sur place (sans déplacement)
    wb.event = EventContext.none()
    wb.ctrl.disarm()
    log(f"  {len(ids)} prises de {'tête' if kind == 'head' else 'mains'} enregistrées")
    return ids


def collect_heads(wb: Workbench, intensities=(0.3, 0.7, 1.0), reps=1, contexts=None, log=print):
    return collect_group(wb, "head", intensities, reps, contexts, log)


def collect_hands(wb: Workbench, intensities=(0.3, 0.7, 1.0), reps=1, contexts=None, log=print):
    return collect_group(wb, "hands", intensities, reps, contexts or DEFAULT_CONTEXTS[:3], log)


def collect_synthetic(wb: Workbench, reps=2, intensities=(0.3, 0.7, 1.0), head_contexts=None, log=print,
                      hands: bool = False):
    """Collecte synthétique des trois passages, chacun indépendant."""
    arm_ids = collect_arms(wb, reps, intensities, log=log)
    head_ids = collect_heads(wb, intensities, contexts=head_contexts, log=log)
    if not hands:
        return arm_ids, head_ids
    return arm_ids, head_ids, collect_hands(wb, intensities, log=log)


def simulate_calibration(wb: Workbench, sweep_s: float = 25.0, activate_in=None):
    """Exerce la procédure de calibration sur le robot simulé (valeurs sans portée réelle)."""
    from .calibration import CalibrationSession, load_procedure, sim_sweeper
    cfg = wb.cfg
    sess = CalibrationSession(cfg, wb.adapter, load_procedure(cfg.root))
    for g in cfg.groups:
        wb.adapter.set_demonstrator(sim_sweeper(cfg, g))
        sess.sweep(g, max_duration_s=sweep_s)
    wb.adapter.set_demonstrator(None)
    for name in ("rest", "start"):
        for g in cfg.groups:
            # en simulation, on « place » le robot sur les poses de remplacement avant capture
            for j, v in cfg.poses[name][g].items():
                wb.adapter.q[j] = v
            sess.capture_pose(name, g)
    for j, v in zip(cfg.all_joints, cfg.pose("rest", cfg.all_joints)):
        wb.adapter.q[j] = v
        wb.adapter.target[j] = v
    return sess.result(version="sim-demo")


def run_demo(workdir="demo_sim", epochs=None, log=print) -> dict:
    work = Path(workdir)
    wb = Workbench(work / "data", work / "runs", adapter_kind="sim", virtual_time=True, sensor_noise_rad=0.002)
    wb.ctrl.connect()
    cfg = wb.cfg
    log("0. Calibration simulée (procédure exercée, valeurs sans portée réelle)")
    cal = simulate_calibration(wb)
    log(f"  {len(cal.limits)} articulations, {len(cal.coupled)} contraintes couplées mesurées "
        "(non activée : la démo garde les limites de remplacement)")

    log("1. Collecte synthétique des bras (simulation)")
    arm_ids = collect_arms(wb, reps=2, intensities=(0.3, 0.7, 1.0), log=log)

    log("2. Consultation et relecture virtuelle d'une prise")
    take = wb.store.load(arm_ids[0])
    rp = replay_trajectory(take, cfg.arm_joints)
    rep = wb.guardian.validate(rp, context_pose=dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints))))
    log(f"  {take.meta.take_id}: {take.meta.emotion} {take.meta.intensity}, {take.meta.n_samples} échantillons, "
        f"segment {tuple(round(x, 2) for x in take.meta.segment)} ; relecture : {rep.summary()}")

    log("3. Entraînement du modèle des bras")
    job = wb.train("arms", epochs=epochs, background=False)
    if job["state"] != "done":
        raise RuntimeError(job["error"])
    m_arms = json.loads((Path(job["run"]) / "metrics.json").read_text())["metrics"]
    log(f"  val RMSE {m_arms.get('val_rmse_rad')} rad (référence moyenne par étiquette "
        f"{m_arms.get('val_baseline_label_mean_rmse_rad')}), guardian {m_arms['guardian']['rate']:.0%}")

    log("4. Génération, validation et exécution simulée (bras seuls, tête maintenue)")
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    wb.ctrl.return_to_rest(confirm=True)
    out = wb.generate("joie", 0.7, use_event=False)
    ok = out.get("cycle", {}).get("report", {}).get("ok")
    log(f"  geste {out['generated']['duration_s']:.2f} s ; cycle {'accepté' if ok else 'refusé'}")
    if ok:
        wb.execute_plan(out["plan_id"], background=False)
        log(f"  exécution simulée : état {wb.ctrl.state.value}")
    wb.ctrl.disarm()

    log("5. Collecte tête indépendante : la tête est guidée (simulée), bras raides sur place")
    head_ids = collect_heads(wb, intensities=(0.3, 0.7, 1.0), reps=2, log=log)

    log("6. Entraînement de la tête seule puis geste combiné bras + tête avec événement")
    job_h = wb.train("head", epochs=epochs, background=False)
    if job_h["state"] != "done":
        raise RuntimeError(job_h["error"])
    m_head = json.loads((Path(job_h["run"]) / "metrics.json").read_text())["metrics"]
    log(f"  val RMSE tête {m_head.get('val_rmse_rad')} rad (référence {m_head.get('val_baseline_label_mean_rmse_rad')}),"
        f" guardian {m_head['guardian']['rate']:.0%}")
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    wb.ctrl.return_to_rest(confirm=True)
    wb.set_event({"present": True, "azimuth": 35, "elevation": 5, "unit": "deg", "frame": "torso", "source": "test"})
    out2 = wb.generate("surprise", 0.8)
    ok = out2.get("cycle", {}).get("report", {}).get("ok")
    log(f"  cycle complet {'accepté' if ok else 'refusé'} {out2.get('cycle_error', '')}")
    if ok:
        wb.execute_plan(out2["plan_id"], background=False)
        log(f"  exécution simulée : état {wb.ctrl.state.value}")
    wb.ctrl.disarm()

    log("7. Collecte mains indépendante (ouverture pilotée, simulée), entraînement, geste combiné")
    hand_ids = collect_hands(wb, log=log)
    job_m = wb.train("hands", epochs=epochs, background=False)
    if job_m["state"] != "done":
        raise RuntimeError(job_m["error"])
    m_hands = json.loads((Path(job_m["run"]) / "metrics.json").read_text())["metrics"]
    log(f"  val RMSE mains {m_hands.get('val_rmse_rad')} (ouverture 0..1), guardian {m_hands['guardian']['rate']:.0%}")
    wb.ctrl.hold(cfg.controlled_joints)
    wb.ctrl.arm(confirm=True)
    wb.ctrl.return_to_rest(confirm=True)
    out3 = wb.generate("joie", 0.9, use_event=False)
    ok = out3.get("cycle", {}).get("report", {}).get("ok")
    log(f"  cycle bras + tête + mains {'accepté' if ok else 'refusé'} {out3.get('cycle_error', '')}")
    if ok:
        wb.execute_plan(out3["plan_id"], background=False)
    wb.ctrl.disarm()
    return {"arm_takes": len(arm_ids), "head_takes": len(head_ids), "hand_takes": len(hand_ids),
            "arms_metrics": m_arms, "head_metrics": m_head, "hands_metrics": m_hands,
            "final_state": wb.ctrl.state.value, "workdir": str(work),
            "calibration": cal.to_dict()}
