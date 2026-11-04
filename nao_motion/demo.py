"""Parcours complet en simulation (temps virtuel), données SYNTHÉTIQUES.

1. collecter et sauvegarder des démonstrations des bras ;
2. les consulter et les rejouer virtuellement (validation guardian) ;
3. entraîner le modèle des bras ;
4. générer et valider une trajectoire, l'exécuter sur le robot simulé ;
5. enregistrer des prises de tête synchronisées sur des prises de bras ;
6. entraîner la tête (bras figés) et exécuter le cycle complet simulé.

Les données synthétiques vérifient le fonctionnement technique, pas l'expressivité.
"""
from __future__ import annotations

import json
from pathlib import Path

from .control.recording import replay_trajectory
from .events import EventContext
from .workbench import Workbench


def collect_synthetic(wb: Workbench, reps=2, intensities=(0.3, 0.7, 1.0), head_per_arm=None,
                      head_contexts=None, log=print):
    """Collecte synthétique par le vrai chemin d'enregistrement (adaptateur simulé)."""
    cfg = wb.cfg
    arm_ids = []
    for e in cfg.emotions:
        for i in intensities:
            for _ in range(reps):
                wb.prepare_arm_take(e, i)
                wb.start_recording(countdown_s=0.3, background=False)
                arm_ids.append(wb.save_recording(status="accepted", notes="synthétique"))
    log(f"  {len(arm_ids)} prises de bras enregistrées")
    head_ids = []
    contexts = head_contexts or [EventContext.none(), EventContext(True, 0.7, 0.1, source="test"),
                                 EventContext(True, -0.6, 0.0, source="test")]
    targets = arm_ids if head_per_arm is None else arm_ids[:head_per_arm]
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    for aid in targets:
        for ev in contexts:
            wb.event = ev
            wb.prepare_head_take(aid)
            wb.start_recording(countdown_s=0.3, background=False)
            head_ids.append(wb.save_recording(status="accepted", notes="synthétique"))
            wb.ctrl.hold(cfg.head_joints)  # tête raidie sur place (sans déplacement)
    wb.event = EventContext.none()
    wb.ctrl.disarm()
    log(f"  {len(head_ids)} prises de tête enregistrées")
    return arm_ids, head_ids


def run_demo(workdir="demo_sim", epochs=300, log=print) -> dict:
    work = Path(workdir)
    wb = Workbench(work / "data", work / "runs", adapter_kind="sim", virtual_time=True, sensor_noise_rad=0.002)
    wb.ctrl.connect()
    cfg = wb.cfg
    log("1. Collecte synthétique des bras et de la tête (simulation)")
    arm_ids, head_ids = collect_synthetic(wb, reps=2, intensities=(0.3, 0.7, 1.0), head_per_arm=None)

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
    log(f"  geste {out['generated']['duration_s']:.2f} s ; {('OK' if out['motion_report']['ok'] else 'REFUS')} ; "
        f"cycle {out.get('cycle', {}).get('report', {}).get('ok')}")
    if out.get("cycle", {}).get("report", {}).get("ok"):
        wb.execute_plan(out["plan_id"], background=False)
        log(f"  exécution simulée : état {wb.ctrl.state.value}")

    log("5-6. Entraînement de la tête (bras figés) puis cycle complet avec événement")
    job_h = wb.train("head", epochs=epochs, background=False)
    if job_h["state"] != "done":
        raise RuntimeError(job_h["error"])
    m_head = json.loads((Path(job_h["run"]) / "metrics.json").read_text())["metrics"]
    log(f"  val RMSE tête {m_head.get('val_rmse_rad')} rad (référence {m_head.get('val_baseline_label_mean_rmse_rad')}),"
        f" guardian {m_head['guardian']['rate']:.0%}")
    wb.set_event({"present": True, "azimuth": 35, "elevation": 5, "unit": "deg", "frame": "torso", "source": "test"})
    out2 = wb.generate("surprise", 0.8)
    ok = out2.get("cycle", {}).get("report", {}).get("ok")
    log(f"  cycle complet {'accepté' if ok else 'refusé'} : {out2.get('cycle_error', '')}")
    if ok:
        wb.execute_plan(out2["plan_id"], background=False)
        log(f"  exécution simulée : état {wb.ctrl.state.value}")
    wb.ctrl.disarm()
    return {"arm_takes": len(arm_ids), "head_takes": len(head_ids), "arms_metrics": m_arms,
            "head_metrics": m_head, "final_state": wb.ctrl.state.value, "workdir": str(work)}
