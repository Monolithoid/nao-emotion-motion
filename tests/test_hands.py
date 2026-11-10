"""Ouverture des mains (LHand, RHand) : configuration, guardian, commande manuelle,
passage mains (indépendant : tout raide, mains pilotées), jeu de données, modèle et génération."""
import numpy as np
import pytest

from nao_motion.control.controller import ControllerError
from nao_motion.control.recording import HandRecording
from nao_motion.events import EventContext
from nao_motion.guardian import Guardian
from nao_motion.models.datasets import head_examples
from nao_motion.trajectory import Trajectory

from .conftest import make_wb
from .test_bridge import bridge  # noqa: F401  (fixture)


def test_hands_config_outside_calibrated_groups(cfg):
    assert cfg.hand_joints == ["LHand", "RHand"]
    assert not set(cfg.hand_joints) & set(cfg.all_joints)  # la calibration ne les demande pas
    assert cfg.controlled_joints == cfg.all_joints + cfg.hand_joints
    for j in cfg.hand_joints:
        assert cfg.limits[j] == (0.0, 1.0) and cfg.rest_limits[j] == (0.0, 1.0)
        assert cfg.guardian.margin(j) == 0.0
    assert cfg.pose("start", cfg.hand_joints) == [0.3, 0.3]


def test_guardian_hands_range_and_speed(cfg):
    g = Guardian(cfg)
    t = np.linspace(0, 1.0, 51)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    full = Trajectory(cfg.hand_joints, t, np.stack([s, 1 - s], axis=1))  # 0 -> 1 et 1 -> 0 en 1 s
    assert g.validate(full).ok
    over = Trajectory(cfg.hand_joints, t, np.stack([1.05 * s, s], axis=1))
    assert {v.code for v in g.validate(over).violations} == {"limit"}
    fast = Trajectory(cfg.hand_joints, t * 0.2, np.stack([s, s], axis=1))  # 0 -> 1 en 0,2 s
    assert {"velocity", "acceleration"} <= {v.code for v in g.validate(fast, check_duration=False).violations}


def test_manual_hands_need_arming_then_move(wb, cfg):
    with pytest.raises(ControllerError):
        wb.set_hands({"LHand": 0.9}, background=False)
    assert wb.adapter.command_log == []
    with pytest.raises(ControllerError):
        wb.set_hands({"LShoulderPitch": 0.0}, background=False)
    wb.ctrl.arm(confirm=True)
    wb.set_hands({"LHand": 0.9, "RHand": 0.1}, background=False)
    q = wb.adapter.read_angles(cfg.hand_joints)
    np.testing.assert_allclose(q, [0.9, 0.1], atol=0.03)
    assert wb.hands_state()["measured"]["LHand"] == pytest.approx(0.9, abs=0.03)
    # bras et tête jamais commandés par ce mouvement
    assert all(set(c[1]) <= set(cfg.hand_joints) for c in wb.adapter.command_log)


def test_cycle_from_rest_accepts_open_hands(wb, cfg):
    from .test_controller_sync import gesture
    for j in cfg.hand_joints:
        wb.adapter.q[j] = 0.9  # mains ouvertes depuis l'interface : le robot reste « au repos »
    g = gesture(cfg)
    hands = Trajectory(cfg.hand_joints, g.t, np.tile(cfg.pose("start", cfg.hand_joints), (len(g.t), 1)))
    plan = wb.ctrl.plan_cycle(Trajectory.merge(g, hands))
    assert plan.report.ok, plan.report.violations
    wb.ctrl.arm(confirm=True)
    assert wb.ctrl.execute(plan) == "done"
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.hand_joints), cfg.pose("rest", cfg.hand_joints), atol=0.03)


def test_hands_pass_dataset_training_generation(trained, cfg, tmp_path):
    src = trained["wb"]
    wb = make_wb(tmp_path, cfg)
    # mêmes modèles bras et tête que le jeu entraîné (pour la génération combinée)
    import shutil
    shutil.copytree(src.runs_dir, wb.runs_dir, dirs_exist_ok=True)
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    ids = []
    contexts = [EventContext.none(), EventContext(True, 0.6, 0.1, source="test", distance_m=0.5)]
    jobs = [{"emotion": e, "intensity": i} for e in ("joie", "colere", "surprise", "tristesse") for i in (0.4, 1.0)]
    for job in jobs:
        for ev in contexts:
            wb.event = ev
            q0 = wb.adapter.read_angles(cfg.all_joints)
            info = wb.prepare_hands_take(job["emotion"], job["intensity"])
            assert info["kind"] == "hands"
            np.testing.assert_allclose(wb.adapter.read_angles(cfg.all_joints), q0, atol=1e-6)  # raidis sur place
            n = len(wb.adapter.command_log)
            assert isinstance(wb.recording, HandRecording)
            assert all(r != "guided" for r in wb.adapter.roles.values())  # rien n'est relâché
            wb.start_recording(countdown_s=0.2, background=False)
            r = wb.recording.result
            assert r.complete and r.reference is not None and r.reference.t.shape == r.traj.t.shape
            # seules les mains sont commandées pendant le passage
            assert all(set(c[1]) <= set(cfg.hand_joints) for c in wb.adapter.command_log[n:])
            ids.append(wb.save_recording(status="accepted"))
    take = wb.store.load(ids[0])
    m = take.meta
    assert m.kind == "hands" and m.guided_joints == [] and set(cfg.hand_joints) <= set(m.joints)
    assert m.hands_initial is not None and m.head_initial is None
    ref = wb.store.load_reference(m.take_id)
    assert ref.joints == cfg.hand_joints and m.joints == cfg.hand_joints
    # la consigne filtrée reste dans la plage et sous les limites du guardian
    assert ref.select(cfg.hand_joints).q.min() >= 0.0 and ref.select(cfg.hand_joints).q.max() <= 1.0
    seg = take.segment_trajectory(cfg.hand_joints)
    assert seg.q.max() - seg.q.min() > 0.2  # les mains ont bougé pendant le geste

    ex = head_examples(wb.store, cfg, kind="hands", target="command")
    assert len(ex) == len(ids) and ex.Y.shape[1] == 8 * len(cfg.hand_joints)
    # les prises de tête ne sont pas mélangées aux prises de mains
    assert not set(ex.take_ids) & {x.take_id for x in wb.store.list(kind="head")}

    job = wb.train("hands", epochs=150, background=False)
    assert job["state"] == "done", job["error"]
    wb.engine = None
    eng = wb.get_engine()
    assert eng.hands_model is not None
    gen = eng.generate("joie", 0.9, EventContext.none())
    assert gen.hands is not None and gen.combined.joints == cfg.all_joints + cfg.hand_joints
    assert set(gen.durations) == {"arms", "head", "hands"}  # trois modèles indépendants combinés
    assert Guardian(cfg).validate(gen.combined).ok
    wb.ctrl.disarm()
    wb.ctrl.hold(cfg.controlled_joints)
    wb.ctrl.arm(confirm=True)
    wb.ctrl.return_to_rest(confirm=True)
    out = wb.generate("joie", 0.9, use_event=False)
    assert out["cycle"]["report"]["ok"], out["cycle"]
    wb.execute_plan(out["plan_id"], background=False)
    assert any(set(cfg.hand_joints) <= set(c[1]) for c in wb.adapter.command_log)


def test_hands_live_target_during_recording(trained, cfg, tmp_path):
    wb = make_wb(tmp_path, cfg)
    wb.sim_demonstration = False  # pas d'opérateur simulé : c'est la cible de l'interface qui compte
    wb.ctrl.hold(cfg.all_joints)
    with pytest.raises(ControllerError):
        wb.prepare_hands_take("joie", 0.5)  # contrôleur désarmé : pilotage impossible
    wb.ctrl.arm(confirm=True)
    wb.prepare_hands_take("joie", 0.5)
    rec = wb.recording
    rec.set_target({"LHand": 1.0})
    rec.set_target({"RHand": 5.0})  # bornée à la plage
    assert rec.hand_target.tolist() == [1.0, 1.0]
    rec.hand_target = None
    seen = []
    orig = rec._follow

    def follow(t):
        if not seen:
            wb.set_hands({"LHand": 0.95, "RHand": 0.05})  # même route que l'interface
        seen.append(1)
        orig(t)
        if t > 2.5:
            rec.request_stop()  # comme le bouton Arrêter
    rec._follow = follow
    wb.start_recording(countdown_s=0.1, background=False)
    ref = rec.result.reference.select(cfg.hand_joints)
    np.testing.assert_allclose(ref.q[-1], [0.95, 0.05], atol=0.02)
    # consigne filtrée : vitesse sous la limite du guardian
    v = np.abs(np.diff(ref.q, axis=0) / np.diff(ref.t)[:, None]).max()
    assert v <= cfg.guardian.velocity_limit("LHand") + 1e-6


def test_bridge_forwards_hands(bridge, cfg):  # noqa: F811
    from nao_motion.adapters.bridge_client import BridgeAdapter
    br, backend, port = bridge
    a = BridgeAdapter(cfg, host="127.0.0.1", port=port)
    a.connect()
    backend.q.update({"LHand": 0.2, "RHand": 0.3})
    np.testing.assert_allclose(a.read_angles(cfg.hand_joints), [0.2, 0.3])
    a.set_roles(driven=cfg.hand_joints)
    a.send_angles(["LHand"], [0.8], 0.3)
    assert backend.q["LHand"] == 0.8 and backend.stiff["RHand"] > 0
    a.disconnect()
