import numpy as np
import pytest

from nao_motion.adapters.base import ConnectionLost, RoleError
from nao_motion.control.controller import ControllerError, State
from nao_motion.control.recording import HeadRecording
from nao_motion.events import EventContext
from nao_motion.trajectory import Trajectory

from .test_data import arm_take


def gesture(cfg, amp=0.15, dur=2.0):
    j = cfg.all_joints
    t = np.linspace(0, dur, int(dur * 50) + 1)
    return Trajectory(j, t, np.array(cfg.pose("start", j)) + amp * np.sin(np.pi * t / dur)[:, None] ** 2)


def test_no_motion_on_connect_or_construction(wb, cfg):
    q0 = dict(wb.adapter.q)
    wb.adapter.sleep(1.0)
    assert wb.adapter.command_log == []
    assert wb.adapter.q == q0
    assert wb.ctrl.armed is False and wb.ctrl.state == State.IDLE


def test_execution_requires_arming(wb, cfg):
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert plan.report.ok
    with pytest.raises(ControllerError):
        wb.ctrl.execute(plan)
    assert wb.adapter.command_log == []
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=False)


def test_full_cycle_returns_to_rest(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    names = [s[0] for s in plan.segments]
    assert names == ["repos_vers_depart", "depart_vers_geste", "geste", "geste_vers_depart", "depart_vers_repos"]
    assert wb.ctrl.execute(plan) == "done"
    q = wb.adapter.read_angles(cfg.all_joints)
    assert np.abs(q - np.array(cfg.pose("rest", cfg.all_joints))).max() < 0.05


def test_plan_refused_if_not_at_rest(wb, cfg):
    for j in cfg.arm_joints:
        wb.adapter.q[j] += 0.3
    with pytest.raises(ControllerError):
        wb.ctrl.plan_cycle(gesture(cfg))
    assert wb.ctrl.plan_cycle(gesture(cfg), from_pose="current").report.ok


def test_stop_holds_without_return_to_rest(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    n_ticks = []

    def on_tick(t, q):
        n_ticks.append(t)
        if t > 1.5:
            wb.ctrl.stop()

    assert wb.ctrl.execute(plan, on_tick=on_tick) == "stopped"
    assert wb.ctrl.state == State.HOLDING and wb.ctrl.armed is False
    n_cmd = len(wb.adapter.command_log)
    held = wb.adapter.read_angles(cfg.all_joints)
    wb.adapter.sleep(2.0)
    assert len(wb.adapter.command_log) == n_cmd  # aucune commande après l'arrêt
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.all_joints), held, atol=1e-9)
    assert np.abs(held - np.array(cfg.pose("rest", cfg.all_joints))).max() > 0.05  # pas revenu au repos


def test_connection_loss_faults_without_further_commands(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    wb.adapter.fail_after_s = wb.adapter.now() - wb.adapter._connect_time + 1.0
    with pytest.raises(ConnectionLost):
        wb.ctrl.execute(plan)
    assert wb.ctrl.state == State.FAULT and not wb.ctrl.armed
    n = len(wb.adapter.command_log)
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=True)
    assert len(wb.adapter.command_log) == n


def test_real_robot_refused_when_uncalibrated(wb, cfg):
    wb.adapter.is_real = True
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=True)


def test_guided_joint_cannot_be_commanded(wb, cfg):
    wb.adapter.set_roles(guided=cfg.head_joints, driven=cfg.arm_joints)
    with pytest.raises(RoleError):
        wb.adapter.send_angles(cfg.head_joints, [0.0, 0.0], 0.1)
    with pytest.raises(RoleError):
        wb.adapter.set_roles(guided=["HeadYaw"], driven=["HeadYaw"])
    with pytest.raises(RoleError):
        wb.adapter.read_angles(["LKneePitch"])


def test_head_take_synchronised_with_arm_replay(wb, cfg):
    """La prise de tête conserve les bras mesurés pendant la relecture ; le segment
    correspond exactement à la fenêtre de relecture sur la même horloge."""
    meta, tr = arm_take(cfg, duration=2.5, n=126)
    meta.status = "accepted"
    wb.store.save(meta, tr)
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    wb.event = EventContext(True, 0.5, 0.0, source="test")
    wb.prepare_head_take(meta.take_id)
    rec = wb.recording
    assert isinstance(rec, HeadRecording)
    assert all(wb.adapter.roles[j] == "guided" for j in cfg.head_joints)
    assert all(wb.adapter.roles[j] == "driven" for j in cfg.arm_joints)
    wb.start_recording(countdown_s=0.5, background=False)
    res = rec.result
    assert res.complete
    s0, s1 = res.segment
    assert abs((s1 - s0) - rec.replay.duration) < 1e-6
    # bras mesurés dans le segment ≈ relecture (latence du simulateur comprise)
    tt = np.linspace(s0, s1, 50)
    meas = res.traj.select(cfg.arm_joints).sample(tt)
    ref = rec.replay.sample(tt - s0)
    assert np.abs(meas - ref).max() < 0.05
    head_id = wb.save_recording(status="accepted")
    hm = wb.store.load_meta(head_id)
    assert hm.arm_take_id == meta.take_id and hm.kind == "head"
    assert hm.event["present"] and hm.segment == (s0, min(s1, res.traj.t[-1]))
    assert set(hm.guided_joints) == set(cfg.head_joints)
    # 2e contexte de regard sur le même mouvement des bras
    wb.ctrl.hold(cfg.head_joints)
    wb.event = EventContext.none()
    wb.prepare_head_take(meta.take_id)
    wb.start_recording(countdown_s=0.2, background=False)
    wb.save_recording(status="accepted")
    assert len(wb.store.head_takes_for(meta.take_id)) == 2


def test_refused_plan_is_never_executed(wb, cfg):
    from nao_motion.guardian import GuardianRejection
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    plan.report.ok = False  # simule un refus du guardian
    wb.plans["p"] = {"gen": None, "plan": plan}
    wb.ctrl.arm(confirm=True)
    with pytest.raises(GuardianRejection):
        wb.execute_plan("p", background=False)
    with pytest.raises(GuardianRejection):
        wb.ctrl.execute(plan)
    assert wb.adapter.command_log == []


def test_rest_outside_motion_zone_is_allowed(tmp_path):
    """Repos hors zone de mouvement (bras posés sur les genoux : contact). Seuls les
    segments repos <-> départ le quittent, vérifiés sur les limites d'usine et la
    vitesse réduite ; le geste reste soumis au guardian complet."""
    import copy
    from nao_motion.config import load_config
    from .conftest import make_wb
    cfg = copy.deepcopy(load_config())
    j = "LShoulderPitch"
    lo, hi = cfg.limits[j]
    cfg.rest_limits[j] = (lo, hi + 0.4)           # « usine » plus large que la zone calibrée
    cfg.poses["rest"]["left_arm"][j] = hi + 0.2   # repos hors zone, dans l'usine
    wb = make_wb(tmp_path, cfg)
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.2
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert plan.report.ok, plan.report.to_dict()
    assert [s[0] for s in plan.segments][0] == "repos_vers_depart"
    # segment repos <-> départ trop rapide : refusé
    cfg.guardian.rest_velocity_rad_s = 0.01
    rep = wb.ctrl.guardian.validate_rest_transition(wb.ctrl._transition(
        np.array(cfg.pose("rest", cfg.all_joints)), np.array(cfg.pose("start", cfg.all_joints)), cfg.all_joints))
    assert not rep.ok and any(v.code == "rest_velocity" for v in rep.violations)
    cfg.guardian.rest_velocity_rad_s = 0.3
    # repos au-delà des limites d'usine : refusé
    cfg.poses["rest"]["left_arm"][j] = hi + 0.5
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.5
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert not plan.report.ok and any(v.code == "rest_limit" for v in plan.report.violations)
    # le geste lui-même reste vérifié par le guardian complet
    cfg.poses["rest"]["left_arm"][j] = hi + 0.2
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.2
    plan = wb.ctrl.plan_cycle(gesture(cfg, amp=1.5))
    assert not plan.report.ok and any(v.code == "limit" for v in plan.report.violations)


def test_replay_guardian_can_be_disabled_for_head_pass(wb, cfg, monkeypatch):
    """Guardian au rejeu désactivé (opérateur à côté du robot) : la relecture refusée par le
    guardian est quand même préparée et enregistrée ; réactivé, elle est de nouveau refusée.
    La génération / exécution de plans reste protégée."""
    from nao_motion.guardian import GuardianRejection
    from nao_motion.ui.server import Api
    meta, tr = arm_take(cfg, duration=2.5, n=126)
    meta.status = "accepted"
    wb.store.save(meta, tr)
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    monkeypatch.setattr(cfg.guardian, "max_velocity_rad_s", 1e-3)  # tout geste devient trop rapide
    monkeypatch.setattr(cfg.guardian, "per_joint_velocity_rad_s", {})
    api = Api(wb)
    assert api.get_status({})["replay_guardian"] is True
    with pytest.raises(GuardianRejection):
        wb.prepare_head_take(meta.take_id)
    assert wb.ctrl.state == State.IDLE
    st = api.post_guardian_replay({"enabled": False})
    assert st["replay_guardian"] is False
    info = wb.prepare_head_take(meta.take_id)
    assert info["guardian_replay"] is False
    assert any("DÉSACTIVÉ" in e for e in wb.ctrl.events)
    wb.start_recording(countdown_s=0.2, background=False)
    assert wb.recording.result.complete
    # pendant un enregistrement on ne bascule pas ; ensuite on réactive
    api.post_guardian_replay({"enabled": True})
    wb.ctrl.hold(cfg.head_joints)
    with pytest.raises(GuardianRejection):
        wb.prepare_head_take(meta.take_id)
    # les plans (génération / exécution) ne sont jamais contournés, même guardian au rejeu désactivé
    api.post_guardian_replay({"enabled": False})
    j = cfg.arm_joints
    q = np.array(cfg.pose("start", j))
    fast = Trajectory(j, [0.0, 0.1], [q, q + 0.3])
    from nao_motion.control.controller import Plan
    bad = Plan(fast, [("x", 0.0, 0.1)], wb.guardian.validate(fast, check_duration=False))
    assert not bad.report.ok
    with pytest.raises(GuardianRejection):
        wb.ctrl.execute(bad)
