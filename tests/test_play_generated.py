"""Generation tab: play the generated gesture on the robot (simulated here).

The cycle is planned again from the measured posture when the operator presses Play,
checked whole by the guardian, and nothing is sent on a refusal or when the motors are
not armed."""
import numpy as np
import pytest

from nao_motion.control.controller import ControllerError
from nao_motion.guardian import GuardianRejection
from nao_motion.ui.server import Api

from .conftest import make_wb
from .test_controller_sync import gesture


@pytest.fixture
def gwb(trained, cfg, tmp_path):
    wb = make_wb(tmp_path, cfg)
    wb.engine = trained["wb"].get_engine()  # same trained models, fresh simulated robot
    return wb


def _generate(wb, cfg, emotion="joie"):
    """Generation through the real path; the gesture to play is then replaced by a fixed one
    known to pass the guardian (the shared test models change from one test to another)."""
    out = wb.generate(emotion, 0.9, use_event=False)
    wb.plans[out["plan_id"]]["played"] = gesture(cfg)
    return out


def _put_in_start_pose(wb, cfg):
    """Robot left in the start pose (as after a recording): not measured at rest."""
    for j, v in zip(cfg.all_joints, cfg.pose("start", cfg.all_joints)):
        wb.adapter.q[j] = wb.adapter.target[j] = v


def test_play_from_current_posture_after_a_recording(gwb, cfg):
    _put_in_start_pose(gwb, cfg)
    out = _generate(gwb, cfg)
    # the old flow planned from rest only: "not measured at rest" and nothing playable
    assert "cycle_error" not in out, out.get("cycle_error")
    assert out["cycle"]["start"] == "current"
    gwb.ctrl.arm(confirm=True)
    r = gwb.execute_plan(out["plan_id"], background=False, start="auto", end="rest")
    assert r["start"] == "current" and r["started"]
    assert gwb.play_info["state"] == "done" and gwb.play_info["result"] == "done"
    assert gwb.adapter.command_log, "nothing was sent"
    # ended at rest
    rest = np.array(cfg.pose("rest", cfg.all_joints))
    np.testing.assert_allclose(gwb.adapter.read_angles(cfg.all_joints), rest, atol=0.05)
    assert gwb.status()["play"]["state"] == "done"


def test_play_from_rest_and_stay_in_start_pose(gwb, cfg):
    out = _generate(gwb, cfg, "colere")
    assert out["cycle"]["start"] == "rest"
    gwb.ctrl.arm(confirm=True)
    r = gwb.execute_plan(out["plan_id"], background=False, start="auto", end="start")
    assert r["start"] == "rest" and [s[0] for s in r["cycle"]["segments"]][-1] == "geste_vers_depart"
    start = np.array(cfg.pose("start", cfg.all_joints))
    np.testing.assert_allclose(gwb.adapter.read_angles(cfg.all_joints), start, atol=0.05)


def test_nothing_sent_when_not_armed_or_not_at_rest(gwb, cfg):
    _put_in_start_pose(gwb, cfg)
    out = _generate(gwb, cfg)
    with pytest.raises(ControllerError, match="désarmé"):
        gwb.execute_plan(out["plan_id"], background=False, start="auto")
    gwb.ctrl.arm(confirm=True)
    with pytest.raises(ControllerError, match="repos"):
        gwb.execute_plan(out["plan_id"], background=False, start="rest")  # explicit "rest only"
    with pytest.raises(ControllerError, match="inconnu"):
        gwb.execute_plan("plan-nope", background=False, start="auto")
    assert gwb.adapter.command_log == []


def test_guardian_refusal_sends_nothing(gwb, cfg):
    out = _generate(gwb, cfg)
    gwb.plans[out["plan_id"]]["played"] = gesture(cfg, amp=1.5)  # beyond the limits
    gwb.ctrl.arm(confirm=True)
    with pytest.raises(GuardianRejection):
        gwb.execute_plan(out["plan_id"], background=False, start="auto")
    assert gwb.adapter.command_log == []
    assert gwb.play_info["state"] == "refused"


def test_play_route_without_confirmation(gwb, cfg):
    api = Api(gwb)
    _put_in_start_pose(gwb, cfg)
    out = api.post_generate({"emotion": "joie", "intensity": 0.8, "use_event": False})
    gwb.plans[out["plan_id"]]["played"] = gesture(cfg)
    with pytest.raises(ControllerError):  # motors not armed: nothing moves
        api.post_play({"plan_id": out["plan_id"]})
    assert gwb.adapter.command_log == []
    gwb.ctrl.arm(confirm=True)
    r = api.post_play({"plan_id": out["plan_id"], "start": "auto", "end": "rest"})
    assert r["started"] and r["start"] == "current"


def test_transition_into_a_fast_gesture_does_not_overshoot():
    """A generated gesture may start faster than the transition's cap (here 1.48 rad/s entering
    while decelerating hard): the transition used to run to its longest duration and overshoot
    by several radians, so the whole cycle was refused and nothing could be played."""
    from nao_motion.control.transitions import quintic_matched
    q0, q1 = np.array([-1.2]), np.array([-1.16])
    v1, a1 = np.array([1.48]), np.array([-3.78])
    z = np.zeros(1)
    tr = quintic_matched(q0, z, z, q1, v1, a1, ["LElbowYaw"], 1.2, 6.4)
    assert tr.q.min() > -1.6 and tr.q.max() < -0.9, (tr.q.min(), tr.q.max())
    v = np.diff(tr.q[:, 0]) / np.diff(tr.t)
    assert np.abs(v).max() <= 1.5


def test_guardian_off_plays_a_refused_gesture_within_factory_limits(gwb, cfg):
    """Operator's test at their own risk: the guardian verdict is not applied, the RAW gesture is
    played, and the factory limits stay an absolute ceiling (positions clipped into them)."""
    out = _generate(gwb, cfg)
    big = gesture(cfg, amp=1.5)  # refused by the guardian, partly beyond the factory limits
    gwb.plans[out["plan_id"]]["gen"] = type("G", (), {"combined": big})()
    gwb.plans[out["plan_id"]]["played"] = big
    gwb.ctrl.arm(confirm=True)
    with pytest.raises(GuardianRejection):
        gwb.execute_plan(out["plan_id"], background=False, start="auto")
    assert gwb.adapter.command_log == []
    r = gwb.execute_plan(out["plan_id"], background=False, start="auto", guardian=False)
    assert r["guardian"] is False and not r["cycle"]["report"]["ok"]  # what the guardian would say
    assert gwb.play_info["state"] == "done", gwb.play_info
    assert any("factory" in n for n in r["notes"])
    sent = np.array([c[2] for c in gwb.adapter.command_log if list(c[1]) == list(big.joints)])
    lo = np.array([cfg.rest_limits[j][0] for j in big.joints])
    hi = np.array([cfg.rest_limits[j][1] for j in big.joints])
    assert sent.size and (sent >= lo - 1e-9).all() and (sent <= hi + 1e-9).all()


def test_guardian_off_still_refuses_beyond_factory_limits(gwb, cfg):
    from nao_motion.control.controller import Plan
    from nao_motion.guardian import Report
    g = gesture(cfg)
    g.q = g.q.copy()
    g.q[:, g.joints.index("LShoulderPitch")] = 5.0
    plan = Plan(g, [("geste", 0.0, g.duration)], Report(True, []))
    gwb.ctrl.arm(confirm=True)
    with pytest.raises(ControllerError, match="usine"):
        gwb.ctrl.execute(plan, guardian=False)
    assert gwb.adapter.command_log == []


def test_play_route_guardian_flag(gwb, cfg):
    api = Api(gwb)
    out = api.post_generate({"emotion": "joie", "intensity": 0.8, "use_event": False})
    gwb.plans[out["plan_id"]]["played"] = gesture(cfg)
    gwb.plans[out["plan_id"]]["gen"] = type("G", (), {"combined": gesture(cfg)})()
    gwb.ctrl.arm(confirm=True)
    r = api.post_play({"plan_id": out["plan_id"], "guardian": False})
    assert r["guardian"] is False


def test_project_point_is_the_nearest_point_inside_the_zone():
    from nao_motion.guardian.project import project_point
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    half = [(np.array([1.0, 1.0]), 1.0)]  # x + y <= 1
    np.testing.assert_allclose(project_point(np.array([0.2, 0.3]), lo, hi, half), [0.2, 0.3])  # inside: untouched
    np.testing.assert_allclose(project_point(np.array([2.0, 0.0]), lo, hi, half), [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(project_point(np.array([1.0, 1.0]), lo, hi, half), [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(project_point(np.array([3.0, 0.5]), lo, hi, half), [1.0, 0.0], atol=1e-6)


def test_projected_control_points_keep_the_curve_in_the_zone(gwb, cfg):
    """Generation with projection: every control point, hence (convex hull) every decoded
    sample, is inside the calibrated limits minus margin and the coupled zone of its group."""
    from nao_motion.guardian.project import zone
    eng = gwb.get_engine()
    hit = 0
    for e in cfg.emotions:
        for i in (0.3, 1.0):
            raw = eng.generate(e, i, None)
            gen = eng.generate(e, i, None, project_zone=True)
            hit += any("moved into the zone" in n for n in gen.notes)
            for tr in (gen.arms, gen.head):
                lo, hi, half = zone(cfg, tr.joints)
                assert (tr.q >= lo - 1e-6).all() and (tr.q <= hi + 1e-6).all()
                for a, b in half:
                    assert (tr.q @ a <= b + 1e-6).all()
            # control points already inside the zone are not moved
            lo, hi, half = zone(cfg, cfg.arm_joints)
            inside = [k for k, p in enumerate(raw.arm_ctrl)
                      if (p >= lo).all() and (p <= hi).all() and all(a @ p <= b for a, b in half)]
            np.testing.assert_array_equal(gen.arm_ctrl[inside], raw.arm_ctrl[inside])
    assert gwb.generate("joie", 0.9, use_event=False, project_zone=False)["generated"]["inputs"]["project_zone"] is False


def _cfg_wide_factory(j="LShoulderRoll", widen=0.4):
    import copy
    from nao_motion.config import load_config
    cfg = copy.deepcopy(load_config())
    lo, hi = cfg.limits[j]
    cfg.rest_limits[j] = (lo - widen, hi + widen)  # factory range wider than the calibrated zone
    return cfg, lo, hi


def test_play_from_a_posture_outside_the_zone(tmp_path):
    """Arms placed by hand outside the calibrated zone (e.g. LShoulderRoll close to the body):
    the approach segment to the start pose is checked on the factory limits, like rest -> start.
    It used to be checked on the zone, so the whole cycle was refused at t = 0 while the gesture
    itself was accepted."""
    cfg, lo, hi = _cfg_wide_factory()
    wb = make_wb(tmp_path, cfg)
    for j, v in zip(cfg.all_joints, cfg.pose("start", cfg.all_joints)):
        wb.adapter.q[j] = wb.adapter.target[j] = v
    wb.adapter.q["LShoulderRoll"] = wb.adapter.target["LShoulderRoll"] = lo - 0.15
    wb.plans["p"] = {"gen": None, "played": gesture(cfg), "plan": None}
    plan, start = wb.plan_generated(gesture(cfg), "auto", "rest")
    assert start == "current" and plan.report.ok, plan.report.to_dict()
    wb.ctrl.arm(confirm=True)
    wb.execute_plan("p", background=False, start="auto")
    assert wb.play_info["state"] == "done"


def test_play_with_rest_outside_the_zone(tmp_path):
    """Rest outside the calibrated zone (arms on the legs: allowed since 2026-11-04). The plan was
    accepted, but execute() checked the whole cycle again against the zone and refused it."""
    cfg, lo, hi = _cfg_wide_factory("LShoulderPitch")
    cfg.poses["rest"]["left_arm"]["LShoulderPitch"] = hi + 0.2
    wb = make_wb(tmp_path, cfg)
    wb.adapter.q["LShoulderPitch"] = wb.adapter.target["LShoulderPitch"] = hi + 0.2
    wb.plans["p"] = {"gen": None, "played": gesture(cfg), "plan": None}
    wb.ctrl.arm(confirm=True)
    r = wb.execute_plan("p", background=False, start="auto")
    assert r["start"] == "rest" and wb.play_info["state"] == "done", wb.play_info


def test_refusal_names_the_cycle_segment(gwb, cfg):
    out = _generate(gwb, cfg)
    gwb.plans[out["plan_id"]]["played"] = gesture(cfg, amp=1.5)
    plan, _ = gwb.plan_generated(gesture(cfg, amp=1.5), "auto", "rest")
    assert not plan.report.ok
    assert {v.segment for v in plan.report.violations} >= {"geste"}
    assert all(v.segment for v in plan.report.violations if v.time_s is not None)


def test_refusal_before_sending_does_not_leave_a_fault(gwb, cfg):
    """A play refused before the first command (robot moved since planning) sends nothing:
    the controller stays IDLE instead of FAULT, so the next Play works without acknowledging."""
    from nao_motion.control.controller import NothingSent, Plan, State
    from nao_motion.guardian import Report
    g = gesture(cfg)
    g.q = g.q + 0.3  # first point far from the measured posture
    plan = Plan(g, [("geste", 0.0, g.duration)], Report(True, []))
    gwb.ctrl.arm(confirm=True)
    with pytest.raises(NothingSent):
        gwb.ctrl.execute(plan, guardian=False)
    assert gwb.ctrl.state == State.IDLE and gwb.ctrl.armed
    assert gwb.adapter.command_log == []
