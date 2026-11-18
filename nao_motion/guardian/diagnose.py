"""Why does the guardian refuse generated gestures? A batch diagnosis (nothing is sent).

For every emotion x intensity, the gesture is generated (fixed seed, no event), then checked at
three stages: the raw model output, the gesture made playable (slowed down / brought back in the
zone), and, when a robot is connected, the whole cycle from the measured posture. Refusals are
counted by check, joint and cycle segment, and the limits in force are listed with their source
(guardian.yaml defaults, active calibration, measured speeds of the takes), so that the reason a
gesture is refused is visible on the operator's own models and calibration.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..events import EventContext


def limits_in_force(cfg) -> dict:
    g = cfg.guardian
    kin = (getattr(cfg.calibration, "kinematics", None) or {}) if cfg.calibration is not None else {}
    measured_v = set((kin.get("velocity_rad_s") or {}))
    measured_a = dict(kin.get("acceleration_rad_s2") or {})
    cal_cc = {c["name"] for c in (cfg.calibration.coupled if cfg.calibration is not None else [])}
    rows = []
    for j in cfg.all_joints + list(cfg.hand_joints):
        rows.append({"joint": j, "zone": [round(cfg.guardian_bounds[j][0], 3), round(cfg.guardian_bounds[j][1], 3)],
                     "velocity": g.velocity_limit(j),
                     "velocity_source": g.velocity_sources.get(j) or ("takes" if j in measured_v else "guardian.yaml"),
                     "acceleration": g.acceleration_limit(j),
                     "acceleration_source": "takes" if j in measured_a and abs(float(measured_a[j]) - g.acceleration_limit(j)) < 1e-9
                     else "guardian.yaml"})
    return {"calibrated": cfg.calibrated, "calibration": cfg.calibration_version, "joints": rows,
            "max_duration_s": g.max_duration_s,
            "coupled": [{"name": c.name, "terms": c.terms, "bound": c.bound,
                         "source": "calibration" if c.name in cal_cc else "guardian.yaml"}
                        for c in g.coupled_constraints]}


def takes_through_guardian(wb, limit: int = 200, include_simulated: bool = False) -> dict:
    """The recorded takes, played back as they are, through the same checks as a generated
    gesture: raw, made playable, and the whole cycle from the measured posture when a robot is
    connected (nothing is sent). They were done by hand on the robot, so every refusal points at
    the guardian, not at the take. No model is involved."""
    from ..control.recording import replay_trajectory
    cfg, G = wb.cfg, wb.guardian
    connected = wb.adapter.is_connected()
    joints = list(cfg.arm_joints)
    others = [j for j in cfg.all_joints if j not in joints]
    ctx = dict(zip(others, cfg.pose("start", others)))
    counts = {"raw": Counter(), "playable": Counter(), "cycle": Counter()}
    ok = {"raw": 0, "playable": 0, "cycle": 0}
    n, refused = 0, []
    metas = [m for m in wb.store.list(kind="arms") if m.status != "rejected"
             and (include_simulated or not m.synthetic)][:limit]
    for m in metas:
        try:
            traj = replay_trajectory(wb.store.load(m.take_id), joints)
        except Exception:
            continue
        n += 1
        raw = G.validate(traj, context_pose=ctx, check_duration=False)
        played, _ = wb.playable(traj)
        rep = G.validate(played, context_pose=ctx)
        reports = [("raw", raw), ("playable", rep)]
        if connected:
            try:
                plan, _ = wb.plan_generated(played, "auto", "rest")
                reports.append(("cycle", plan.report))
            except Exception as ex:
                counts["cycle"][("plan_error", "", str(ex)[:80])] += 1
        for stage, r in reports:
            ok[stage] += r.ok
            for v in r.violations:
                counts[stage][(v.code, v.joint or "", v.segment or "")] += 1
        if not all(r.ok for st, r in reports if st != "raw"):
            refused.append(m.take_id)

    def table(c: Counter) -> List[dict]:
        return [{"code": k[0], "joint": k[1], "segment": k[2], "takes": v} for k, v in c.most_common()]
    return {"n": n, "connected": connected, "ok": ok, "refusals": {k: table(v) for k, v in counts.items()},
            "refused_takes": refused[:20]}


def diagnose(wb, intensities: Sequence[float] = (0.3, 0.6, 0.9), project_zone: Optional[bool] = None,
             seed: int = 0) -> dict:
    cfg, G = wb.cfg, wb.guardian
    eng = wb.get_engine()
    if project_zone is None:
        project_zone = bool(cfg.robot.get("execution", {}).get("project_control_points", True))
    connected = wb.adapter.is_connected()
    counts = {"raw": Counter(), "playable": Counter(), "cycle": Counter()}
    ok = {"raw": 0, "playable": 0, "cycle": 0}
    scales, capped, n, per_gesture = [], 0, 0, []
    for e in cfg.emotions:
        for i in intensities:
            gen = eng.generate(e, float(i), EventContext.none(), seed=seed, project_zone=project_zone)
            n += 1
            raw = G.validate(gen.combined)
            played, adapt = wb.playable(gen.combined)
            rep = G.validate(played)
            row = {"emotion": e, "intensity": i, "raw": raw.ok, "playable": rep.ok}
            for stage, r in (("raw", raw), ("playable", rep)):
                ok[stage] += r.ok
                for v in r.violations:
                    counts[stage][(v.code, v.joint or "", "")] += 1
            if adapt is not None:
                scales.append(adapt.time_scale)
                capped += any("capped" in x for x in adapt.notes)
                row["slow_down"] = round(adapt.time_scale, 2)
            if connected:
                try:
                    plan, start = wb.plan_generated(played, "auto", "rest")
                    ok["cycle"] += plan.report.ok
                    row["cycle"] = plan.report.ok
                    for v in plan.report.violations:
                        counts["cycle"][(v.code, v.joint or "", v.segment or "")] += 1
                except Exception as ex:  # not at rest with start=rest, etc.
                    counts["cycle"][("plan_error", "", str(ex)[:80])] += 1
                    row["cycle"] = False
            per_gesture.append(row)

    def table(c: Counter) -> List[dict]:
        return [{"code": k[0], "joint": k[1], "segment": k[2], "gestures": v} for k, v in c.most_common()]

    lim = limits_in_force(cfg)
    try:
        takes = takes_through_guardian(wb)
    except Exception as ex:  # no store, unreadable take...
        takes = {"error": str(ex)[:120]}
    out = {"takes": takes, "n": n, "project_zone": project_zone, "connected": connected, "ok": ok,
           "refusals": {k: table(v) for k, v in counts.items()},
           "slow_down": {"median": round(float(np.median(scales)), 2) if scales else None,
                         "max": round(float(np.max(scales)), 2) if scales else None,
                         "capped_by_max_duration": capped},
           "gestures": per_gesture, "limits": lim}
    out["hints"] = hints(out)
    return out


def hints(d: dict) -> List[str]:
    """Plain reading of the counts (English, as the UI)."""
    h, lim = [], d["limits"]
    joints = {r["joint"]: r for r in lim["joints"]}
    raw = d["refusals"]["raw"]
    kin_codes = [r for r in raw if r["code"] in ("velocity", "acceleration")]
    defaults = [r for r in kin_codes if joints.get(r["joint"], {}).get(r["code"] + "_source", "takes") != "takes"]
    if defaults:
        js = sorted({r["joint"] for r in defaults})
        shown = ", ".join(f"{j} {joints[j]['velocity']} rad/s, {joints[j]['acceleration']} rad/s²" for j in js[:4])
        h.append(f"Speed limits: the raw gestures exceed the DEFAULT limits on {js} ({shown}{', ...' if len(js) > 4 else ''}): "
                 f"guardian.yaml values or a fraction of the factory speed, not measured on your gestures. "
                 f"Use Calibration > Extend with the takes with « speeds » ticked to use the speeds of "
                 f"your takes (fastest take), or raise them in guardian.yaml (velocity_fraction_of_factory).")
    sd = d["slow_down"]
    if sd["median"] and sd["median"] > 1.5:
        h.append(f"To respect those limits, gestures are slowed down x{sd['median']} (median, max x{sd['max']}): "
                 f"a quick oscillation such as waving becomes a slow sway.")
    if sd["capped_by_max_duration"]:
        h.append(f"{sd['capped_by_max_duration']} gesture(s) would need to last more than "
                 f"{lim['max_duration_s']} s to respect the limits: they stay refused (velocity / acceleration).")
    cc = [r for r in raw if r["code"] == "coupled"]
    examples = {c["name"] for c in lim["coupled"] if c["source"] == "guardian.yaml"}
    if cc and examples:
        h.append(f"Coupled constraints from guardian.yaml are EXAMPLES, never measured on your robot "
                 f"({sorted(examples)}); they are hit by {sum(r['gestures'] for r in cc)} refusal(s).")
    lims = [r for r in raw if r["code"] == "limit"]
    if lims:
        js = sorted({r["joint"] for r in lims})
        h.append(f"Calibrated zone: raw gestures leave it on {js}. Projection / adaptation bring them back; "
                 f"if the zone is narrower than your takes, use Calibration > Extend with the takes.")
    tk = d.get("takes") or {}
    if tk.get("n"):
        stage = "cycle" if tk["connected"] else "playable"
        bad = tk["n"] - tk["ok"][stage]
        if bad:
            h.append(f"YOUR TAKES: {bad}/{tk['n']} recorded takes are refused ({'whole cycle' if stage == 'cycle' else 'made playable'}). "
                     f"They were done on the robot, so these refusals are guardian errors, not take errors: see the "
                     f"takes table below.")
    cyc = d["refusals"]["cycle"]
    if cyc:
        segs = Counter(r["segment"] for r in cyc for _ in range(r["gestures"]))
        h.append(f"Whole cycle refusals by segment: {dict(segs)}.")
    if not h:
        h.append("Nothing refused on this batch.")
    return h
