"""Extend a saved calibration with the recorded takes.

The takes are gestures done by hand on the real robot, like the calibration sweep:
they show where the joints can actually go. This builds a NEW calibration file (the
base one is never overwritten) whose ranges and coupled envelopes are the union of
the base calibration and the takes, with the same margin and the same rules as
`compute_calibration`. Factory limits stay the absolute ceiling.

Optionally, per-joint velocity / acceleration limits are proposed from the takes
(95th percentile of each take's peak, never below the current guardian limit). They
are stored in the calibration file (`kinematics`) and applied by the configuration
loader only when that calibration is active.

Nothing is activated here: the new file shows up in the list of saved calibrations
and is activated (or not) with the usual Activate button.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import (Calibration, CalibrationError, convex_hull, hull_constraints, is_simulated,
                          load_procedure, simplify_hull, zone_polygon)
from .config import Config
from .data.store import TakeStore

PERCENTILE = 95
MAX_POINTS = 600  # points per pair sent to the page


def _take_samples(cfg: Config, store: TakeStore, joints: Sequence[str], statuses) -> List[dict]:
    """Useful segment of each real, complete take with the given statuses."""
    out = []
    for m in store.list():
        if m.status not in statuses or m.synthetic:
            continue
        if "incomplete" in ((m.annotations or {}).get("tags") or []):
            continue
        take = store.load(m.take_id)
        js = [j for j in take.traj.joints if j in joints]
        if not js:
            continue
        seg = take.segment_trajectory(js)
        if len(seg.t) < 2:
            continue
        guided = [j for j in (m.guided_joints or []) if j in js]
        out.append({"id": m.take_id, "kind": m.kind, "joints": js, "q": seg.q, "take": take, "guided": guided})
    return out


def _base_polygon(cons: List[dict], a: str, b: str, margin: float, box) -> np.ndarray:
    """Envelope measured by the base calibration (margin removed again), as a polygon."""
    hp = [(float(c["terms"].get(a, 0.0)), float(c["terms"].get(b, 0.0)), float(c["bound"]) + margin) for c in cons]
    return zone_polygon(hp, box)


def _down(p: np.ndarray, n: int = MAX_POINTS) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.linspace(0, len(p) - 1, n).astype(int)]


def _kinematics(cfg: Config, samples: List[dict]) -> Dict[str, dict]:
    """Peak velocity / acceleration of each take on its GUIDED joints (moved by hand),
    measured on the smoothed replay (raw sensor noise would inflate accelerations)."""
    from .control.recording import replay_trajectory
    peaks_v: Dict[str, List[float]] = {}
    peaks_a: Dict[str, List[float]] = {}
    for s in samples:
        if not s["guided"]:
            continue
        try:
            tr = replay_trajectory(s["take"], s["guided"])
        except Exception:
            continue
        dt = np.diff(tr.t)
        vel = np.diff(tr.q, axis=0) / dt[:, None]
        acc = np.diff(vel, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None] if len(vel) >= 2 else None
        for k, j in enumerate(tr.joints):
            peaks_v.setdefault(j, []).append(float(np.abs(vel[:, k]).max()))
            if acc is not None:
                peaks_a.setdefault(j, []).append(float(np.abs(acc[:, k]).max()))
    g = cfg.guardian
    out = {}
    for j, pv in peaks_v.items():
        v = float(np.percentile(pv, PERCENTILE))
        a = float(np.percentile(peaks_a.get(j, [0.0]), PERCENTILE))
        out[j] = {"velocity_rad_s": round(max(v, g.velocity_limit(j)), 3),
                  "acceleration_rad_s2": round(max(a, g.acceleration_limit(j)), 3),
                  "current_velocity": g.velocity_limit(j), "current_acceleration": g.acceleration_limit(j),
                  "takes_velocity_p95": round(v, 3), "takes_acceleration_p95": round(a, 3), "n_takes": len(pv)}
    return out


def extend_with_takes(cfg: Config, store: TakeStore, base_path, statuses=("accepted",), speeds: bool = False,
                      version: Optional[str] = None):
    """-> (new Calibration, not saved; preview dict for the page)."""
    base = Calibration.load(base_path)
    if is_simulated(base):
        raise CalibrationError("Simulated calibration: it cannot be extended")
    proc = load_procedure(cfg.root)
    m = float(base.margin_rad)
    joints = [j for j in cfg.all_joints if j in base.limits]
    samples = _take_samples(cfg, store, joints, tuple(statuses))
    if not samples:
        raise CalibrationError("No usable take (real, complete, with the requested status)")

    # per-joint ranges: same rule as compute_calibration (measured range ∩ factory, minus margin)
    limits, measured, rows, warnings = {}, {}, [], list(base.warnings)
    for j in joints:
        vals = [s["q"][:, s["joints"].index(j)] for s in samples if j in s["joints"]]
        mb = dict((base.measured or {}).get(j) or {})
        lo0, hi0 = base.limits[j]
        if not vals:
            limits[j], measured[j] = (lo0, hi0), mb
            continue
        v = np.concatenate(vals)
        tlo, thi = float(v.min()), float(v.max())
        fac = mb.get("factory")
        if not fac:
            # no factory limits read for this joint: no ceiling known, so the range is not extended
            limits[j], measured[j] = (lo0, hi0), mb
            rows.append({"joint": j, "old": [lo0, hi0], "new": [lo0, hi0], "takes": [tlo, thi], "factory": None,
                         "extended": False, "capped_by_factory": False, "no_factory": True})
            warnings.append(f"{j}: factory limits unknown, range not extended with the takes")
            continue
        flo, fhi = float(fac[0]), float(fac[1])
        lo = max(min(tlo, float(mb.get("min", lo0 - m))), flo) + m
        hi = min(max(thi, float(mb.get("max", hi0 + m))), fhi) - m
        new = (min(lo, lo0), max(hi, hi0))
        limits[j] = new
        measured[j] = dict(mb, takes_min=tlo, takes_max=thi, n_takes=len(vals))
        rows.append({"joint": j, "old": [lo0, hi0], "new": list(new), "takes": [tlo, thi],
                     "factory": [flo, fhi],
                     "extended": bool(new[0] < lo0 - 1e-9 or new[1] > hi0 + 1e-9),
                     "capped_by_factory": bool(tlo < flo or thi > fhi)})

    # coupled envelopes: union of the base envelope and the takes' points, per measured pair
    coupled, pairs, seen = [], [], set()
    for a, b in proc.get("pairs", []):
        prefix = f"calib:{a}-{b}:"
        cons = [c for c in base.coupled if str(c.get("name", "")).startswith(prefix)]
        if not cons:
            continue
        seen.update(c["name"] for c in cons)
        mb_a, mb_b = (base.measured or {}).get(a) or {}, (base.measured or {}).get(b) or {}
        fa, fb = mb_a.get("factory"), mb_b.get("factory")
        if not fa or not fb:
            coupled.extend(cons)  # no ceiling known: envelope kept as is
            continue
        old_box = (base.limits[a][0], base.limits[a][1], base.limits[b][0], base.limits[b][1])
        old_zone = zone_polygon([(float(c["terms"].get(a, 0)), float(c["terms"].get(b, 0)), float(c["bound"]))
                                 for c in cons], old_box)
        base_poly = _base_polygon(cons, a, b, m, (float(fa[0]), float(fa[1]), float(fb[0]), float(fb[1])))
        pts = [s["q"][:, [s["joints"].index(a), s["joints"].index(b)]] for s in samples
               if a in s["joints"] and b in s["joints"]]
        if not pts:
            coupled.extend(cons)
            continue
        pts = np.vstack(pts)
        allp = np.vstack([base_poly, pts]) if len(base_poly) else pts
        hull = simplify_hull(convex_hull(allp), int(proc.get("max_hull_edges", 12)))
        new_cons = []
        for k, (na, nb, bound) in enumerate(hull_constraints(hull, m)):
            new_cons.append({"name": f"{prefix}{k}", "terms": {a: round(na, 6), b: round(nb, 6)},
                             "bound": round(bound, 6), "description": f"enveloppe mesurée {a}/{b} + prises"})
        coupled.extend(new_cons)
        new_box = (limits[a][0], limits[a][1], limits[b][0], limits[b][1])
        new_zone = zone_polygon([(c["terms"][a], c["terms"][b], c["bound"]) for c in new_cons], new_box)

        def inside(p, cs, box):
            return (box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]
                    and all(c["terms"].get(a, 0) * p[0] + c["terms"].get(b, 0) * p[1] <= c["bound"] + 1e-9 for c in cs))
        out_old = np.array([not inside(p, cons, old_box) for p in pts])
        pairs.append({"joints": [a, b], "factory": [list(map(float, fa)), list(map(float, fb))],
                      "old_zone": np.round(old_zone, 4).tolist(), "new_zone": np.round(new_zone, 4).tolist(),
                      "points_inside": np.round(_down(pts[~out_old]), 4).tolist(),
                      "points_outside": np.round(_down(pts[out_old]), 4).tolist(),
                      "n_points": int(len(pts)), "share_outside_before": round(float(out_old.mean()), 3),
                      "area_before": round(_area(old_zone), 4), "area_after": round(_area(new_zone), 4)})
    coupled.extend(c for c in base.coupled if c.get("name") not in seen)

    kin = _kinematics(cfg, samples) if speeds else {}
    robot = dict(base.robot or {})
    version = version or f"{robot.get('name') or 'nao'}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-takes"
    kinds: Dict[str, int] = {}
    for s in samples:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    notes = (f"extended from {base.version} with {len(samples)} takes "
             f"({', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}, status {'/'.join(statuses)})"
             + ("; speed limits from the takes (p95)" if kin else ""))
    cal = Calibration(version=version, robot=robot, limits=limits, measured=measured, poses=base.poses or {},
                      coupled=coupled, margin_rad=m, warnings=warnings, notes=notes)
    if kin:
        cal.kinematics = {"source": f"takes p{PERCENTILE}",
                          "velocity_rad_s": {j: k["velocity_rad_s"] for j, k in kin.items()},
                          "acceleration_rad_s2": {j: k["acceleration_rad_s2"] for j, k in kin.items()}}
    preview = {"base": base.version, "base_file": Path(base_path).name, "version": version, "n_takes": len(samples),
               "takes_by_kind": kinds, "joints": rows, "pairs": pairs, "kinematics": kin, "notes": notes}
    return cal, preview


def _area(poly) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
