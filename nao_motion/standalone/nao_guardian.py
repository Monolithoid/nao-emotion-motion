# -*- coding: utf-8 -*-
"""Standalone guardian for NAO: checks a movement BEFORE sending it to the robot.

This file is deliberately self-contained: compatible with Python 2.7 (NAOqi
2.1, OpenNAO) AND Python 3, standard library only (no numpy).
It reads a `guardian.json` file exported by nao-emotion-motion
(`nao-motion export-guardian`) and applies exactly the same rules as the
project's guardian:

  - known joints only, legs always refused;
  - min/max limits (measured by the calibration), minus a margin;
  - linear coupled constraints sum(coef * q) <= bound (areas measured by
    the calibration + manual constraints, e.g. arm <-> head);
  - maximum velocity and acceleration, minimum and maximum duration;
  - first point close to the measured state (optional).

Hands (LHand, RHand): their "angle" is an opening between 0 (closed) and
1 (open), bounded by the factory range. Margin and acceleration can be
overridden per joint (`per_joint_margin`, `per_joint_acceleration`).

The full path is checked (linear interpolation, fine grid), not just its
end points. The guardian NEVER MODIFIES a movement: it accepts or refuses
it, listing the violations. It commands nothing and does not depend on
NAOqi.

What it does not guarantee: absence of collision beyond the configured
constraints, nor the electrical or software stability of the robot.

Library usage:

    from nao_guardian import Guardian
    g = Guardian.load("guardian.json")
    rep = g.check_pose({"HeadYaw": 0.3, "HeadPitch": -0.1})
    rep = g.check_move(current_pose, target_pose, duration_s=1.2)
    rep = g.check_trajectory(times, joints, positions, start_state=current_pose)
    if not rep.ok:
        print(rep.summary())       # do not send

Command line:

    python nao_guardian.py guardian.json trajectory.json   # exit code 0 = accepted
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import io
import json
import math
import sys

# <guardian-core> (copied as is into nao_gesture.py by `nao-motion export-generator`)
KIND = "nao_motion.guardian_export"
SCHEMA_VERSION = 1
LEG_PREFIXES = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
MAX_LISTED = 50

class GuardianError(ValueError):
    """Invalid guardian file or incorrect usage."""


class Violation(object):
    def __init__(self, code, message, joint=None, time_s=None, value=None, limit=None):
        self.code, self.message = code, message
        self.joint, self.time_s, self.value, self.limit = joint, time_s, value, limit

    def to_dict(self):
        d = {"code": self.code, "message": self.message}
        for k in ("joint", "time_s", "value", "limit"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    def __repr__(self):
        return "Violation(%r)" % (self.to_dict(),)


class Report(object):
    def __init__(self, ok, violations=None, stats=None):
        self.ok = bool(ok)
        self.violations = list(violations or [])
        self.stats = dict(stats or {})

    def to_dict(self):
        return {"ok": self.ok, "violations": [v.to_dict() for v in self.violations[:MAX_LISTED]],
                "n_violations": len(self.violations), "stats": self.stats}

    def summary(self):
        if self.ok:
            return "mouvement accepté"
        codes = sorted(set(v.code for v in self.violations))
        return "mouvement refusé (%d violations : %s)" % (len(self.violations), ", ".join(codes))

    def __bool__(self):
        return self.ok

    __nonzero__ = __bool__  # Python 2


# -- numeric helpers (no numpy) ----------------------------------------------------
def _finite(x):
    return not (math.isinf(x) or math.isnan(x))


def _linspace(a, b, n):
    """Like numpy.linspace: a + i * step, last point exactly b."""
    if n == 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    out = [a + i * step for i in range(n)]
    out[-1] = float(b)
    return out


def _interp(x, xs, ys):
    """Like numpy.interp (xs strictly increasing)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _sample(ts, cols, tt):
    """cols: one list of values per joint -> sampled rows."""
    return [[_interp(x, ts, c) for c in cols] for x in tt]


# -- guardian ----------------------------------------------------------------------------
class Guardian(object):
    def __init__(self, data, allow_uncalibrated=False):
        if not isinstance(data, dict) or data.get("kind") != KIND:
            raise GuardianError("Ce n'est pas un fichier guardian exporté par nao-emotion-motion")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise GuardianError("Version de schéma non prise en charge : %r" % (data.get("schema_version"),))
        if data.get("unit") != "rad" or data.get("time_unit") != "s":
            raise GuardianError("Unités attendues : rad et s")
        src = data.get("source") or {}
        self.calibrated = bool(src.get("calibrated"))
        if not self.calibrated and not allow_uncalibrated:
            raise GuardianError("Limites de remplacement (aucune calibration mesurée) : refusé pour un robot réel. "
                                "Exporter après une calibration, ou passer allow_uncalibrated=True pour un essai.")
        self.data = data
        self.source = src
        self.limits = {}
        for j, lh in data["limits"].items():
            lo, hi = float(lh[0]), float(lh[1])
            if not (_finite(lo) and _finite(hi) and lo < hi):
                raise GuardianError("Limite invalide pour %s" % j)
            if j.startswith(LEG_PREFIXES):
                raise GuardianError("Les jambes ne peuvent pas figurer dans le guardian : %s" % j)
            self.limits[j] = (lo, hi)
        self.margin = float(data["limit_margin_rad"])
        # optional overrides (LHand/RHand hands: opening 0..1, full range)
        self.per_joint_margin = dict((k, float(v)) for k, v in (data.get("per_joint_margin") or {}).items())
        self.per_joint_acceleration = dict((k, float(v))
                                           for k, v in (data.get("per_joint_acceleration") or {}).items())
        vel = data["velocity_rad_s"]
        self.max_velocity = float(vel["default"])
        self.per_joint_velocity = dict((k, float(v)) for k, v in (vel.get("per_joint") or {}).items())
        self.max_acceleration = float(data["max_acceleration_rad_s2"])
        self.check_rate_hz = float(data["check_rate_hz"])
        self.min_duration_s, self.max_duration_s = [float(x) for x in data["duration_s"]]
        self.start_tolerance = float(data["start_tolerance_rad"])
        self.constraints = []
        for c in data.get("coupled_constraints") or []:
            terms = dict((k, float(v)) for k, v in c["terms"].items())
            for k in terms:
                if k not in self.limits:
                    raise GuardianError("Contrainte %s : articulation inconnue %s" % (c["name"], k))
            self.constraints.append({"name": c["name"], "terms": terms, "bound": float(c["bound"])})
        self.groups = data.get("groups") or {}
        self.poses = data.get("poses") or {}
        # rest <-> start segments: rest may be outside the zone (contact); bounded by the
        # factory limits and a velocity (same rules as the project guardian): one value for
        # every joint, or "factory" = the per-joint gesture limits (velocity_limit)
        self.rest_limits = dict((j, (float(lh[0]), float(lh[1])))
                                for j, lh in (data.get("rest_limits") or data["limits"]).items())
        rv = data.get("rest_velocity_rad_s", self.max_velocity)
        self.rest_velocity_factory = rv == "factory"
        if self.rest_velocity_factory:
            # single value kept for callers of older versions: the lowest per-joint limit
            self.rest_velocity = min([self.velocity_limit(j) for j in self.rest_limits] or [self.max_velocity])
        else:
            self.rest_velocity = float(rv)
        self.transition_velocity = float(data.get("transition_velocity_rad_s", self.max_velocity))

    @classmethod
    def load(cls, path, allow_uncalibrated=False):
        with io.open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), allow_uncalibrated=allow_uncalibrated)

    # -- accessors -----------------------------------------------------------------------
    @property
    def joints(self):
        return sorted(self.limits)

    def allowed_range(self, joint):
        """[min, max] actually allowed (margin included)."""
        lo, hi = self.limits[joint]
        m = self.per_joint_margin.get(joint, self.margin)
        return lo + m, hi - m

    def velocity_limit(self, joint):
        return self.per_joint_velocity.get(joint, self.max_velocity)

    def acceleration_limit(self, joint):
        return self.per_joint_acceleration.get(joint, self.max_acceleration)

    def rest_velocity_limit(self, joint):
        """Velocity limit of the rest <-> start segments for this joint."""
        return self.velocity_limit(joint) if self.rest_velocity_factory else self.rest_velocity

    def rest_transition_caps(self, joints):
        """(velocity, acceleration) caps per joint used to BUILD rest <-> start segments.
        "factory": 0.8 x the gesture limits, like the start <-> gesture transitions; otherwise
        0.8 x rest_velocity and 0.25 x max acceleration (former rule)."""
        if self.rest_velocity_factory:
            return ([self.velocity_limit(j) * 0.8 for j in joints],
                    [self.acceleration_limit(j) * 0.8 for j in joints])
        return [self.rest_velocity * 0.8] * len(joints), [self.max_acceleration * 0.25] * len(joints)

    def pose(self, name):
        """Calibrated "rest" or "start" pose: {joint: rad}."""
        return dict(self.poses.get(name) or {})

    def min_duration(self, start, target):
        """Minimum duration (s) of a linear move start -> target within the allowed velocities."""
        d = self.min_duration_s
        for j, v in target.items():
            if j in start and j in self.limits:
                d = max(d, abs(float(v) - float(start[j])) / self.velocity_limit(j))
        return d

    # -- checks -----------------------------------------------------------------
    def _joint_names(self, joints):
        v = []
        for j in joints:
            if j.startswith(LEG_PREFIXES):
                v.append(Violation("legs", "Les jambes sont hors périmètre", joint=j))
            elif j not in self.limits:
                v.append(Violation("unknown_joint", "Articulation non configurée", joint=j))
        return v

    def _coupled(self, joints, rows, tt, context, v):
        ctx = dict(context or {})
        idx = dict((j, k) for k, j in enumerate(joints))
        for c in self.constraints:
            usable = all(j in idx or j in ctx for j in c["terms"])
            if not usable:
                continue
            first, count, first_val = None, 0, None
            for i, row in enumerate(rows):
                total = 0.0
                for j, coef in c["terms"].items():
                    total += coef * (row[idx[j]] if j in idx else float(ctx[j]))
                if total > c["bound"]:
                    count += 1
                    if first is None:
                        first, first_val = i, total
            if count:
                v.append(Violation("coupled", "Contrainte couplée « %s » violée sur %d points" % (c["name"], count),
                                   time_s=None if tt is None else float(tt[first]), value=float(first_val),
                                   limit=c["bound"]))

    def check_pose(self, pose, context=None):
        """A target posture {joint: rad}: limits (margin included) and coupled constraints.
        `context`: angles of the joints that are not commanded (e.g. head held)."""
        joints = list(pose)
        if not all(hasattr(j, "startswith") for j in joints):
            return Report(False, [Violation("format", "Noms d'articulations invalides")])
        v = self._joint_names(joints)
        if v:
            return Report(False, v)
        row = []
        for j in joints:
            x = float(pose[j])
            if not _finite(x):
                return Report(False, [Violation("format", "Valeur non finie", joint=j)])
            row.append(x)
            lo, hi = self.allowed_range(j)
            if x < lo or x > hi:
                v.append(Violation("limit", "Hors limites (marge incluse)", joint=j, value=x, limit=lo if x < lo else hi))
        self._coupled(joints, [row], None, context, v)
        return Report(not v, v)

    def check_move(self, start, target, duration_s, context=None, check_duration=True):
        """Linear move from `start` to `target` in `duration_s` seconds
        (joints of `target`; `start` = measured posture). See min_duration()."""
        joints = list(target)
        missing = [j for j in joints if j not in start]
        if missing:
            return Report(False, [Violation("format", "Position de départ inconnue pour %s" % ", ".join(missing))])
        times = [0.0, float(duration_s)]
        positions = [[float(start[j]) for j in joints], [float(target[j]) for j in joints]]
        return self.check_trajectory(times, joints, positions, context=context, check_duration=check_duration)

    def check_trajectory(self, times, joints, positions, start_state=None, context=None, check_duration=True,
                         expected_joints=None):
        """times: N strictly increasing timestamps (s); joints: J names;
        positions: N rows of J angles (rad). Same check as the project's guardian."""
        try:
            ts = [float(x) for x in times]
            joints = list(joints)
            qs = [[float(x) for x in row] for row in positions]
        except (TypeError, ValueError) as e:
            return Report(False, [Violation("format", "Trajectoire invalide : %s" % e)])
        err = None
        if len(qs) != len(ts) or any(len(r) != len(joints) for r in qs):
            err = "Forme incohérente : %d horodatages, %d lignes, %d articulations" % (len(ts), len(qs), len(joints))
        elif len(ts) < 2:
            err = "Au moins deux échantillons requis"
        elif not all(_finite(x) for x in ts) or not all(_finite(x) for r in qs for x in r):
            err = "Valeurs non finies"
        elif any(b - a <= 0 for a, b in zip(ts, ts[1:])):
            err = "Horodatages non strictement croissants"
        elif not all(hasattr(j, "startswith") for j in joints):
            err = "Noms d'articulations invalides"
        elif len(set(joints)) != len(joints):
            err = "Noms d'articulations dupliqués"
        if err:
            return Report(False, [Violation("format", err)])

        v = self._joint_names(joints)
        if expected_joints is not None and set(expected_joints) != set(joints):
            v.append(Violation("incompatible", "Articulations attendues %s, reçues %s"
                               % (sorted(expected_joints), sorted(joints))))
        if v:
            return Report(False, v)

        J = len(joints)
        cols = [[r[k] for r in qs] for k in range(J)]
        dur = ts[-1] - ts[0]
        if check_duration and not (self.min_duration_s <= dur <= self.max_duration_s):
            v.append(Violation("duration", "Durée hors bornes", value=dur,
                               limit=self.max_duration_s if dur > self.max_duration_s else self.min_duration_s))

        # full path: original samples + fine grid
        n = max(2, int(math.ceil(dur * self.check_rate_hz)) + 1)
        tt = sorted(set(ts) | set(_linspace(ts[0], ts[-1], n)))
        rows = _sample(ts, cols, tt)
        for k, j in enumerate(joints):
            lo, hi = self.allowed_range(j)
            bad = [i for i, r in enumerate(rows) if r[k] < lo or r[k] > hi]
            if bad:
                i = bad[0]
                x = rows[i][k]
                v.append(Violation("limit", "Hors limites sur %d points (marge incluse)" % len(bad), joint=j,
                                   time_s=float(tt[i]), value=float(x), limit=lo if x < lo else hi))

        # velocities and accelerations of the executed piecewise-linear path
        dt = [b - a for a, b in zip(ts, ts[1:])]
        vel = [[(qs[i + 1][k] - qs[i][k]) / dt[i] for k in range(J)] for i in range(len(dt))]
        max_v = [max(abs(r[k]) for r in vel) for k in range(J)]
        for k, j in enumerate(joints):
            lim = self.velocity_limit(j)
            if max_v[k] > lim:
                i = max(range(len(vel)), key=lambda m: abs(vel[m][k]))
                v.append(Violation("velocity", "Vitesse excessive", joint=j, time_s=float(ts[i]),
                                   value=float(max_v[k]), limit=lim))
        max_a = [0.0] * J
        if len(vel) >= 2:
            acc = [[(vel[i + 1][k] - vel[i][k]) / ((dt[i + 1] + dt[i]) / 2) for k in range(J)]
                   for i in range(len(vel) - 1)]
            max_a = [max(abs(r[k]) for r in acc) for k in range(J)]
            for k, j in enumerate(joints):
                alim = self.acceleration_limit(j)
                if max_a[k] > alim:
                    i = max(range(len(acc)), key=lambda m: abs(acc[m][k]))
                    v.append(Violation("acceleration", "Accélération excessive", joint=j, time_s=float(ts[i + 1]),
                                       value=float(max_a[k]), limit=alim))

        # coupled constraints over the whole path (including head <-> arm through `context`)
        self._coupled(joints, rows, tt, context, v)

        if start_state is not None:
            for k, j in enumerate(joints):
                if j in start_state:
                    d = abs(qs[0][k] - float(start_state[j]))
                    if d > self.start_tolerance:
                        v.append(Violation("start_mismatch", "Premier point éloigné de l'état mesuré", joint=j,
                                           value=d, limit=self.start_tolerance))

        stats = {"duration_s": dur, "n_checked_points": len(tt),
                 "max_velocity": dict((j, round(max_v[k], 4)) for k, j in enumerate(joints)),
                 "max_acceleration": dict((j, round(max_a[k], 4)) for k, j in enumerate(joints))}
        if any(x.code in ("velocity", "acceleration") for x in v):
            # slow-down factor that would be enough (information only, never applied)
            sv = max(max_v[k] / self.velocity_limit(j) for k, j in enumerate(joints))
            sa = math.sqrt(max(max_a[k] / self.acceleration_limit(j) for k, j in enumerate(joints))) if max_a else 0.0
            stats["suggested_time_scale"] = round(float(max(sv, sa, 1.0)), 3)
        return Report(not v, v, stats)

    def check_rest_transition(self, times, joints, positions, start_state=None, time_offset=0.0):
        """Rest <-> start segment, the ONLY one allowed to leave the movement zone (rest in
        contact, arms on the knees...): factory limits, rest velocity (rest_velocity_rad_s)
        and acceleration; no margin, no coupled constraints. That this path is free of
        collisions is up to the calibration (start reachable from rest)."""
        try:
            ts = [float(x) for x in times]
            joints = list(joints)
            qs = [[float(x) for x in row] for row in positions]
        except (TypeError, ValueError) as e:
            return Report(False, [Violation("format", "Invalid trajectory: %s" % e)])
        if (len(qs) != len(ts) or len(ts) < 2 or any(len(r) != len(joints) for r in qs)
                or not all(_finite(x) for x in ts) or not all(_finite(x) for r in qs for x in r)
                or any(b - a <= 0 for a, b in zip(ts, ts[1:])) or len(set(joints)) != len(joints)):
            return Report(False, [Violation("format", "Invalid trajectory")])
        for j in joints:
            if not hasattr(j, "startswith") or j.startswith(LEG_PREFIXES) or j not in self.rest_limits:
                return Report(False, [Violation("unknown_joint", "Joint not configured", joint=j)])
        v = []
        J = len(joints)
        cols = [[r[k] for r in qs] for k in range(J)]
        dur = ts[-1] - ts[0]
        n = max(2, int(math.ceil(dur * self.check_rate_hz)) + 1)
        tt = sorted(set(ts) | set(_linspace(ts[0], ts[-1], n)))
        rows = _sample(ts, cols, tt)
        for k, j in enumerate(joints):
            lo, hi = self.rest_limits[j]
            bad = [i for i, r in enumerate(rows) if r[k] < lo or r[k] > hi]
            if bad:
                x = rows[bad[0]][k]
                v.append(Violation("rest_limit", "Rest <-> start segment outside factory limits", joint=j,
                                   time_s=float(tt[bad[0]]) + time_offset, value=float(x), limit=lo if x < lo else hi))
        dt = [b - a for a, b in zip(ts, ts[1:])]
        vel = [[(qs[i + 1][k] - qs[i][k]) / dt[i] for k in range(J)] for i in range(len(dt))]
        for k, j in enumerate(joints):
            i = max(range(len(vel)), key=lambda m: abs(vel[m][k]))
            lim = self.rest_velocity_limit(j)
            if abs(vel[i][k]) > lim:
                v.append(Violation("rest_velocity", "Rest <-> start segment too fast", joint=j,
                                   time_s=float(ts[i]) + time_offset, value=float(abs(vel[i][k])),
                                   limit=lim))
        if len(vel) >= 2:
            acc = [[(vel[i + 1][k] - vel[i][k]) / ((dt[i + 1] + dt[i]) / 2) for k in range(J)]
                   for i in range(len(vel) - 1)]
            for k, j in enumerate(joints):
                i = max(range(len(acc)), key=lambda m: abs(acc[m][k]))
                if abs(acc[i][k]) > self.acceleration_limit(j):
                    v.append(Violation("acceleration", "Excessive acceleration (rest <-> start segment)", joint=j,
                                       time_s=float(ts[i + 1]) + time_offset, value=float(abs(acc[i][k])),
                                       limit=self.acceleration_limit(j)))
        if start_state is not None:
            for k, j in enumerate(joints):
                if j in start_state and abs(qs[0][k] - float(start_state[j])) > self.start_tolerance:
                    v.append(Violation("start_mismatch", "First point far from the measured state", joint=j,
                                       value=abs(qs[0][k] - float(start_state[j])), limit=self.start_tolerance))
        return Report(not v, v, {"duration_s": dur, "rest_transition": True})

    def check_trajectory_dict(self, d, **kw):
        """nao-emotion-motion trajectory format: {"joints", "t", "q", "unit": "rad", "time_unit": "s"}."""
        if not isinstance(d, dict):
            return Report(False, [Violation("format", "Trajectoire invalide : dictionnaire attendu")])
        if d.get("unit", "rad") != "rad" or d.get("time_unit", "s") != "s":
            return Report(False, [Violation("format", "Trajectoire invalide : unités attendues rad et s")])
        try:
            return self.check_trajectory(d["t"], d["joints"], d["q"], **kw)
        except KeyError as e:
            return Report(False, [Violation("format", "Trajectoire invalide : champ manquant %s" % e)])


# </guardian-core>


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    allow = "--allow-uncalibrated" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print("usage : python nao_guardian.py guardian.json trajectoire.json [--allow-uncalibrated]")
        return 2
    g = Guardian.load(args[0], allow_uncalibrated=allow)
    with io.open(args[1], "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "trajectory" in d:  # file produced by nao-motion generate --out
        d = d["trajectory"]
    rep = g.check_trajectory_dict(d)
    out = json.dumps(rep.to_dict(), indent=1, ensure_ascii=False)
    if not isinstance(out, str):  # Python 2 : unicode -> octets
        out = out.encode("utf-8")
    print(out)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
