# -*- coding: utf-8 -*-
"""Guardian autonome pour NAO : vérifie un mouvement AVANT de l'envoyer au robot.

Ce fichier est volontairement indépendant : compatible Python 2.7 (NAOqi
2.1, OpenNAO) ET Python 3, bibliothèque standard uniquement (pas de numpy).
Il lit un fichier `guardian.json` exporté par nao-emotion-motion
(`nao-motion export-guardian`) et applique exactement les mêmes règles que le
guardian du projet :

  - articulations connues uniquement, jambes toujours refusées ;
  - limites min/max (mesurées par la calibration), moins une marge ;
  - contraintes couplées linéaires sum(coef * q) <= bound (zones mesurées par
    la calibration + contraintes manuelles, par ex. bras <-> tête) ;
  - vitesse et accélération maximales, durée minimale et maximale ;
  - premier point proche de l'état mesuré (optionnel).

Mains (LHand, RHand) : leur « angle » est une ouverture entre 0 (fermée) et
1 (ouverte), bornée par la plage d'usine. Marge et accélération peuvent être
surchargées par articulation (`per_joint_margin`, `per_joint_acceleration`).

Le trajet complet est vérifié (interpolation linéaire, grille fine), pas
seulement ses extrémités. Le guardian NE MODIFIE JAMAIS un mouvement : il
l'accepte ou le refuse en listant les violations. Il ne commande rien et ne
dépend pas de NAOqi.

Ce qu'il ne garantit pas : l'absence de collision au-delà des contraintes
configurées, ni la stabilité électrique ou logicielle du robot.

Utilisation en bibliothèque :

    from nao_guardian import Guardian
    g = Guardian.load("guardian.json")
    rep = g.check_pose({"HeadYaw": 0.3, "HeadPitch": -0.1})
    rep = g.check_move(current_pose, target_pose, duration_s=1.2)
    rep = g.check_trajectory(times, joints, positions, start_state=current_pose)
    if not rep.ok:
        print(rep.summary())       # ne pas envoyer

En ligne de commande :

    python nao_guardian.py guardian.json trajectoire.json   # code retour 0 = acceptée
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import io
import json
import math
import sys

KIND = "nao_motion.guardian_export"
SCHEMA_VERSION = 1
LEG_PREFIXES = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
MAX_LISTED = 50

class GuardianError(ValueError):
    """Fichier guardian invalide ou usage incorrect."""


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


# -- outils numériques (sans numpy) ----------------------------------------------------
def _finite(x):
    return not (math.isinf(x) or math.isnan(x))


def _linspace(a, b, n):
    """Comme numpy.linspace : a + i * pas, dernier point exactement b."""
    if n == 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    out = [a + i * step for i in range(n)]
    out[-1] = float(b)
    return out


def _interp(x, xs, ys):
    """Comme numpy.interp (xs strictement croissant)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _sample(ts, cols, tt):
    """cols : une liste de valeurs par articulation -> lignes échantillonnées."""
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
        # surcharges optionnelles (mains LHand/RHand : ouverture 0..1, plage entière)
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

    @classmethod
    def load(cls, path, allow_uncalibrated=False):
        with io.open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), allow_uncalibrated=allow_uncalibrated)

    # -- accès -----------------------------------------------------------------------
    @property
    def joints(self):
        return sorted(self.limits)

    def allowed_range(self, joint):
        """[min, max] réellement autorisé (marge comprise)."""
        lo, hi = self.limits[joint]
        m = self.per_joint_margin.get(joint, self.margin)
        return lo + m, hi - m

    def velocity_limit(self, joint):
        return self.per_joint_velocity.get(joint, self.max_velocity)

    def acceleration_limit(self, joint):
        return self.per_joint_acceleration.get(joint, self.max_acceleration)

    def pose(self, name):
        """Pose calibrée « rest » ou « start » : {articulation: rad}."""
        return dict(self.poses.get(name) or {})

    def min_duration(self, start, target):
        """Durée minimale (s) d'un déplacement linéaire start -> target sous les vitesses permises."""
        d = self.min_duration_s
        for j, v in target.items():
            if j in start and j in self.limits:
                d = max(d, abs(float(v) - float(start[j])) / self.velocity_limit(j))
        return d

    # -- vérifications -----------------------------------------------------------------
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
        """Une posture cible {articulation: rad} : limites (marge comprise) et contraintes couplées.
        `context` : angles des articulations non commandées (ex. tête maintenue)."""
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
        """Déplacement linéaire de `start` vers `target` en `duration_s` secondes
        (articulations de `target` ; `start` = posture mesurée). Voir min_duration()."""
        joints = list(target)
        missing = [j for j in joints if j not in start]
        if missing:
            return Report(False, [Violation("format", "Position de départ inconnue pour %s" % ", ".join(missing))])
        times = [0.0, float(duration_s)]
        positions = [[float(start[j]) for j in joints], [float(target[j]) for j in joints]]
        return self.check_trajectory(times, joints, positions, context=context, check_duration=check_duration)

    def check_trajectory(self, times, joints, positions, start_state=None, context=None, check_duration=True,
                         expected_joints=None):
        """times : N horodatages (s) strictement croissants ; joints : J noms ;
        positions : N lignes de J angles (rad). Même vérification que le guardian du projet."""
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

        # trajet complet : échantillons d'origine + grille fine
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

        # vitesses et accélérations du trajet linéaire par morceaux exécuté
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

        # contraintes couplées sur tout le trajet (y compris tête <-> bras via `context`)
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
            # facteur de ralentissement qui suffirait (information seulement, jamais appliqué)
            sv = max(max_v[k] / self.velocity_limit(j) for k, j in enumerate(joints))
            sa = math.sqrt(max(max_a[k] / self.acceleration_limit(j) for k, j in enumerate(joints))) if max_a else 0.0
            stats["suggested_time_scale"] = round(float(max(sv, sa, 1.0)), 3)
        return Report(not v, v, stats)

    def check_trajectory_dict(self, d, **kw):
        """Format des trajectoires de nao-emotion-motion : {"joints", "t", "q", "unit": "rad", "time_unit": "s"}."""
        if not isinstance(d, dict):
            return Report(False, [Violation("format", "Trajectoire invalide : dictionnaire attendu")])
        if d.get("unit", "rad") != "rad" or d.get("time_unit", "s") != "s":
            return Report(False, [Violation("format", "Trajectoire invalide : unités attendues rad et s")])
        try:
            return self.check_trajectory(d["t"], d["joints"], d["q"], **kw)
        except KeyError as e:
            return Report(False, [Violation("format", "Trajectoire invalide : champ manquant %s" % e)])


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
    if isinstance(d, dict) and "trajectory" in d:  # fichier produit par nao-motion generate --out
        d = d["trajectory"]
    rep = g.check_trajectory_dict(d)
    out = json.dumps(rep.to_dict(), indent=1, ensure_ascii=False)
    if not isinstance(out, str):  # Python 2 : unicode -> octets
        out = out.encode("utf-8")
    print(out)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
