"""Guardian : validation indépendante des trajectoires avant toute exécution.

Le guardian ne modifie jamais une trajectoire : il l'accepte ou la refuse en
listant les violations. Il vérifie le TRAJET COMPLET (échantillonné finement),
pas seulement ses extrémités.

Ce qu'il ne garantit pas :
  - l'absence de collision au-delà des contraintes configurées (des limites
    min/max indépendantes et quelques contraintes linéaires couplées ne
    décrivent pas toute la géométrie du robot) ;
  - la stabilité électrique ou logicielle du robot : il ne corrige pas des
    redémarrages dont la cause n'est pas établie.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..adapters.base import LEG_PREFIXES
from ..config import Config
from ..trajectory import Trajectory, TrajectoryError


@dataclass
class Violation:
    code: str
    message: str
    joint: Optional[str] = None
    time_s: Optional[float] = None
    value: Optional[float] = None
    limit: Optional[float] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Report:
    ok: bool
    violations: List[Violation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self):
        return {"ok": self.ok, "violations": [v.to_dict() for v in self.violations[:50]],
                "n_violations": len(self.violations), "stats": self.stats}

    def summary(self) -> str:
        if self.ok:
            return "trajectoire acceptée"
        codes = sorted({v.code for v in self.violations})
        return f"trajectoire refusée ({len(self.violations)} violations : {', '.join(codes)})"


class GuardianRejection(RuntimeError):
    def __init__(self, report: Report):
        super().__init__(report.summary())
        self.report = report


class Guardian:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.g = cfg.guardian

    def validate_raw(self, data: dict, **kw) -> Report:
        """Valide une trajectoire reçue sous forme de dictionnaire (ex. réseau)."""
        try:
            if not isinstance(data, dict):
                raise TrajectoryError("dictionnaire attendu")
            traj = Trajectory.from_dict(data)
        except (TrajectoryError, KeyError, TypeError, ValueError) as e:
            return Report(False, [Violation("format", f"Trajectoire invalide : {e}")])
        return self.validate(traj, **kw)

    def validate(self, traj: Trajectory, *, expected_joints: Optional[Sequence[str]] = None,
                 start_state: Optional[dict] = None, check_duration: bool = True,
                 context_pose: Optional[dict] = None) -> Report:
        """
        expected_joints : ensemble exact attendu (trajectoire incompatible sinon).
        start_state     : {joint: rad} mesuré ; le premier point doit en être proche.
        context_pose    : {joint: rad} pour les articulations non commandées, utilisé
                          dans les contraintes couplées (ex. tête maintenue pendant
                          un mouvement des bras).
        """
        v: List[Violation] = []
        try:
            from ..trajectory import check_samples
            check_samples(traj.t, traj.q, traj.joints)
        except TrajectoryError as e:
            return Report(False, [Violation("format", str(e))])

        for j in traj.joints:
            if j.startswith(LEG_PREFIXES):
                v.append(Violation("legs", "Les jambes sont hors périmètre", joint=j))
            elif j not in self.cfg.limits:
                v.append(Violation("unknown_joint", "Articulation non configurée", joint=j))
        if expected_joints is not None and set(expected_joints) != set(traj.joints):
            v.append(Violation("incompatible", f"Articulations attendues {sorted(expected_joints)}, "
                                               f"reçues {sorted(traj.joints)}"))
        if v:
            return Report(False, v)

        dur = traj.duration
        if check_duration and not (self.g.min_duration_s <= dur <= self.g.max_duration_s):
            v.append(Violation("duration", "Durée hors bornes", value=dur,
                               limit=self.g.max_duration_s if dur > self.g.max_duration_s else self.g.min_duration_s))

        # trajet complet : échantillons d'origine + grille fine
        n = max(2, int(np.ceil(dur * self.g.check_rate_hz)) + 1)
        tt = np.union1d(traj.t, np.linspace(traj.t[0], traj.t[-1], n))
        qq = traj.sample(tt)
        m = self.g.limit_margin_rad
        for k, j in enumerate(traj.joints):
            lo, hi = self.cfg.limits[j]
            lo, hi = lo + m, hi - m
            bad = np.nonzero((qq[:, k] < lo) | (qq[:, k] > hi))[0]
            if bad.size:
                i = bad[0]
                v.append(Violation("limit", f"Hors limites sur {bad.size} points (marge incluse)", joint=j,
                                   time_s=float(tt[i]), value=float(qq[i, k]),
                                   limit=lo if qq[i, k] < lo else hi))

        # vitesses et accélérations du trajet linéaire par morceaux exécuté
        dt = np.diff(traj.t)
        vel = np.diff(traj.q, axis=0) / dt[:, None]
        max_v = np.abs(vel).max(axis=0) if vel.size else np.zeros(len(traj.joints))
        for k, j in enumerate(traj.joints):
            lim = self.g.velocity_limit(j)
            if max_v[k] > lim:
                i = int(np.argmax(np.abs(vel[:, k])))
                v.append(Violation("velocity", "Vitesse excessive", joint=j, time_s=float(traj.t[i]),
                                   value=float(max_v[k]), limit=lim))
        max_a = np.zeros(len(traj.joints))
        if vel.shape[0] >= 2:
            dtm = (dt[1:] + dt[:-1]) / 2
            acc = np.diff(vel, axis=0) / dtm[:, None]
            max_a = np.abs(acc).max(axis=0)
            for k, j in enumerate(traj.joints):
                if max_a[k] > self.g.max_acceleration_rad_s2:
                    i = int(np.argmax(np.abs(acc[:, k])))
                    v.append(Violation("acceleration", "Accélération excessive", joint=j,
                                       time_s=float(traj.t[i + 1]), value=float(max_a[k]),
                                       limit=self.g.max_acceleration_rad_s2))

        # contraintes couplées sur tout le trajet (y compris tête <-> bras)
        ctx = dict(context_pose or {})
        for c in self.g.coupled_constraints:
            total = np.zeros(tt.shape[0])
            usable = True
            for j, coef in c.terms.items():
                if j in traj.joints:
                    total += coef * qq[:, traj.joints.index(j)]
                elif j in ctx:
                    total += coef * float(ctx[j])
                else:
                    usable = False
                    break
            if not usable:
                continue
            bad = np.nonzero(total > c.bound)[0]
            if bad.size:
                i = bad[0]
                v.append(Violation("coupled", f"Contrainte couplée « {c.name} » violée sur {bad.size} points",
                                   time_s=float(tt[i]), value=float(total[i]), limit=c.bound))

        if start_state is not None:
            for k, j in enumerate(traj.joints):
                if j in start_state:
                    d = abs(float(traj.q[0, k]) - float(start_state[j]))
                    if d > self.g.start_tolerance_rad:
                        v.append(Violation("start_mismatch", "Premier point éloigné de l'état mesuré", joint=j,
                                           value=d, limit=self.g.start_tolerance_rad))

        stats = {"duration_s": dur, "n_checked_points": int(tt.shape[0]),
                 "max_velocity": dict(zip(traj.joints, np.round(max_v, 4).tolist())),
                 "max_acceleration": dict(zip(traj.joints, np.round(max_a, 4).tolist()))}
        if any(x.code in ("velocity", "acceleration") for x in v):
            # facteur de ralentissement qui suffirait (information seulement, pas appliqué)
            sv = max(max_v[k] / self.g.velocity_limit(j) for k, j in enumerate(traj.joints))
            sa = float(np.sqrt(max(max_a) / self.g.max_acceleration_rad_s2)) if max_a.size else 0
            stats["suggested_time_scale"] = round(float(max(sv, sa, 1.0)), 3)
        return Report(not v, v, stats)

    def require(self, traj: Trajectory, **kw) -> Report:
        rep = self.validate(traj, **kw)
        if not rep.ok:
            raise GuardianRejection(rep)
        return rep
