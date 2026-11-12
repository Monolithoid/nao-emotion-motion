"""Tirage aléatoire ÉQUILIBRÉ des consignes de prise.

But : que le bouton « Tirage aléatoire » propose en priorité ce qui manque dans
le jeu de données, pour que les prises se répartissent uniformément, tout en
restant aléatoire (deux tirages de suite ne donnent pas la même chose, aucune
consigne n'est jamais impossible).

Méthode (stratification + « softmin » sur le déficit) :

1. L'espace des consignes est découpé en CASES : émotion × tranche d'intensité ×
   zone d'événement. Une zone = secteur d'azimut × tranche d'élévation × tranche
   de distance (distance en échelle log : autant de proches que de lointains),
   plus la zone « sans événement » si elle est autorisée. Le nombre de tranches
   se règle dans robot.yaml (events.balance).
2. Chaque case a une PART ÉQUITABLE : uniforme sur les émotions et les
   intensités ; pour les zones, `none_share` (20 % par défaut) pour « sans
   événement » et le reste réparti également entre les zones avec événement.
3. On compte les prises déjà enregistrées du même passage (bras, tête ou mains),
   hors prises rejetées (elles sont à refaire). Une prise dont la distance est
   inconnue compte pour moitié dans chaque tranche de distance, etc.
4. Score d'une case = 4 × son excédent (prises présentes − part équitable ×
   total) + l'excédent de son émotion + celui de sa tranche d'intensité + celui
   de sa zone. Tous les termes sont en « nombre de prises ». Le poids 4 de la
   case (`cell_weight`) fait passer la case d'abord : comme un paquet de cartes
   distribué, chaque case reçoit une prise avant qu'une autre en reçoive une
   deuxième. Entre cases également remplies, les marges départagent : une
   émotion en retard de 2 prises est nettement favorisée, si bien que les
   répartitions globales (par émotion, intensité, zone) restent équilibrées à
   1-2 prises près à tout moment, même avant que toutes les cases soient vues.
5. Probabilité d'une case ∝ exp(−force × (score − score minimal)). Avec la
   force par défaut (2), une émotion en retard d'une prise rend ses cases
   e² ≈ 7 fois plus probables ; une case en retard d'une prise l'est des
   milliers de fois plus. Force 0 = tirage uniforme d'avant. Le hasard reste
   entier pour CHOISIR PARMI les cases les moins remplies (souvent des dizaines).
6. Dans la case choisie, valeurs continues tirées uniformément (intensité au
   pas de 0,05, angles au degré, distance log-uniforme au centimètre) : on ne
   rejoue jamais deux fois exactement la même consigne.

Ce module ne commande aucun mouvement : il ne fait que proposer une consigne.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULTS = {
    "intensity": [0.1, 1.0],
    "intensity_bins": 3,
    "azimuth_bins": 3,
    "elevation_bins": 2,
    "distance_bins": 2,
    "none_share": 0.2,
    "cell_weight": 4.0,
    "strength": 2.0,
}
EVENT_DEFAULTS = {"distance_m": [0.3, 3.0], "azimuth_deg": [-90, 90], "elevation_deg": [-30, 45]}
NONE = "none"


def _bin(x: float, lo: float, hi: float, k: int) -> int:
    if k <= 1 or hi <= lo:
        return 0
    return int(min(k - 1, max(0, math.floor((x - lo) / (hi - lo) * k))))


def _edges(lo: float, hi: float, k: int, i: int) -> Tuple[float, float]:
    w = (hi - lo) / k
    return lo + i * w, lo + (i + 1) * w


@dataclass
class Grid:
    emotions: List[str]
    intensity: Tuple[float, float]
    n_int: int
    az: Tuple[float, float]
    el: Tuple[float, float]
    dist: Tuple[float, float]
    n_az: int
    n_el: int
    n_dist: int
    none_share: float
    strength: float
    cell_weight: float = 4.0

    @classmethod
    def from_config(cls, cfg) -> "Grid":
        ev = {**EVENT_DEFAULTS, **((cfg.robot.get("events") or {}) if hasattr(cfg, "robot") else {})}
        b = {**DEFAULTS, **(ev.get("balance") or {})}
        g = cls(list(cfg.emotions), tuple(map(float, b["intensity"])), int(b["intensity_bins"]),
                tuple(map(float, ev["azimuth_deg"])), tuple(map(float, ev["elevation_deg"])),
                tuple(map(float, ev["distance_m"])), int(b["azimuth_bins"]), int(b["elevation_bins"]),
                int(b["distance_bins"]), float(b["none_share"]), float(b["strength"]),
                float(b["cell_weight"]))
        if not (0 <= g.intensity[0] < g.intensity[1] <= 1):
            raise ValueError("events.balance.intensity : plage [a, b] avec 0 <= a < b <= 1 attendue")
        if min(g.n_int, g.n_az, g.n_el, g.n_dist) < 1 or not 0 < g.dist[0] < g.dist[1]:
            raise ValueError("events.balance : nombres de tranches >= 1 et distances positives attendus")
        if not 0 <= g.none_share <= 1 or g.strength < 0 or g.cell_weight < 0:
            raise ValueError("events.balance : none_share dans [0, 1], strength et cell_weight >= 0 attendus")
        return g

    # -- zones ----------------------------------------------------------------
    def zones(self, allow_none: bool) -> List[object]:
        z = [(a, e, d) for a in range(self.n_az) for e in range(self.n_el) for d in range(self.n_dist)]
        if allow_none and self.none_share >= 1:
            return [NONE]  # série « sans événement » uniquement
        return ([NONE] if allow_none and self.none_share > 0 else []) + z

    def zone_share(self, z, allow_none: bool) -> float:
        n_present = self.n_az * self.n_el * self.n_dist
        p_none = self.none_share if allow_none and self.none_share > 0 else 0.0
        return p_none if z == NONE else (1.0 - p_none) / n_present

    def int_bin(self, x: float) -> int:
        return _bin(x, *self.intensity, self.n_int)

    def dist_bin(self, d: float) -> int:
        return _bin(math.log(d), math.log(self.dist[0]), math.log(self.dist[1]), self.n_dist)

    def event_zones(self, ev: dict) -> List[Tuple[object, float]]:
        """Zone(s) d'un événement enregistré, avec poids (distance inconnue : réparti)."""
        if not ev or not ev.get("present"):
            return [(NONE, 1.0)]
        a = _bin(math.degrees(ev.get("azimuth_rad") or 0.0), *self.az, self.n_az)
        e = _bin(math.degrees(ev.get("elevation_rad") or 0.0), *self.el, self.n_el)
        d = ev.get("distance_m")
        if d is None or not d > 0:
            return [((a, e, k), 1.0 / self.n_dist) for k in range(self.n_dist)]
        return [((a, e, self.dist_bin(float(d))), 1.0)]

    # -- libellés -------------------------------------------------------------
    def int_label(self, i: int) -> str:
        if self.n_int == 3:
            return ("faible", "moyenne", "forte")[i]
        lo, hi = _edges(*self.intensity, self.n_int, i)
        return f"{lo:.2f}–{hi:.2f}"

    def zone_label(self, z) -> str:
        if z == NONE:
            return "sans événement"
        a, e, d = z
        alo, ahi = _edges(*self.az, self.n_az, a)
        elo, ehi = _edges(*self.el, self.n_el, e)
        ca, ce = (alo + ahi) / 2, (elo + ehi) / 2
        wa = "devant" if abs(ca) < 1e-9 or (alo < 0 < ahi) else ("gauche" if ca > 0 else "droite")
        we = "haut" if ce > 0 and elo >= 0 else ("bas" if ehi <= 0 else ("plutôt haut" if ce > 0 else "plutôt bas"))
        lo, hi = math.log(self.dist[0]), math.log(self.dist[1])
        dlo, dhi = (math.exp(x) for x in _edges(lo, hi, self.n_dist, d))
        wd = ("proche", "loin")[d] if self.n_dist == 2 else f"{dlo:.2f}–{dhi:.2f} m"
        return f"{wa} ({alo:.0f}…{ahi:.0f}°), {we} ({elo:.0f}…{ehi:.0f}°), {wd}"


def count_takes(grid: Grid, metas: Iterable, kind: Optional[str], statuses: Optional[Sequence[str]] = None,
                include_synthetic: bool = True):
    """Comptes (case, émotion, intensité, zone) des prises du passage `kind`.

    Par défaut : toutes les prises hors rejetées (tirage). `statuses` restreint
    aux statuts donnés (ex. ("accepted",) pour la progression) ;
    `include_synthetic=False` écarte les prises simulées.
    """
    joint, emo, inten, zone = Counter(), Counter(), Counter(), Counter()
    total = 0.0
    for m in metas:
        if kind is None or m.kind != kind or m.emotion not in grid.emotions:
            continue
        if (m.status not in statuses) if statuses is not None else m.status == "rejected":
            continue
        if not include_synthetic and getattr(m, "synthetic", False):
            continue
        i = grid.int_bin(float(m.intensity))
        emo[m.emotion] += 1
        inten[i] += 1
        total += 1
        for z, w in grid.event_zones(m.event):
            zone[z] += w
            joint[(m.emotion, i, z)] += w
    return joint, emo, inten, zone, total


def cell_weights(grid: Grid, counts, allow_none: bool):
    """Cases candidates, scores (excédent en prises) et probabilités."""
    joint, emo, inten, zone, n = counts
    zs = grid.zones(allow_none)
    pe, pi = 1.0 / len(grid.emotions), 1.0 / grid.n_int
    cells, scores = [], []
    for e in grid.emotions:
        for i in range(grid.n_int):
            for z in zs:
                pz = grid.zone_share(z, allow_none)
                s = grid.cell_weight * (joint[(e, i, z)] - pe * pi * pz * n) + (emo[e] - pe * n) + (inten[i] - pi * n) + (zone[z] - pz * n)
                cells.append((e, i, z))
                scores.append(s)
    scores = np.asarray(scores, float)
    logits = -grid.strength * (scores - scores.min())
    p = np.exp(logits - logits.max())
    return cells, scores, p / p.sum()


def _draw_in_cell(grid: Grid, cell, rng: np.random.Generator) -> dict:
    e, i, z = cell
    lo, hi = _edges(*grid.intensity, grid.n_int, i)
    steps = [round(x / 20, 2) for x in range(0, 21)]
    pts = [x for x in steps if grid.intensity[0] - 1e-9 <= x <= grid.intensity[1] + 1e-9 and grid.int_bin(x) == i]
    intensity = float(rng.choice(pts)) if pts else round(float(rng.uniform(lo, hi)), 3)
    if z == NONE:
        return {"emotion": e, "intensity": intensity, "event": {"present": False, "source": "tirage"}}
    a, el, d = z
    ev = None
    for _ in range(50):  # arrondis : on garde une valeur qui retombe bien dans la case
        az = round(float(rng.uniform(*_edges(*grid.az, grid.n_az, a))))
        elv = round(float(rng.uniform(*_edges(*grid.el, grid.n_el, el))))
        llo, lhi = _edges(math.log(grid.dist[0]), math.log(grid.dist[1]), grid.n_dist, d)
        dist = round(math.exp(float(rng.uniform(llo, lhi))), 2)
        ev = {"present": True, "unit": "deg", "frame": "torso", "source": "tirage",
              "azimuth": az, "elevation": elv, "distance_m": dist}
        if (_bin(az, *grid.az, grid.n_az), _bin(elv, *grid.el, grid.n_el), grid.dist_bin(dist)) == (a, el, d):
            break
    return {"emotion": e, "intensity": intensity, "event": ev}


def balanced_draw(cfg, metas: Sequence, kind: Optional[str], allow_none: bool = False,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """Propose une consigne (émotion, intensité, événement) qui comble les manques.

    kind : "arms" | "head" | "hands" (comptes de ce passage) ; None = aucun
    compte (onglet Génération : tirage uniforme sur les cases).
    """
    grid = Grid.from_config(cfg)
    rng = rng or np.random.default_rng()
    counts = count_takes(grid, metas, kind)
    cells, scores, p = cell_weights(grid, counts, allow_none)
    k = int(rng.choice(len(cells), p=p))
    cell = cells[k]
    out = _draw_in_cell(grid, cell, rng)
    joint, emo, inten, zone, n = counts
    e, i, z = cell
    pz = grid.zone_share(z, allow_none)
    fair = n / len(grid.emotions) / grid.n_int * pz
    out["balance"] = {
        "kind": kind,
        "takes": n,
        "cell": {"emotion": e, "intensity": grid.int_label(i), "zone": grid.zone_label(z)},
        "cell_takes": round(joint[cell], 2),
        "cell_fair_share": round(fair, 2),
        "emotion_takes": emo[e],
        "probability": float(p[k]),
        "cells": len(cells),
        "empty_cells": int(sum(1 for c in cells if joint[c] < 1e-9)),
        "least_filled_probability": float(p[scores <= scores.min() + 1e-9].sum()),
        "strength": grid.strength,
    }
    return out


def coverage(cfg, metas: Sequence, kind: str, allow_none: bool = True) -> dict:
    """Répartition actuelle (pour l'affichage) : par émotion, intensité, zone."""
    grid = Grid.from_config(cfg)
    joint, emo, inten, zone, n = count_takes(grid, metas, kind)
    zs = grid.zones(allow_none)
    return {
        "takes": n,
        "emotions": {e: emo[e] for e in grid.emotions},
        "intensities": {grid.int_label(i): inten[i] for i in range(grid.n_int)},
        "zones": {grid.zone_label(z): round(zone[z], 2) for z in zs},
        "cells": len(grid.emotions) * grid.n_int * len(zs),
        "empty_cells": sum(1 for e in grid.emotions for i in range(grid.n_int) for z in zs if joint[(e, i, z)] < 1e-9),
    }
