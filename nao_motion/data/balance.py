"""BALANCED random draw of take prompts.

Goal: have the « Tirage aléatoire » (random draw) button suggest first what is
missing from the dataset, so that takes are spread evenly, while staying random
(two draws in a row do not give the same thing, no prompt is ever impossible).

Method (stratification + "softmin" on the deficit):

1. The prompt space is split into CELLS: emotion × intensity band × event zone.
   A zone = azimuth sector × elevation band × distance band (distance on a log
   scale: as many near as far), plus the "no event" zone if allowed. The number of
   bands is set in robot.yaml (events.balance).
2. Each cell has a FAIR SHARE: uniform over emotions and intensities; for zones,
   `none_share` (20 % by default) for "no event" and the rest split evenly between
   the zones with an event.
3. The takes already recorded for the same pass (arms, head or hands) are counted,
   excluding rejected takes (they need redoing). A take whose distance is unknown
   counts for half in each distance band, etc.
4. Score of a cell = 4 × its surplus (takes present − fair share × total) + the
   surplus of its emotion + that of its intensity band + that of its zone. All terms
   are in "number of takes". The cell weight of 4 (`cell_weight`) makes the cell come
   first: like dealing a deck of cards, each cell gets one take before another gets a
   second. Between equally filled cells, the margins break the tie: an emotion 2 takes
   behind is clearly favoured, so the overall distributions (by emotion, intensity,
   zone) stay balanced within 1-2 takes at all times, even before every cell has been
   seen.
5. Probability of a cell ∝ exp(−strength × (score − minimum score)). With the default
   strength (2), an emotion one take behind makes its cells e² ≈ 7 times more likely;
   a cell one take behind is thousands of times more likely. Strength 0 = the former
   uniform draw. Randomness stays complete for CHOOSING AMONG the least filled cells
   (often dozens).
6. Within the chosen cell, continuous values are drawn uniformly (intensity in steps
   of 0.05, angles to the degree, log-uniform distance to the centimetre): the exact
   same prompt is never replayed twice.

This module commands no movement: it only suggests a prompt.
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
        if not 0 <= g.none_share < 1 or g.strength < 0 or g.cell_weight < 0:
            raise ValueError("events.balance : none_share dans [0, 1[, strength et cell_weight >= 0 attendus")
        return g

    # -- zones ----------------------------------------------------------------
    def zones(self, allow_none: bool, event: bool = True) -> List[object]:
        if not event:  # event unchecked in the interface: only the "no event" zone
            return [NONE]
        z = [(a, e, d) for a in range(self.n_az) for e in range(self.n_el) for d in range(self.n_dist)]
        return ([NONE] if allow_none and self.none_share > 0 else []) + z

    def zone_share(self, z, allow_none: bool, event: bool = True) -> float:
        if not event:
            return 1.0 if z == NONE else 0.0
        n_present = self.n_az * self.n_el * self.n_dist
        p_none = self.none_share if allow_none and self.none_share > 0 else 0.0
        return p_none if z == NONE else (1.0 - p_none) / n_present

    def int_bin(self, x: float) -> int:
        return _bin(x, *self.intensity, self.n_int)

    def dist_bin(self, d: float) -> int:
        return _bin(math.log(d), math.log(self.dist[0]), math.log(self.dist[1]), self.n_dist)

    def event_zones(self, ev: dict) -> List[Tuple[object, float]]:
        """Zone(s) of a recorded event, with weights (unknown distance: spread)."""
        if not ev or not ev.get("present"):
            return [(NONE, 1.0)]
        a = _bin(math.degrees(ev.get("azimuth_rad") or 0.0), *self.az, self.n_az)
        e = _bin(math.degrees(ev.get("elevation_rad") or 0.0), *self.el, self.n_el)
        d = ev.get("distance_m")
        if d is None or not d > 0:
            return [((a, e, k), 1.0 / self.n_dist) for k in range(self.n_dist)]
        return [((a, e, self.dist_bin(float(d))), 1.0)]

    # -- labels -------------------------------------------------------------
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
    """Counts (cell, emotion, intensity, zone) of the takes of pass `kind`.

    By default: all takes except rejected ones (draw). `statuses` restricts
    to the given statuses (e.g. ("accepted",) for progress);
    `include_synthetic=False` excludes simulated takes.
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


def cell_weights(grid: Grid, counts, allow_none: bool, event: bool = True):
    """Candidate cells, scores (surplus in takes) and probabilities."""
    joint, emo, inten, zone, n = counts
    zs = grid.zones(allow_none, event)
    pe, pi = 1.0 / len(grid.emotions), 1.0 / grid.n_int
    cells, scores = [], []
    for e in grid.emotions:
        for i in range(grid.n_int):
            for z in zs:
                pz = grid.zone_share(z, allow_none, event)
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
    for _ in range(50):  # rounding: keep a value that still falls inside the cell
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
                  event: bool = True, rng: Optional[np.random.Generator] = None) -> dict:
    """Suggests a prompt (emotion, intensity, event) that fills the gaps.

    kind: "arms" | "head" | "hands" (counts for that pass); None = no
    counts (Generation tab: uniform draw over the cells).
    event=False: the event is unchecked in the interface; only emotion and
    intensity are drawn (balanced), the suggested event is absent.
    """
    grid = Grid.from_config(cfg)
    rng = rng or np.random.default_rng()
    counts = count_takes(grid, metas, kind)
    cells, scores, p = cell_weights(grid, counts, allow_none, event)
    k = int(rng.choice(len(cells), p=p))
    cell = cells[k]
    out = _draw_in_cell(grid, cell, rng)
    joint, emo, inten, zone, n = counts
    e, i, z = cell
    pz = grid.zone_share(z, allow_none, event)
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
    """Current distribution (for display): by emotion, intensity, zone."""
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



def coverage_grid(cfg, metas: Sequence, kind: str, emotion: Optional[str] = None,
                  intensity: Optional[float] = None, step: float = 0.1) -> dict:
    """Coverage map of one group (arms, head, hands) for the recording tabs.

    Rows = emotions, columns = intensity in `step` slices over the draw range
    (`events.balance.intensity`), grouped under the draw's intensity bins
    (faible / moyenne / forte). Counts ACCEPTED real takes, like the Takes tab
    progress (simulated takes are reported apart). With `emotion` and
    `intensity`, also returns the cell the next take would fill. Display only.
    """
    grid = Grid.from_config(cfg)
    lo, hi = grid.intensity
    n = max(1, int(round((hi - lo) / step)))
    col = lambda x: _bin(float(x), lo, hi, n)
    metas = [m for m in metas if m.kind == kind]
    real = [m for m in metas if not getattr(m, "synthetic", False)]
    counts = [[0] * n for _ in grid.emotions]
    for m in real:
        if m.status == "accepted" and m.emotion in grid.emotions:
            counts[grid.emotions.index(m.emotion)][col(m.intensity)] += 1
    cols = []
    for j in range(n):
        a, b = _edges(lo, hi, n, j)
        cols.append({"lo": round(a, 3), "hi": round(b, 3), "band": grid.int_bin((a + b) / 2)})
    current = None
    if emotion in grid.emotions and intensity is not None:
        current = [grid.emotions.index(emotion), col(intensity)]
    return {
        "kind": kind,
        "emotions": list(grid.emotions),
        "columns": cols,
        "bands": [grid.int_label(i) for i in range(grid.n_int)],
        "counts": counts,
        "max": max((c for row in counts for c in row), default=0),
        "accepted": sum(map(sum, counts)),
        "simulated": sum(1 for m in metas if getattr(m, "synthetic", False)),
        "current": current,
    }
