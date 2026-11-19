"""Collection progress: ACCEPTED takes per group, milestones, coverage.

Used by the « Prises » (takes) tab: for each group (arms, head, hands, trained
independently), how many accepted takes, which milestone is reached, how many are
missing before the next one, and what share of the balanced-draw cells (see
balance.py) has already received at least one accepted take.

Only ACCEPTED takes count (that is what training uses by default). Simulated takes
(`synthetic`) are counted separately and do not move the bar: they do not replace a
demonstration on the robot.

Milestones: `progress` section of robot.yaml, one list per group, each milestone
being a number or {takes: N, label: "..."}; the last milestone is the goal.

This module commands no movement.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from .balance import NONE, Grid, count_takes

KIND_LABELS = {"arms": "Bras", "head": "Tête", "hands": "Mains"}
DEFAULT_MILESTONES = {
    "arms": [{"takes": 60, "label": "1er entraînement : tester la chaîne"},
             {"takes": 150, "label": "modèle utilisable"},
             {"takes": 200, "label": "objectif"}],
    "head": [{"takes": 60, "label": "1er entraînement : tester la chaîne"},
             {"takes": 150, "label": "modèle utilisable"},
             {"takes": 200, "label": "objectif"}],
    "hands": [{"takes": 60, "label": "1er entraînement : tester la chaîne"},
              {"takes": 100, "label": "objectif"}],
}


def milestones(cfg) -> Dict[str, List[dict]]:
    """Milestones per group (robot.yaml `progress`, otherwise defaults), sorted and checked."""
    conf = (cfg.robot.get("progress") or {}) if hasattr(cfg, "robot") else {}
    out = {}
    for kind, default in DEFAULT_MILESTONES.items():
        raw = conf.get(kind, default)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"progress.{kind} : liste de paliers non vide attendue")
        ms = []
        for x in raw:
            n, label = (x.get("takes"), x.get("label")) if isinstance(x, dict) else (x, None)
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"progress.{kind} : nombre de prises entier >= 1 attendu, reçu {n!r}")
            ms.append({"takes": n, "label": str(label) if label else f"{n} prises"})
        ms.sort(key=lambda m: m["takes"])
        if len({m["takes"] for m in ms}) != len(ms):
            raise ValueError(f"progress.{kind} : paliers en double")
        out[kind] = ms
    return out


def progress(cfg, metas: Sequence) -> dict:
    """Progress per group: statuses, milestones, next milestone, cell coverage."""
    grid = Grid.from_config(cfg)
    ms_all = milestones(cfg)
    metas = list(metas)
    out = {}
    for kind, ms in ms_all.items():
        mine = [m for m in metas if m.kind == kind]
        real = [m for m in mine if not getattr(m, "synthetic", False)]
        st = Counter(m.status for m in real)
        acc = st.get("accepted", 0)
        reached = [m for m in ms if acc >= m["takes"]]
        nxt = next((m for m in ms if acc < m["takes"]), None)
        joint, emo, inten, zone, _ = count_takes(grid, real, kind, statuses=("accepted",))
        ev_zones = grid.zones(False)
        n_ev = len(grid.emotions) * grid.n_int * len(ev_zones)
        ev_cov = sum(1 for e in grid.emotions for i in range(grid.n_int) for z in ev_zones if joint[(e, i, z)] > 1e-9)
        n_none = len(grid.emotions) * grid.n_int
        none_cov = sum(1 for e in grid.emotions for i in range(grid.n_int) if joint[(e, i, NONE)] > 1e-9)
        emotions = {e: emo[e] for e in grid.emotions}
        out[kind] = {
            "label": KIND_LABELS[kind],
            "accepted": acc,
            "pending": st.get("pending", 0),
            "rejected": st.get("rejected", 0),
            "synthetic_accepted": sum(1 for m in mine if getattr(m, "synthetic", False) and m.status == "accepted"),
            "milestones": [{**m, "reached": acc >= m["takes"]} for m in ms],
            "goal": ms[-1]["takes"],
            "reached": reached[-1] if reached else None,
            "next": {**nxt, "remaining": nxt["takes"] - acc} if nxt else None,
            "coverage": {
                "event_cells": n_ev, "event_covered": ev_cov,
                "none_cells": n_none, "none_covered": none_cov,
                "emotions": emotions,
                "intensities": {grid.int_label(i): inten[i] for i in range(grid.n_int)},
            },
        }
    return {"kinds": out, "emotion_labels": {e: cfg.emotion_labels.get(e, e) for e in grid.emotions}}
