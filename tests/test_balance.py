"""Tirage aléatoire équilibré (nao_motion/data/balance.py)."""
import math
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from nao_motion.data.balance import NONE, Grid, balanced_draw, cell_weights, count_takes, coverage


def meta(d, kind="arms", status="accepted"):
    """Métadonnées minimales d'une prise à partir d'un tirage."""
    ev = d["event"]
    if ev["present"]:
        ev = {"present": True, "azimuth_rad": math.radians(ev["azimuth"]),
              "elevation_rad": math.radians(ev["elevation"]), "distance_m": ev["distance_m"]}
    return SimpleNamespace(kind=kind, status=status, emotion=d["emotion"], intensity=d["intensity"], event=ev)


def simulate(cfg, n, strength, allow_none=False, seed=0):
    cfg.robot.setdefault("events", {}).setdefault("balance", {})
    old = cfg.robot["events"]["balance"].get("strength")
    cfg.robot["events"]["balance"]["strength"] = strength
    try:
        rng = np.random.default_rng(seed)
        metas = []
        for _ in range(n):
            metas.append(meta(balanced_draw(cfg, metas, "arms", allow_none=allow_none, rng=rng)))
        return metas
    finally:
        if old is None:
            del cfg.robot["events"]["balance"]["strength"]
        else:
            cfg.robot["events"]["balance"]["strength"] = old


def spread(metas, cfg):
    g = Grid.from_config(cfg)
    emo = Counter(m.emotion for m in metas)
    cells = Counter((m.emotion, g.int_bin(m.intensity), g.event_zones(m.event)[0][0]) for m in metas)
    return max(emo.values()) - min(emo[e] for e in g.emotions), cells


def test_draw_stays_in_configured_ranges(cfg):
    ev = cfg.robot["events"]
    rng = np.random.default_rng(1)
    for _ in range(300):
        d = balanced_draw(cfg, [], "arms", allow_none=True, rng=rng)
        assert d["emotion"] in cfg.emotions and 0.1 - 1e-9 <= d["intensity"] <= 1.0
        assert abs(d["intensity"] * 20 - round(d["intensity"] * 20)) < 1e-9  # pas de 0,05
        e = d["event"]
        if e["present"]:
            assert ev["azimuth_deg"][0] <= e["azimuth"] <= ev["azimuth_deg"][1]
            assert ev["elevation_deg"][0] <= e["elevation"] <= ev["elevation_deg"][1]
            assert ev["distance_m"][0] <= e["distance_m"] <= ev["distance_m"][1]


def test_empty_cell_is_preferred(cfg):
    """Toutes les cases remplies sauf une : elle sort presque toujours."""
    g = Grid.from_config(cfg)
    rng = np.random.default_rng(2)
    metas = []
    target = ("peur", 2, (0, 1, 0))
    for e in g.emotions:
        for i in range(g.n_int):
            for z in g.zones(False):
                if (e, i, z) == target:
                    continue
                for _ in range(3):
                    d = balanced_draw(cfg, [], "arms", rng=rng)
                    while (d["emotion"], g.int_bin(d["intensity"]), g.event_zones(meta(d).event)[0][0]) != (e, i, z):
                        d = balanced_draw(cfg, [], "arms", rng=rng)
                    metas.append(meta(d))
    hits = 0
    for _ in range(40):
        d = balanced_draw(cfg, metas, "arms", rng=rng)
        hits += (d["emotion"], g.int_bin(d["intensity"]), g.event_zones(meta(d).event)[0][0]) == target
        assert d["balance"]["empty_cells"] == 1
    assert hits == 40  # seule case en retard : elle sort à chaque fois


def test_balanced_beats_uniform(cfg):
    """À nombre de prises égal, la répartition est nettement plus régulière qu'au hasard pur."""
    for n in (36, 216):
        bal, uni = [], []
        for seed in range(5):
            sb, cb = spread(simulate(cfg, n, 2.0, seed=seed), cfg)
            su, cu = spread(simulate(cfg, n, 0.0, seed=seed), cfg)
            bal.append((sb, max(cb.values()))); uni.append((su, max(cu.values())))
        assert max(b[0] for b in bal) <= 2          # émotions à ±1-2 prises près
        assert np.mean([b[0] for b in bal]) < np.mean([u[0] for u in uni])
        assert np.mean([b[1] for b in bal]) <= np.mean([u[1] for u in uni])
    metas = simulate(cfg, 216, 2.0)
    _, cells = spread(metas, cfg)
    assert len(cells) >= 210 and max(cells.values()) <= 2  # 216 cases : (presque) toutes une fois


def test_none_share_and_other_kinds_ignored(cfg):
    metas = simulate(cfg, 200, 2.0, allow_none=True, seed=3)
    share = sum(not m.event["present"] for m in metas) / len(metas)
    assert 0.15 <= share <= 0.25  # visé : 20 %
    g = Grid.from_config(cfg)
    # les prises rejetées et celles d'un autre passage ne comptent pas
    other = [SimpleNamespace(**{**vars(m), "kind": "head"}) for m in metas[:50]]
    rej = [SimpleNamespace(**{**vars(m), "status": "rejected"}) for m in metas[:50]]
    assert count_takes(g, other + rej, "arms")[4] == 0
    cov = coverage(cfg, metas, "arms")
    assert cov["takes"] == 200 and sum(cov["emotions"].values()) == 200


def test_none_share_one_draws_only_without_event(cfg):
    cfg.robot["events"].setdefault("balance", {})["none_share"] = 1.0
    try:
        metas = simulate(cfg, 60, 2.0, allow_none=True, seed=4)
        assert not any(m.event["present"] for m in metas)
        emo = Counter(m.emotion for m in metas)
        assert max(emo.values()) - min(emo.values()) <= 2  # émotions toujours équilibrées
        # case décochée : on retombe sur les zones avec événement
        assert all(m.event["present"] for m in simulate(cfg, 20, 2.0, allow_none=False, seed=5))
    finally:
        del cfg.robot["events"]["balance"]["none_share"]


def test_unknown_distance_split(cfg):
    g = Grid.from_config(cfg)
    zs = g.event_zones({"present": True, "azimuth_rad": 0.0, "elevation_rad": 0.0, "distance_m": None})
    assert len(zs) == g.n_dist and abs(sum(w for _, w in zs) - 1) < 1e-9
    assert g.event_zones({"present": False}) == [(NONE, 1.0)]


def test_strength_zero_is_uniform(cfg):
    cfg.robot["events"].setdefault("balance", {})["strength"] = 0.0
    try:
        g = Grid.from_config(cfg)
        metas = simulate(cfg, 30, 2.0)
        _, _, p = cell_weights(g, count_takes(g, metas, "arms"), False)
        assert np.allclose(p, 1 / len(p))
    finally:
        del cfg.robot["events"]["balance"]["strength"]


def test_bad_balance_config_rejected(cfg):
    bad = SimpleNamespace(emotions=cfg.emotions, robot={"events": {"balance": {"intensity_bins": 0}}})
    with pytest.raises(ValueError):
        Grid.from_config(bad)


def test_api_draw_endpoint(cfg, tmp_path):
    """La route /api/draw compte les prises du passage demandé et ne commande rien."""
    import sys
    import types
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch absent : la route est couverte par les tests ci-dessus")
    from nao_motion.ui.server import Api
    from tests.conftest import make_wb
    wb = make_wb(tmp_path, cfg)
    api = Api(wb)
    n = len(wb.adapter.command_log)
    d = api.post_draw({"kind": "arms", "allow_none": True})
    assert d["emotion"] in cfg.emotions and d["balance"]["takes"] == 0
    assert api.post_draw({"kind": "gen"})["balance"]["kind"] is None
    with pytest.raises(ValueError):
        api.post_draw({"kind": "jambes"})
    assert len(wb.adapter.command_log) == n


def test_progress_counts_only_accepted_real_takes(cfg):
    from nao_motion.data.progress import milestones, progress
    metas = simulate(cfg, 70, 2.0)
    for m in metas:
        m.synthetic = False
    metas[0].status = "pending"
    metas[1].status = "rejected"
    sim = [SimpleNamespace(**{**vars(m), "synthetic": True}) for m in metas[:10]]
    head = [SimpleNamespace(**{**vars(m), "kind": "head"}) for m in metas[2:7]]
    pr = progress(cfg, metas + sim + head)["kinds"]
    a = pr["arms"]
    assert (a["accepted"], a["pending"], a["rejected"], a["synthetic_accepted"]) == (68, 1, 1, 8)
    assert a["reached"]["takes"] == 60 and a["next"] == {**milestones(cfg)["arms"][1], "remaining": 150 - 68}
    assert a["goal"] == milestones(cfg)["arms"][-1]["takes"]
    assert a["coverage"]["event_cells"] == 216 and 60 <= a["coverage"]["event_covered"] <= 68
    assert sum(a["coverage"]["emotions"].values()) == 68
    assert pr["head"]["accepted"] == 5 and pr["head"]["next"]["remaining"] == 55
    assert pr["hands"]["accepted"] == 0 and pr["hands"]["reached"] is None


def test_progress_milestones_config(cfg):
    from nao_motion.data.progress import milestones
    old = cfg.robot.get("progress")
    try:
        cfg.robot["progress"] = {"hands": [80, {"takes": 20, "label": "essai"}]}
        ms = milestones(cfg)
        assert [m["takes"] for m in ms["hands"]] == [20, 80] and ms["hands"][0]["label"] == "essai"
        assert ms["arms"][0]["takes"] == 60  # groupe absent : valeurs par défaut
        for bad in ([], [0], [20, 20], ["x"]):
            cfg.robot["progress"] = {"head": bad}
            with pytest.raises(ValueError):
                milestones(cfg)
    finally:
        if old is None:
            cfg.robot.pop("progress", None)
        else:
            cfg.robot["progress"] = old
