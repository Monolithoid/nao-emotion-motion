"""Trash for takes and training runs.

Deleting moves the folder into a trash next to it, never out of the disk at once:

    <data_dir>/trash/takes/<take_id>/   takes moved out of <data_dir>/takes/
    <runs_dir>/trash/<run>/             runs moved out of <runs_dir>/

Everything that reads takes (training, progress, coverage, `export-dataset`) only
looks into <data_dir>/takes/, and the run list only into <runs_dir>/*/metrics.json,
so a trashed item is ignored everywhere at once. It can be restored as it was, until
the trash is emptied (the only irreversible step).

A run that is the active model of its group (runs/latest_<group>.txt) can be trashed:
the newest remaining run of the same group becomes the active model (or the group has
no model any more), and the caller is told so.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TRASH = "trash"
STAMP = ".trashed.json"
GROUPS = ("arms", "head", "hands")


class TrashError(ValueError):
    pass


def _safe_name(name: str) -> str:
    name = str(name)
    if not name or "/" in name or "\\" in name or name.startswith(".") or name == TRASH:
        raise TrashError(f"Invalid name: {name!r}")
    return name


def _size(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def _move(src: Path, dst_root: Path, info: dict) -> Path:
    dst = dst_root / src.name
    if dst.exists():  # same name already in the trash (restored then trashed again): keep the newest
        shutil.rmtree(dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    (dst / STAMP).write_text(json.dumps({"trashed_at": _dt.datetime.now().isoformat(timespec="seconds"), **info},
                                        ensure_ascii=False), encoding="utf-8")
    return dst


def _restore(src: Path, dst_root: Path) -> Path:
    dst = dst_root / src.name
    if dst.exists():
        raise TrashError(f"{src.name} already exists, not restored")
    (src / STAMP).unlink(missing_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def _stamp(d: Path) -> dict:
    try:
        return json.loads((d / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# -- takes ----------------------------------------------------------------------------
def takes_trash_dir(data_dir) -> Path:
    return Path(data_dir) / TRASH / "takes"


def _read_meta(d: Path) -> Optional[dict]:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def trash_takes(data_dir, take_ids: Iterable[str]) -> dict:
    """Move takes to the trash. Refuses (nothing moved) if an old head/hands take recorded on a
    replayed arm take would lose that arm take: trash them together."""
    root = Path(data_dir) / "takes"
    ids = list(dict.fromkeys(_safe_name(t) for t in take_ids))
    if not ids:
        raise TrashError("No take selected")
    missing = [t for t in ids if not (root / t / "meta.json").exists()]
    if missing:
        raise TrashError(f"Unknown take(s): {', '.join(missing)}")
    chosen = set(ids)
    blocked = {}
    for p in root.glob("*/meta.json"):
        d = _read_meta(p.parent) or {}
        arm = d.get("arm_take_id")
        if arm in chosen and d.get("take_id") not in chosen and (d.get("arm_source") in (None, "take")):
            blocked.setdefault(arm, []).append(d.get("take_id"))
    if blocked:
        raise TrashError("These arm takes were replayed under other takes, select those too: " +
                         "; ".join(f"{a} <- {', '.join(sorted(b))}" for a, b in sorted(blocked.items())))
    trash = takes_trash_dir(data_dir)
    size = 0
    for t in ids:
        size += _size(root / t)
        _move(root / t, trash, {"kind": "take"})
    return {"trashed": ids, "bytes": size}


def restore_takes(data_dir, take_ids: Iterable[str]) -> dict:
    """Put takes back into data/takes/ (arm takes first, so that old head/hands takes find them)."""
    root, trash = Path(data_dir) / "takes", takes_trash_dir(data_dir)
    ids = list(dict.fromkeys(_safe_name(t) for t in take_ids))
    missing = [t for t in ids if not (trash / t).is_dir()]
    if missing:
        raise TrashError(f"Not in the trash: {', '.join(missing)}")
    metas = {t: _read_meta(trash / t) or {} for t in ids}
    for t, d in metas.items():
        arm = d.get("arm_take_id")
        if arm and d.get("arm_source") in (None, "take") and not (root / arm).exists() and arm not in metas:
            raise TrashError(f"{t} was recorded on arm take {arm}, which is in the trash: restore it too")
    taken = [t for t in ids if (root / t).exists()]
    if taken:
        raise TrashError(f"Already in the takes: {', '.join(taken)}")
    root.mkdir(parents=True, exist_ok=True)
    for t in sorted(ids, key=lambda t: metas[t].get("kind") != "arms"):
        _restore(trash / t, root)
    return {"restored": ids}


def list_trashed_takes(data_dir) -> List[dict]:
    out = []
    trash = takes_trash_dir(data_dir)
    for d in sorted(trash.glob("*")) if trash.exists() else []:
        if not d.is_dir():
            continue
        m = _read_meta(d) or {}
        out.append({"take_id": d.name, "kind": m.get("kind"), "emotion": m.get("emotion"),
                    "intensity": m.get("intensity"), "status": m.get("status"), "synthetic": bool(m.get("synthetic")),
                    "tags": (m.get("annotations") or {}).get("tags", []), "duration_s": m.get("duration_s"),
                    "bytes": _size(d), "trashed_at": _stamp(d).get("trashed_at")})
    out.sort(key=lambda r: r.get("trashed_at") or "", reverse=True)
    return out


# -- training runs -------------------------------------------------------------------------
def runs_trash_dir(runs_dir) -> Path:
    return Path(runs_dir) / TRASH


def active_runs(runs_dir) -> Dict[str, str]:
    """{group: run name} read from runs/latest_<group>.txt (same rule as training.latest())."""
    out = {}
    for g in GROUPS:
        p = Path(runs_dir) / f"latest_{g}.txt"
        if not p.exists():
            continue
        name = Path(p.read_text().strip())
        out[g] = name.parent.name if name.name == "model.pt" else name.name
    return out


def _newest_run(runs_dir: Path, group: str) -> Optional[str]:
    cands = [p.parent.name for p in runs_dir.glob(f"{group}-*/model.pt")]
    return max(cands, key=lambda n: n.partition("-")[2]) if cands else None


def trash_runs(runs_dir, runs: Iterable[str]) -> dict:
    """Move runs to the trash. `active_changed`: {group: new active run or None} for every group
    whose active model was trashed."""
    runs_dir = Path(runs_dir)
    names = list(dict.fromkeys(_safe_name(r) for r in runs))
    if not names:
        raise TrashError("No training selected")
    missing = [r for r in names if not (runs_dir / r).is_dir()]
    if missing:
        raise TrashError(f"Unknown training(s): {', '.join(missing)}")
    active = active_runs(runs_dir)
    size = 0
    for r in names:
        size += _size(runs_dir / r)
        _move(runs_dir / r, runs_trash_dir(runs_dir), {"kind": "run", "was_active": r in active.values()})
    changed = {}
    for g, run in active.items():
        if run in names:
            new = _newest_run(runs_dir, g)
            p = runs_dir / f"latest_{g}.txt"
            if new:
                p.write_text(new)
            else:
                p.unlink(missing_ok=True)
            changed[g] = new
    return {"trashed": names, "bytes": size, "active_changed": changed}


def restore_runs(runs_dir, runs: Iterable[str]) -> dict:
    """Put runs back. A group left without an active model gets its newest restored run back
    as active model; otherwise the active model is not changed."""
    runs_dir, trash = Path(runs_dir), runs_trash_dir(runs_dir)
    names = list(dict.fromkeys(_safe_name(r) for r in runs))
    missing = [r for r in names if not (trash / r).is_dir()]
    if missing:
        raise TrashError(f"Not in the trash: {', '.join(missing)}")
    taken = [r for r in names if (runs_dir / r).exists()]
    if taken:
        raise TrashError(f"Already in the trainings: {', '.join(taken)}")
    for r in names:
        _restore(trash / r, runs_dir)
    active = active_runs(runs_dir)
    changed = {}
    for g in GROUPS:
        if g not in active or not (runs_dir / active[g] / "model.pt").exists():
            mine = [r for r in names if r.startswith(g + "-") and (runs_dir / r / "model.pt").exists()]
            if mine:
                new = max(mine, key=lambda n: n.partition("-")[2])
                (runs_dir / f"latest_{g}.txt").write_text(new)
                changed[g] = new
    return {"restored": names, "active_changed": changed}


def list_trashed_runs(runs_dir) -> List[dict]:
    out = []
    trash = runs_trash_dir(runs_dir)
    for d in sorted(trash.glob("*")) if trash.exists() else []:
        if not d.is_dir():
            continue
        comp, _, stamp = d.name.partition("-")
        st = _stamp(d)
        out.append({"run": d.name, "component": comp, "stamp": stamp, "bytes": _size(d),
                    "trashed_at": st.get("trashed_at"), "was_active": bool(st.get("was_active"))})
    out.sort(key=lambda r: r.get("trashed_at") or "", reverse=True)
    return out


# -- emptying (irreversible) --------------------------------------------------------------------
def empty(data_dir=None, runs_dir=None, items: Optional[Iterable[str]] = None) -> dict:
    """Delete for good what is in the trash of takes (data_dir) and/or runs (runs_dir).
    `items`: only these names (take ids / run names), else everything."""
    wanted = None if items is None else {_safe_name(i) for i in items}
    out = {"deleted": [], "bytes": 0}
    for trash in ([takes_trash_dir(data_dir)] if data_dir is not None else []) + \
                 ([runs_trash_dir(runs_dir)] if runs_dir is not None else []):
        if not trash.exists():
            continue
        for d in sorted(trash.glob("*")):
            if d.is_dir() and (wanted is None or d.name in wanted):
                out["bytes"] += _size(d)
                shutil.rmtree(d)
                out["deleted"].append(d.name)
    return out
