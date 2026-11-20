"""Renames emotion ids in the stored takes (metadata only, samples untouched).

Used once to move the takes recorded with the former French ids to the English
ids of configs/emotions.yaml:

    joie -> joy, tristesse -> sadness, colere -> anger, peur -> fear, calme -> calm
    (surprise is unchanged)

Rewritten fields of <data_dir>/takes/<id>/meta.json and of the trashed takes
(<data_dir>/trash/takes/<id>/meta.json):
- `emotion`;
- `arm_generation.emotion` of old head/hands takes recorded on a generated arm
  gesture (a name or a mix {name: weight}).

Each file is written to a temporary file then renamed: a take is never left
half-written. Running it again changes nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

FRENCH_TO_ENGLISH: Dict[str, str] = {
    "joie": "joy",
    "tristesse": "sadness",
    "colere": "anger",
    "peur": "fear",
    "surprise": "surprise",
    "calme": "calm",
}


def _rename(value, mapping):
    """(new value, changed?) for a name or a mix {name: weight}."""
    if isinstance(value, str):
        new = mapping.get(value, value)
        return new, new != value
    if isinstance(value, dict):
        out, changed = {}, False
        for k, w in value.items():
            nk = mapping.get(k, k)
            changed |= nk != k
            out[nk] = out[nk] + w if nk in out else w
        return out, changed
    return value, False


def rename_meta(meta: dict, mapping: Dict[str, str]) -> bool:
    """Renames the emotions of one take's metadata in place; True if something changed."""
    changed = False
    if "emotion" in meta:
        meta["emotion"], c = _rename(meta["emotion"], mapping)
        changed |= c
    gen = meta.get("arm_generation")
    if isinstance(gen, dict) and "emotion" in gen:
        gen["emotion"], c = _rename(gen["emotion"], mapping)
        changed |= c
    return changed


def _write_json(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def rename_emotions(data_dir, mapping: Optional[Dict[str, str]] = None, known=None,
                    dry_run: bool = False) -> dict:
    """Renames the emotions of every take (trash included).

    `known`: the emotion ids of the current configuration; takes whose emotion is
    still unknown after renaming are listed in `unknown` (not changed further).
    Returns the counts: takes seen, changed, per renaming, unknown."""
    mapping = dict(FRENCH_TO_ENGLISH if mapping is None else mapping)
    root = Path(data_dir)
    dirs = [root / "takes", root / "trash" / "takes"]
    seen, changed, unreadable = 0, [], []
    per_name: Dict[str, int] = {}
    unknown: Dict[str, list] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for meta_path in sorted(d.glob("*/meta.json")):
            seen += 1
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                unreadable.append(str(meta_path))
                continue
            before = meta.get("emotion")
            if rename_meta(meta, mapping):
                changed.append(meta_path.parent.name)
                if isinstance(before, str) and mapping.get(before, before) != before:
                    key = f"{before} -> {mapping[before]}"
                    per_name[key] = per_name.get(key, 0) + 1
                if not dry_run:
                    _write_json(meta_path, meta)
            emo = meta.get("emotion")
            if known is not None and isinstance(emo, str) and emo not in known:
                unknown.setdefault(emo, []).append(meta_path.parent.name)
    return {"data_dir": str(root), "dry_run": dry_run, "takes_seen": seen, "takes_changed": len(changed),
            "renamed": per_name, "unknown": unknown, "unreadable": unreadable}
