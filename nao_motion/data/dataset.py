"""Dataset export and import (takes, calibrations, latest models).

`export_dataset` produces a self-contained folder, ready to be pushed as is to a
separate repository (GitHub, or later Hugging Face, which is also a Git repository):

    <out>/README.md                dataset card (English, for the community)
    <out>/LICENSE                  licence of the DATA (CC BY 4.0 by default; the code stays MIT)
    <out>/dataset.json             manifest: version, filters, counts, sha256 of every file
    <out>/index.csv                one line per take (prompt, event, duration, ...)
    <out>/takes/<id>/meta.json     EXACT copies of the takes (same format as data/takes/)
    <out>/takes/<id>/samples.npz
    <out>/calibration/*.yaml       calibrations referenced by the exported takes
    <out>/configs/                 emotions.yaml and joints.yaml (to read labels and joints)
    <out>/models/<group>/          latest arms / head / hands models (optional)
    <out>/generator/               nao_gesture.py + nao_gesture.json, the same models ready to use
                                   (same files as `nao-motion export-generator`, optional)

By default, only ACCEPTED takes, from the real robot (not `synthetic`) and complete
(without the `incomplete` tag) are exported. Re-exporting into the same folder
updates it: new takes added, changed metadata rewritten, takes no longer selected
removed (only in takes/, and only if the folder is already an export: dataset.json
present). Any .git folder is never touched.

`import_dataset` does the reverse (restoring on another PC): it copies the missing
takes into data/takes/, checks the sha256, adds the missing calibrations to
configs/calibration/ (without activating any) and, if requested, the models into
runs/. Nothing is ever overwritten.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .store import TakeMeta

DATASET_FORMAT = 1
MANIFEST = "dataset.json"
COMPONENTS = ("arms", "head", "hands")
MODEL_FILES = ("model.pt", "model_numpy.npz", "metrics.json", "split.json")
CONFIG_FILES = ("emotions.yaml", "joints.yaml")
GENERATOR_DIR = "generator"
LICENSES = {
    "CC-BY-4.0": ("Creative Commons Attribution 4.0 International",
                  "https://creativecommons.org/licenses/by/4.0/"),
    "CC-BY-SA-4.0": ("Creative Commons Attribution-ShareAlike 4.0 International",
                     "https://creativecommons.org/licenses/by-sa/4.0/"),
    "CC0-1.0": ("Creative Commons Zero v1.0 Universal (public domain dedication)",
                "https://creativecommons.org/publicdomain/zero/1.0/"),
}
# Files whose integrity is checked on import; README.md, LICENSE, index.csv are
# documentation that can be edited by hand (e.g. on GitHub) without blocking the import.
DATA_PREFIXES = ("takes/", "calibration/", "models/", "configs/")
INDEX_COLUMNS = ("take_id", "kind", "emotion", "intensity", "intention", "status", "event_present",
                 "azimuth_rad", "elevation_rad", "distance_m", "duration_s", "segment_start_s",
                 "segment_end_s", "n_samples", "sample_rate_hz", "calibration_version", "created_at", "tags")


class DatasetError(ValueError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_meta(p: Path) -> TakeMeta:
    return TakeMeta.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _copy_if_changed(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def select_takes(data_dir, statuses: Sequence[str] = ("accepted",), include_synthetic: bool = False,
                 include_incomplete: bool = False, kinds: Optional[Iterable[str]] = None) -> List[TakeMeta]:
    """Takes selected for export, sorted by identifier."""
    kinds = set(kinds or COMPONENTS)
    out = []
    for p in sorted((Path(data_dir) / "takes").glob("*/meta.json")):
        m = _read_meta(p)
        if m.kind not in kinds or m.status not in statuses:
            continue
        if m.synthetic and not include_synthetic:
            continue
        if "incomplete" in (m.annotations or {}).get("tags", []) and not include_incomplete:
            continue
        out.append(m)
    return out


def _calibration_files(config_root: Path) -> Dict[str, Path]:
    """{version: file} for the readable calibrations in configs/calibration/."""
    from ..calibration import Calibration
    out = {}
    for f in sorted((config_root / "calibration").glob("*.yaml")):
        try:
            out.setdefault(Calibration.load(f).version, f)
        except Exception:
            continue
    return out


def _latest_runs(runs_dir: Path) -> Dict[str, Path]:
    """Same rule as training.latest(), without importing torch (export possible without PyTorch)."""
    out = {}
    for c in COMPONENTS:
        p = runs_dir / f"latest_{c}.txt"
        if not p.exists():
            continue
        name = Path(p.read_text().strip())
        run_name = name.parent.name if name.name == "model.pt" else name.name
        if (runs_dir / run_name / "model.pt").exists():
            out[c] = runs_dir / run_name
    return out


def _placeholder_version(config_root: Path) -> Optional[str]:
    """Name written in a take recorded while no calibration was active (joints.yaml
    `calibration.version`, e.g. placeholder-0): not a calibration file, nothing to export."""
    try:
        import yaml
        return str(yaml.safe_load((config_root / "joints.yaml").read_text(encoding="utf-8"))["calibration"]["version"])
    except Exception:
        return None


def _export_generator(config_root: Path, runs_dir: Path, out: Path) -> dict:
    """generator/ = `nao-motion export-generator` built from the same runs as models/ (active
    model of each group) and the ACTIVE calibration of `config_root` (guardian limits)."""
    from ..config import load_config
    from ..gesture_export import ExportError, export_generator
    runs = _latest_runs(runs_dir)
    models = {c: (runs[c] / "model_numpy.npz" if c in runs and (runs[c] / "model_numpy.npz").exists() else None)
              for c in COMPONENTS}
    if not any(models.values()):
        raise ExportError("no model_numpy.npz in the active runs (train at least one group)")
    r = export_generator(load_config(config_root), out, models=models)
    return {"files": [f"{GENERATOR_DIR}/{Path(r['library']).name}", f"{GENERATOR_DIR}/{Path(r['data']).name}"],
            "models": r["models"], "missing": r["missing"], "calibration_version": r["calibration_version"],
            "warnings": r["warnings"]}


def _row(m: TakeMeta) -> dict:
    ev = m.event or {}
    return {"take_id": m.take_id, "kind": m.kind, "emotion": m.emotion, "intensity": m.intensity,
            "intention": m.intention or "", "status": m.status, "event_present": int(bool(ev.get("present"))),
            "azimuth_rad": ev.get("azimuth_rad") if ev.get("present") else "",
            "elevation_rad": ev.get("elevation_rad") if ev.get("present") else "",
            "distance_m": (ev.get("distance_m") if ev.get("distance_m") is not None else "") if ev.get("present") else "",
            "duration_s": round(m.duration_s, 4), "segment_start_s": m.segment[0], "segment_end_s": m.segment[1],
            "n_samples": m.n_samples, "sample_rate_hz": m.sample_rate_hz,
            "calibration_version": m.calibration_version, "created_at": m.created_at,
            "tags": ";".join((m.annotations or {}).get("tags", []))}


def _stats(metas: Sequence[TakeMeta]) -> dict:
    by_kind = Counter(m.kind for m in metas)
    by_emotion = {k: dict(Counter(m.emotion for m in metas if m.kind == k)) for k in by_kind}
    dur = {k: round(sum(m.duration_s for m in metas if m.kind == k), 1) for k in by_kind}
    joints = {k: next(m.joints for m in metas if m.kind == k) for k in by_kind}
    with_event = {k: sum(1 for m in metas if m.kind == k and (m.event or {}).get("present")) for k in by_kind}
    rates = sorted({float(m.sample_rate_hz) for m in metas})
    return {"takes": len(metas), "by_kind": dict(by_kind), "by_emotion": by_emotion, "duration_s": dur,
            "joints": joints, "with_event": with_event, "sample_rates_hz": rates}


def export_dataset(data_dir, out_dir, config_root=None, runs_dir=None, *, name: str = "nao-emotion-motion-dataset",
                   version: Optional[str] = None, license_id: str = "CC-BY-4.0", author: str = "",
                   statuses: Sequence[str] = ("accepted",), include_synthetic: bool = False,
                   include_incomplete: bool = False, strip_notes: bool = False,
                   kinds: Optional[Iterable[str]] = None, generator: bool = True) -> dict:
    """Writes (or updates) the export folder. Returns a summary.

    generator: also write generator/ (nao_gesture.py + nao_gesture.json) from the exported models,
    when `runs_dir` is given. If it cannot be built (no active calibration, no numpy model...),
    the dataset is exported without it and the reason is in the summary (`generator.error`)."""
    from ..config import DEFAULT_CONFIG_DIR
    if license_id not in LICENSES:
        raise DatasetError(f"Licence inconnue : {license_id} (choix : {', '.join(LICENSES)})")
    data_dir, out = Path(data_dir), Path(out_dir)
    config_root = Path(config_root) if config_root else DEFAULT_CONFIG_DIR
    if out.resolve() == data_dir.resolve() or data_dir.resolve() in out.resolve().parents:
        raise DatasetError("Le dossier d'export doit être en dehors du dossier des données")
    if out.exists() and any(p.name != ".git" for p in out.iterdir()) and not (out / MANIFEST).exists():
        raise DatasetError(f"{out} n'est pas vide et n'est pas un export précédent (pas de {MANIFEST})")
    metas = select_takes(data_dir, statuses, include_synthetic, include_incomplete, kinds)
    if not metas:
        raise DatasetError("Aucune prise à exporter avec ces filtres")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

    # takes: exact copies (meta.json possibly without the free-text notes)
    keep = set()
    for m in metas:
        src, dst = data_dir / "takes" / m.take_id, out / "takes" / m.take_id
        existed = (dst / "meta.json").exists()
        keep.add(m.take_id)
        changed = _copy_if_changed(src / "samples.npz", dst / "samples.npz")
        if strip_notes:
            d = m.to_dict()
            d["annotations"] = dict(d.get("annotations") or {}, notes="")
            text = json.dumps(d, indent=2, ensure_ascii=False)
            if not (dst / "meta.json").exists() or (dst / "meta.json").read_text(encoding="utf-8") != text:
                (dst / "meta.json").write_text(text, encoding="utf-8")
                changed = True
        else:
            changed = _copy_if_changed(src / "meta.json", dst / "meta.json") or changed
        summary["updated" if existed and changed else "added" if not existed else "unchanged"] += 1
    for d in sorted((out / "takes").glob("*")) if (out / MANIFEST).exists() else []:
        if d.is_dir() and d.name not in keep and (d / "meta.json").exists():
            shutil.rmtree(d)
            summary["removed"] += 1

    # referenced calibrations, configuration needed for reading
    cals = _calibration_files(config_root)
    placeholder = _placeholder_version(config_root)
    uncalibrated = sum(1 for m in metas if m.calibration_version == placeholder)
    used = sorted({m.calibration_version for m in metas} - {placeholder})
    missing_cal = [v for v in used if v not in cals]
    for v in used:
        if v in cals:
            _copy_if_changed(cals[v], out / "calibration" / cals[v].name)
    for f in CONFIG_FILES:
        _copy_if_changed(config_root / f, out / "configs" / f)

    # latest models
    models = {}
    if runs_dir is not None:
        for c, run in _latest_runs(Path(runs_dir)).items():
            files = [f for f in MODEL_FILES if (run / f).exists()]
            for f in files:
                _copy_if_changed(run / f, out / "models" / c / f)
            models[c] = {"run": run.name, "files": files}
    for d in sorted((out / "models").glob("*")) if (out / "models").exists() else []:
        if d.is_dir() and d.name not in models:
            shutil.rmtree(d)

    # ready-to-use generator built from those same models
    gen = None
    if runs_dir is not None and generator and models:
        try:
            gen = _export_generator(config_root, Path(runs_dir), out / GENERATOR_DIR)
        except Exception as e:  # never blocks the dataset export
            gen = {"error": str(e)}
    if (gen is None or "error" in gen) and (out / GENERATOR_DIR).exists():
        shutil.rmtree(out / GENERATOR_DIR)  # no stale generator next to newer models

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for m in metas:
            w.writerow(_row(m))

    stats = _stats(metas)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    version = version or _dt.date.today().isoformat()
    lic_name, lic_url = LICENSES[license_id]
    (out / "LICENSE").write_text(
        f"The data in this folder (takes/, calibration/, index.csv, models/) is licensed under the\n"
        f"{lic_name} ({license_id}).\n\nFull text: {lic_url}\n\n"
        f"Copyright (c) {_dt.date.today().year} {author or 'the dataset authors'}\n\n"
        f"The code that produced it (https://github.com/Monolithoid/nao-emotion-motion) is MIT licensed.\n",
        encoding="utf-8")
    filters = {"statuses": list(statuses), "include_synthetic": include_synthetic,
               "include_incomplete": include_incomplete}
    card = _card(name, version, license_id, author, stats, [v for v in used if v in cals], models, filters,
                 gen if gen and "error" not in gen else None, uncalibrated)
    readme = out / "README.md"
    prev = (json.loads((out / MANIFEST).read_text(encoding="utf-8")).get("files") or {}) \
        if (out / MANIFEST).exists() else {}
    if readme.exists() and prev.get("README.md") and _sha256(readme) != prev["README.md"]:
        # README edited by hand since the last export: keep it, updated card next to it
        (out / "README.generated.md").write_text(card, encoding="utf-8")
        summary["readme_kept"] = True
    else:
        readme.write_text(card, encoding="utf-8")
        (out / "README.generated.md").unlink(missing_ok=True)

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(out).parts and p.name != MANIFEST:
            files[p.relative_to(out).as_posix()] = _sha256(p)
    manifest = {"format": DATASET_FORMAT, "name": name, "version": version, "exported_at": now,
                "license": license_id, "take_format_version": metas[0].format_version,
                "filters": {"statuses": list(statuses), "include_synthetic": include_synthetic,
                            "include_incomplete": include_incomplete, "strip_notes": strip_notes,
                            "kinds": sorted(kinds or COMPONENTS)},
                "stats": stats, "calibrations": used, "missing_calibrations": missing_cal,
                "uncalibrated_takes": uncalibrated, "models": models, "generator": gen, "files": files}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary.update(takes=len(metas), calibrations=len(used) - len(missing_cal), missing_calibrations=missing_cal,
                   uncalibrated_takes=uncalibrated, models=sorted(models), generator=gen, out=str(out))
    return summary


def _card(name, version, license_id, author, stats, calibrations, models, filters, generator=None,
          uncalibrated=0) -> str:
    kinds_txt = {"arms": "arms guided by hand (head held stiff)",
                 "head": "head guided by hand (arms held stiff)",
                 "hands": "hands driven by the operator from the interface (opening 0..1, not an angle)"}
    lines = [
        "---",
        f"license: {license_id.lower()}",
        "task_categories:\n- robotics",
        "tags:\n- nao\n- emotion\n- gesture-generation\n- kinesthetic-teaching\n- human-robot-interaction",
        f"pretty_name: {name}",
        "---", "",
        f"# {name}", "",
        f"Version {version}. Expressive gestures demonstrated on a real NAO robot (NAO V4, NAOqi 2.1), "
        "recorded with [nao-emotion-motion](https://github.com/Monolithoid/nao-emotion-motion). "
        "Each take is a short movement performed for a prompt: an emotion, an intensity in [0, 1], and an "
        "optional event (a point the robot reacts to, given as azimuth, elevation and distance in the torso frame). "
        "The goal is to train generators (emotion + intensity + event -> gesture), not emotion recognisers.", "",
        f"Author: {author}" if author else "", "",
        "## Content", "",
        "| Group | Takes | With event | Total duration (s) | Joints |",
        "|---|---|---|---|---|",
    ]
    for k in COMPONENTS:
        if k in stats["by_kind"]:
            lines.append(f"| {k} | {stats['by_kind'][k]} | {stats['with_event'][k]} | {stats['duration_s'][k]} | "
                         f"{', '.join(stats['joints'][k])} |")
    lines += ["", "Takes per emotion:", ""]
    for k, emo in stats["by_emotion"].items():
        lines.append(f"- {k}: " + ", ".join(f"{e} {n}" for e, n in sorted(emo.items())))
    lines += [
        "", "Groups:", "",
        *[f"- `{k}`: {kinds_txt[k]}" for k in COMPONENTS if k in stats["by_kind"]],
        "", f"Selection: status in {filters['statuses']}, "
        + ("simulated takes INCLUDED (flag `synthetic` in meta.json), " if filters["include_synthetic"]
           else "recorded on the real robot (no simulated take), ")
        + ("interrupted recordings included (tag `incomplete`). " if filters["include_incomplete"]
           else "complete recordings only. ")
        + "Emotion labels are working labels in French "
        "(see `configs/emotions.yaml`), not a scientific taxonomy. All takes come from a single operator "
        "and a single robot.", "",
        "## Layout", "",
        "```",
        "takes/<take_id>/meta.json     prompt (emotion, intensity, event), useful segment, status, calibration",
        "takes/<take_id>/samples.npz   t (N,) in s ; q (N, J) MEASURED positions in rad ; joints (J,)",
        "                              hands: opening in [0, 1] ; ref_t / ref_q / ref_joints = command sent",
        "index.csv                     one row per take",
        "calibration/*.yaml            joint limits measured on the robot (cited by meta.calibration_version)",
        "configs/                      emotions.yaml, joints.yaml",
        *(["models/<group>/               trained models (PyTorch and numpy)"] if models else []),
        *(["generator/                    nao_gesture.py + nao_gesture.json: the models, ready to use"]
          if generator else []),
        "dataset.json                  manifest with sha256 of every file",
        "```", "",
        f"Sample rates: {', '.join(f'{r:g} Hz' for r in stats['sample_rates_hz'])}. "
        "Only the `segment` [start, end] of each take is the useful gesture. Raw samples are never resampled.", "",
        "```python",
        "import json, numpy as np",
        "meta = json.load(open('takes/<take_id>/meta.json'))",
        "z = np.load('takes/<take_id>/samples.npz')",
        "t, q, joints = z['t'], z['q'], list(z['joints'])",
        "```", "",
        "To use the takes with the project: `nao-motion import-dataset <this folder>`, then `nao-motion train`.", "",
    ]
    if generator:
        g_models = ", ".join(f"{c} (run `{r}`)" for c, r in generator["models"].items())
        lines += [
            "## Ready-to-use gesture generator", "",
            "`generator/` holds the trained models as two files to copy into another project, nothing to install: "
            "`nao_gesture.py` (generator + safety guardian, Python 2.7 and 3, standard library only: no PyTorch, "
            "no numpy, runs on the robot's own Python) and `nao_gesture.json` (network weights, settings, "
            f"guardian limits). Models: {g_models}"
            + (f"; no model for {', '.join(generator['missing'])} (arms/head held in the start position, "
               "hands not commanded)" if generator["missing"] else "")
            + f". Guardian limits: calibration `{generator['calibration_version']}` of the robot the takes were "
            "recorded on; another NAO should be calibrated and the generator re-exported with the project "
            "(`nao-motion export-generator`).", "",
            "```python",
            "from nao_gesture import GestureGenerator",
            "gen = GestureGenerator.load('generator/nao_gesture.json')",
            "print(gen.emotions)",
            "g = gen.generate('joie', 0.7,                       # emotion (or a mix {'joie': 0.7, 'surprise': 0.3}), intensity 0..1",
            "                 event={'azimuth_deg': 30, 'elevation_deg': 0, 'distance_m': 1.2},  # optional, torso frame",
            "                 current_pose=measured_posture)     # {joint: rad}, adds the transitions",
            "if g.ok:                                            # accepted by the guardian",
            "    names, angles, times = g.angle_interpolation()",
            "    motion.angleInterpolation(names, angles, times, True)   # ALMotion, absolute angles",
            "else:",
            "    print(g.report.summary())                       # never send a refused gesture",
            "```", "",
            "Azimuth is positive to the robot's left, elevation positive up, 0 = straight ahead. Without "
            "`current_pose`, only the gesture is returned and the robot must be brought to the start position first. "
            "The library never commands the robot itself. Quick check from a shell: "
            "`python generator/nao_gesture.py generator/nao_gesture.json joie 0.7 30 0 1.2` "
            "(exit code 0 accepted, 1 refused). Full documentation: "
            "[Gesture generator library](https://github.com/Monolithoid/nao-emotion-motion#gesture-generator-library-export-for-another-project).", "",
        ]
    if calibrations:
        lines += ["## Calibrations", "", *[f"- `{v}`" for v in calibrations], ""]
    if uncalibrated:
        lines += [f"{uncalibrated} take(s) were recorded before the first calibration (`calibration_version` "
                  "is the placeholder name of `configs/joints.yaml`): their positions are measured like the "
                  "others, only no calibration file goes with them.", ""]
    if models:
        lines += ["## Models", "",
                  "`models/<group>/model.pt` (PyTorch) and `model_numpy.npz` (numpy only), last training of each "
                  "group at export time:", "", *[f"- {c}: run `{m['run']}`" for c, m in models.items()], ""]
    lines += ["## License", "", f"Data: {license_id} ({LICENSES[license_id][1]}). Code: MIT.", ""]
    return "\n".join(l for l in lines if l is not None) + "\n"


def import_dataset(src_dir, data_dir, config_root=None, runs_dir=None, *, verify: bool = True) -> dict:
    """Copies an export into data/takes/ (and configs/calibration/, runs/). Never overwrites anything."""
    from ..config import DEFAULT_CONFIG_DIR
    src, data_dir = Path(src_dir), Path(data_dir)
    config_root = Path(config_root) if config_root else DEFAULT_CONFIG_DIR
    man_p = src / MANIFEST
    if not man_p.exists():
        raise DatasetError(f"{src} n'est pas un export (pas de {MANIFEST})")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man.get("format") != DATASET_FORMAT:
        raise DatasetError(f"Format d'export {man.get('format')} non supporté")
    files = man.get("files") or {}
    if verify:
        bad = [rel for rel, h in files.items() if rel.startswith(DATA_PREFIXES)
               and (not (src / rel).exists() or _sha256(src / rel) != h)]
        if bad:
            raise DatasetError(f"{len(bad)} fichier(s) absent(s) ou modifié(s), rien n'est importé : {bad[:5]}")
    res = {"takes_added": 0, "takes_skipped": 0, "calibrations_added": [], "models_added": []}
    for d in sorted((src / "takes").glob("*")):
        if not (d / "meta.json").exists():
            continue
        _read_meta(d / "meta.json")  # raises if the format cannot be read
        dst = data_dir / "takes" / d.name
        if dst.exists():
            res["takes_skipped"] += 1
            continue
        dst.mkdir(parents=True)
        for f in ("samples.npz", "meta.json"):  # meta.json last: the take is visible once complete
            shutil.copyfile(d / f, dst / f)
        res["takes_added"] += 1
    cal_dir = config_root / "calibration"
    have = set(_calibration_files(config_root))
    from ..calibration import Calibration
    for f in sorted((src / "calibration").glob("*.yaml")):
        v = Calibration.load(f).version
        if v in have or (cal_dir / f.name).exists():
            continue
        cal_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, cal_dir / f.name)
        res["calibrations_added"].append(f.name)
    if runs_dir is not None:
        runs = Path(runs_dir)
        for c, m in (man.get("models") or {}).items():
            run = runs / m["run"]
            if not run.exists():
                run.mkdir(parents=True)
                for f in m.get("files", []):
                    shutil.copyfile(src / "models" / c / f, run / f)
                res["models_added"].append(c)
            if not (runs / f"latest_{c}.txt").exists():
                (runs / f"latest_{c}.txt").write_text(m["run"])
    return res
