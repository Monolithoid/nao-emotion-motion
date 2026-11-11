"""Export et import du jeu de données (prises, calibrations, derniers modèles).

`export_dataset` produit un dossier autonome, prêt à être poussé tel quel dans un
dépôt séparé (GitHub, ou plus tard Hugging Face, qui est aussi un dépôt Git) :

    <out>/README.md                fiche du jeu de données (anglais, pour la communauté)
    <out>/LICENSE                  licence des DONNÉES (CC BY 4.0 par défaut ; le code reste MIT)
    <out>/dataset.json             manifeste : version, filtres, comptes, sha256 de chaque fichier
    <out>/index.csv                une ligne par prise (consigne, événement, durée, ...)
    <out>/takes/<id>/meta.json     copies EXACTES des prises (même format que data/takes/)
    <out>/takes/<id>/samples.npz
    <out>/calibration/*.yaml       calibrations citées par les prises exportées
    <out>/configs/                 emotions.yaml et joints.yaml (lecture des étiquettes et articulations)
    <out>/models/<groupe>/         derniers modèles bras / tête / mains (option)

Par défaut, seules les prises ACCEPTÉES, issues du robot réel (pas `synthetic`) et
complètes (sans le tag `incomplete`) sont exportées. Ré-exporter dans le même
dossier le met à jour : prises nouvelles ajoutées, métadonnées modifiées
réécrites, prises sorties de la sélection retirées (seulement dans takes/, et
seulement si le dossier est déjà un export : présence de dataset.json). Le
dossier .git éventuel n'est jamais touché.

`import_dataset` fait l'inverse (reprise sur un autre PC) : il recopie dans
data/takes/ les prises absentes, vérifie les sha256, ajoute les calibrations
absentes à configs/calibration/ (sans en activer aucune) et, si demandé, les
modèles dans runs/. Rien n'est jamais écrasé.
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
LICENSES = {
    "CC-BY-4.0": ("Creative Commons Attribution 4.0 International",
                  "https://creativecommons.org/licenses/by/4.0/"),
    "CC-BY-SA-4.0": ("Creative Commons Attribution-ShareAlike 4.0 International",
                     "https://creativecommons.org/licenses/by-sa/4.0/"),
    "CC0-1.0": ("Creative Commons Zero v1.0 Universal (public domain dedication)",
                "https://creativecommons.org/publicdomain/zero/1.0/"),
}
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
    """Prises retenues pour l'export, triées par identifiant."""
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
    """{version: fichier} pour les calibrations lisibles de configs/calibration/."""
    from ..calibration import Calibration
    out = {}
    for f in sorted((config_root / "calibration").glob("*.yaml")):
        try:
            out.setdefault(Calibration.load(f).version, f)
        except Exception:
            continue
    return out


def _latest_runs(runs_dir: Path) -> Dict[str, Path]:
    """Même règle que training.latest(), sans importer torch (export possible sans PyTorch)."""
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
                   kinds: Optional[Iterable[str]] = None) -> dict:
    """Écrit (ou met à jour) le dossier d'export. Retourne un résumé."""
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

    # prises : copies exactes (meta.json éventuellement sans les notes libres)
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

    # calibrations citées, configuration utile à la lecture
    cals = _calibration_files(config_root)
    used = sorted({m.calibration_version for m in metas})
    missing_cal = [v for v in used if v not in cals]
    for v in used:
        if v in cals:
            _copy_if_changed(cals[v], out / "calibration" / cals[v].name)
    for f in CONFIG_FILES:
        _copy_if_changed(config_root / f, out / "configs" / f)

    # derniers modèles
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
    (out / "README.md").write_text(_card(name, version, license_id, author, stats,
                                         [v for v in used if v in cals], models, filters),
                                   encoding="utf-8")

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
                "models": models, "files": files}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary.update(takes=len(metas), calibrations=len(used) - len(missing_cal), missing_calibrations=missing_cal,
                   models=sorted(models), out=str(out))
    return summary


def _card(name, version, license_id, author, stats, calibrations, models, filters) -> str:
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
    if calibrations:
        lines += ["## Calibrations", "", *[f"- `{v}`" for v in calibrations], ""]
    if models:
        lines += ["## Models", "",
                  "`models/<group>/model.pt` (PyTorch) and `model_numpy.npz` (numpy only), last training of each "
                  "group at export time:", "", *[f"- {c}: run `{m['run']}`" for c, m in models.items()], ""]
    lines += ["## License", "", f"Data: {license_id} ({LICENSES[license_id][1]}). Code: MIT.", ""]
    return "\n".join(l for l in lines if l is not None) + "\n"


def import_dataset(src_dir, data_dir, config_root=None, runs_dir=None, *, verify: bool = True) -> dict:
    """Recopie un export dans data/takes/ (et configs/calibration/, runs/). N'écrase jamais rien."""
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
        bad = [rel for rel, h in files.items() if not (src / rel).exists() or _sha256(src / rel) != h]
        if bad:
            raise DatasetError(f"{len(bad)} fichier(s) absent(s) ou modifié(s), rien n'est importé : {bad[:5]}")
    res = {"takes_added": 0, "takes_skipped": 0, "calibrations_added": [], "models_added": []}
    for d in sorted((src / "takes").glob("*")):
        if not (d / "meta.json").exists():
            continue
        _read_meta(d / "meta.json")  # lève si le format n'est pas lisible
        dst = data_dir / "takes" / d.name
        if dst.exists():
            res["takes_skipped"] += 1
            continue
        dst.mkdir(parents=True)
        for f in ("samples.npz", "meta.json"):  # meta.json en dernier : prise visible une fois complète
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
