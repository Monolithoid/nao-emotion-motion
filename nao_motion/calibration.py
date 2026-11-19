"""Guardian calibration: a dedicated procedure that MEASURES the limits on the robot.

The guardian only relies on "real" limits if a calibration file is active. That
file is produced by a full procedure of its own, robot by robot:

  1. robot identification (name, NAOqi, configuration, factory limits read
     through ALMotion.getLimits when the bridge is connected);
  2. guided exploration, group by group: the group is relaxed (read-only),
     the others are held; the operator slowly moves each joint to its useful
     end stops and goes through the combinations (e.g. head turned AND
     lowered). Nothing is commanded during this step;
  3. capture of the rest and start poses (sensor readings);
  4. computation: limits = measured range ∩ factory limits, minus a margin;
     coupled constraints = convex hull of the explored pairs, simplified
     conservatively, minus the margin;
  5. review by the operator, then versioned saving and activation.

Limits of the method: the convex hull assumes a convex reachable area for each
pair; an L-shaped area must be split by hand into several pairs/constraints.
Collisions between limbs (arm ↔ head, arm ↔ arm) cannot be measured this way
and remain manual constraints in guardian.yaml.
"""
from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

CAL_KIND = "nao_motion.calibration"


class CalibrationError(ValueError):
    pass


# -- geometry -------------------------------------------------------------------
def convex_hull(points: np.ndarray) -> np.ndarray:
    """2D convex hull (monotone chain), vertices counter-clockwise."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def simplify_hull(hull: np.ndarray, max_vertices: int) -> np.ndarray:
    """Removes the vertices with the smallest area (Visvalingam). Removing a vertex from a
    convex polygon shrinks it: the simplification stays conservative."""
    h = list(map(tuple, hull))
    while len(h) > max(3, max_vertices):
        n = len(h)
        areas = []
        for i in range(n):
            a, b, c = np.array(h[i - 1]), np.array(h[i]), np.array(h[(i + 1) % n])
            areas.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2)
        h.pop(int(np.argmin(areas)))
    return np.array(h)


def hull_constraints(hull: np.ndarray, margin: float) -> List[Tuple[float, float, float]]:
    """Convex polygon (counter-clockwise) -> half-planes n·q <= b, shrunk by `margin`."""
    out = []
    n = len(hull)
    if n < 3:
        return out
    for i in range(n):
        p, q = hull[i], hull[(i + 1) % n]
        d = q - p
        norm = float(np.hypot(*d))
        if norm < 1e-9:
            continue
        nx, ny = d[1] / norm, -d[0] / norm  # outward normal
        out.append((float(nx), float(ny), float(nx * p[0] + ny * p[1] - margin)))
    return out


def zone_polygon(constraints: Sequence[Tuple[float, float, float]],
                 box: Tuple[float, float, float, float]) -> np.ndarray:
    """Polygon of the retained area: half-planes n·q <= b ∩ box (xmin, xmax, ymin, ymax).
    Display only (Sutherland-Hodgman); empty if the area is empty."""
    x0, x1, y0, y1 = box
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for nx, ny, b in constraints:
        out = []
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            fp, fq = nx * p[0] + ny * p[1] - b, nx * q[0] + ny * q[1] - b
            if fp <= 0:
                out.append(p)
            if (fp <= 0) != (fq <= 0):
                s = fp / (fp - fq)
                out.append((p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])))
        poly = out
        if not poly:
            break
    return np.array(poly)


def in_zone(pt, constraints, box) -> bool:
    x, y = float(pt[0]), float(pt[1])
    return (box[0] <= x <= box[1] and box[2] <= y <= box[3]
            and all(nx * x + ny * y <= b + 1e-9 for nx, ny, b in constraints))


# -- calibration file --------------------------------------------------------
@dataclass
class Calibration:
    version: str
    robot: dict
    limits: Dict[str, Tuple[float, float]]
    measured: Dict[str, dict]
    poses: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    coupled: List[dict] = field(default_factory=list)
    margin_rad: float = 0.05
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    notes: str = ""
    warnings: List[str] = field(default_factory=list)
    path: Optional[Path] = None
    # optional per-joint speed limits (e.g. measured on the takes, see calibration_extend.py):
    # {"source": str, "velocity_rad_s": {joint: v}, "acceleration_rad_s2": {joint: a}}
    kinematics: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"schema_version": 1, "kind": CAL_KIND, "version": self.version, "created_at": self.created_at,
                "robot": self.robot, "margin_rad": self.margin_rad, "unit": "rad", "notes": self.notes,
                "limits": {j: [float(a), float(b)] for j, (a, b) in self.limits.items()},
                "measured": self.measured, "poses": self.poses, "coupled_constraints": self.coupled,
                "warnings": list(self.warnings), **({"kinematics": self.kinematics} if self.kinematics else {})}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise CalibrationError(f"{path} existe déjà : une calibration n'est jamais écrasée")
        path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.path = path
        return path

    @classmethod
    def load(cls, path) -> "Calibration":
        path = Path(path)
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        if d.get("kind") != CAL_KIND or d.get("schema_version") != 1 or d.get("unit") != "rad":
            raise CalibrationError(f"{path.name} n'est pas un fichier de calibration valide")
        lim = {}
        for j, (a, b) in d["limits"].items():
            a, b = float(a), float(b)
            if not (np.isfinite(a) and np.isfinite(b) and a < b):
                raise CalibrationError(f"{path.name}: limite invalide pour {j}")
            lim[j] = (a, b)
        return cls(version=str(d["version"]), robot=d.get("robot") or {}, limits=lim,
                   measured=d.get("measured") or {}, poses=d.get("poses") or {},
                   coupled=d.get("coupled_constraints") or [], margin_rad=float(d.get("margin_rad", 0)),
                   created_at=str(d.get("created_at", "")), notes=d.get("notes", ""),
                   warnings=list(d.get("warnings") or []), path=path, kinematics=_kinematics(d, path))


def _kinematics(d: dict, path: Path) -> Optional[dict]:
    k = d.get("kinematics")
    if not k:
        return None
    out = {"source": str(k.get("source", ""))}
    for key in ("velocity_rad_s", "acceleration_rad_s2"):
        vals = {}
        for j, v in (k.get(key) or {}).items():
            v = float(v)
            if not (np.isfinite(v) and v > 0):
                raise CalibrationError(f"{path.name}: kinematics {key} invalide pour {j}")
            vals[str(j)] = v
        out[key] = vals
    return out


def active_calibration_path(config_root) -> Optional[Path]:
    p = Path(config_root) / "calibration" / "ACTIVE"
    if not p.exists():
        return None
    name = p.read_text(encoding="utf-8").strip()
    if not name:
        return None
    f = (Path(config_root) / "calibration" / name)
    if not f.exists():
        raise CalibrationError(f"Calibration active introuvable : {f}")
    return f


def is_simulated(cal: "Calibration") -> bool:
    """Calibration produced in simulation (synthetic demonstrator): never activatable."""
    r = cal.robot or {}
    return r.get("adapter") == "sim" or r.get("backend") == "sim"


def check_compatible(config_root, cal_path, allow_simulated: bool = False) -> "Calibration":
    """Checks that a calibration file is usable with the CURRENT configuration
    (format, joints, poses within the limits…) without activating anything. Raises
    CalibrationError with the reason otherwise."""
    from .config import ConfigError, load_config
    cal = Calibration.load(cal_path)
    if is_simulated(cal) and not allow_simulated:
        raise CalibrationError("calibration simulée : valeurs sans portée réelle, activation impossible")
    try:
        load_config(config_root, calibration_file=cal_path)
    except ConfigError as e:
        raise CalibrationError(str(e)) from e
    return cal


def list_calibrations(config_root, allow_simulated: bool = False) -> List[dict]:
    """Files in configs/calibration/, newest first, with their state:
    active, activatable (otherwise the reason), calibrated groups and poses present."""
    d = Path(config_root) / "calibration"
    try:
        active = active_calibration_path(config_root)
    except CalibrationError:
        active = None
    out = []
    for f in sorted(d.glob("*.yaml")) if d.exists() else []:
        e = {"file": f.name, "active": bool(active and f.resolve() == active.resolve())}
        try:
            cal = Calibration.load(f)
        except Exception as ex:  # unreadable file or of another type
            e.update(version=None, valid=False, activable=False, reason=str(ex))
            out.append(e)
            continue
        e.update(version=cal.version, created_at=cal.created_at, robot=cal.robot, valid=True,
                 simulated=is_simulated(cal), joints=sorted(cal.limits),
                 poses={n: sorted((cal.poses or {}).get(n, {})) for n in ("rest", "start")},
                 notes=cal.notes)
        try:
            check_compatible(config_root, f, allow_simulated)
            e.update(activable=True, reason=None)
        except CalibrationError as ex:
            e.update(activable=False, reason=str(ex))
        out.append(e)
    out.sort(key=lambda e: (e.get("created_at") or "", e["file"]), reverse=True)
    return out


def activate(config_root, cal_path, allow_simulated: bool = False) -> Path:
    """Points the guardian to a file in configs/calibration/. The file must be readable,
    compatible with the current configuration and, unless `allow_simulated` (trials in
    simulation), come from the real robot: otherwise nothing changes."""
    cal_path = Path(cal_path)
    d = Path(config_root) / "calibration"
    if not cal_path.is_absolute() and not cal_path.exists():
        cal_path = d / cal_path  # bare name: file in configs/calibration/
    if cal_path.resolve().parent != d.resolve():
        raise CalibrationError("Le fichier doit se trouver dans configs/calibration/")
    if not cal_path.exists():
        raise CalibrationError(f"Calibration introuvable : {cal_path.name}")
    check_compatible(config_root, cal_path, allow_simulated)  # validate BEFORE writing ACTIVE
    (d / "ACTIVE").write_text(cal_path.name + "\n", encoding="utf-8")
    return d / "ACTIVE"


# -- calcul --------------------------------------------------------------------------
def compute_calibration(sweeps: Dict[str, Dict[str, np.ndarray]], joints: Sequence[str], proc: dict,
                        factory: Optional[Dict[str, Sequence[float]]] = None, poses=None, robot=None,
                        version: Optional[str] = None, imported: Optional[Dict[str, dict]] = None) -> Calibration:
    """sweeps: {group: {"joints": [...], "q": (N, J)}} measured during exploration.
    imported: {group: import_group(...)} groups taken as is from a saved
    calibration (limits, measurements, coupled constraints) instead of being explored again."""
    imported = imported or {}
    imp_lim = {j: tuple(v) for g in imported.values() for j, v in g["limits"].items()}
    imp_meas = {j: dict(m, imported_from=g["source"]) for g in imported.values() for j, m in g["measured"].items()}
    margin = float(proc.get("margin_rad", 0.05))
    min_range = float(proc.get("min_range_rad", 0.2))
    min_samples = int(proc.get("min_samples", 50))
    samples: Dict[str, np.ndarray] = {}
    for g, s in sweeps.items():
        q = np.asarray(s["q"], dtype=float)
        if q.shape[0] < min_samples:
            raise CalibrationError(f"Groupe {g} : {q.shape[0]} échantillons, {min_samples} requis")
        if not np.all(np.isfinite(q)):
            raise CalibrationError(f"Groupe {g} : mesures non finies")
        for k, j in enumerate(s["joints"]):
            samples[j] = q[:, k]
    missing = [j for j in joints if j not in samples and j not in imp_lim]
    if missing:
        raise CalibrationError(f"Articulations non explorées : {missing}")
    tol = float(proc.get("factory_tolerance_rad", 0.005))  # bruit capteur
    limits, measured, warnings = {}, {}, []
    for j in joints:
        if j in imp_lim and j not in samples:
            limits[j], measured[j] = imp_lim[j], imp_meas.get(j) or {}
            continue
        lo, hi = float(samples[j].min()), float(samples[j].max())
        if hi - lo < min_range:
            raise CalibrationError(f"{j} : amplitude explorée {hi - lo:.2f} rad < {min_range} rad. "
                                   "Explorer toute l'amplitude utile.")
        flo, fhi = (factory or {}).get(j, (lo, hi))[:2]
        flo, fhi = float(flo), float(fhi)
        # Exploration beyond a factory limit: not refused, the factory limit is kept
        # (intersection) and it is reported.
        beyond = []
        if lo < flo - tol:
            beyond.append(f"min exploré {lo:.3f} < limite d'usine {flo:.3f}")
        if hi > fhi + tol:
            beyond.append(f"max exploré {hi:.3f} > limite d'usine {fhi:.3f}")
        if beyond:
            warnings.append(f"{j} : {' ; '.join(beyond)} rad. Limite d'usine retenue.")
        a, b = max(lo, flo) + margin, min(hi, fhi) - margin
        if not a < b:
            raise CalibrationError(f"{j} : amplitude nulle après marge")
        limits[j] = (a, b)
        measured[j] = {"min": lo, "max": hi, "factory": [float(flo), float(fhi)] if factory and j in factory else None,
                       "n": int(samples[j].size), "beyond_factory": bool(beyond)}
    coupled = []
    for g in imported.values():
        coupled.extend(dict(c) for c in g["coupled"])
    for a, b in proc.get("pairs", []):
        if a in imp_lim or b in imp_lim:
            continue  # hull taken over with the group
        if a not in samples or b not in samples or len(samples[a]) != len(samples[b]):
            continue  # pair not explored together (different groups)
        pts = np.stack([samples[a], samples[b]], axis=1)
        hull = simplify_hull(convex_hull(pts), int(proc.get("max_hull_edges", 12)))
        for k, (na, nb, bound) in enumerate(hull_constraints(hull, margin)):
            coupled.append({"name": f"calib:{a}-{b}:{k}", "terms": {a: round(na, 6), b: round(nb, 6)},
                            "bound": round(bound, 6), "description": f"enveloppe mesurée {a}/{b}"})
    if poses:
        # Rest may be outside the movement zone (e.g. arms resting on the knees:
        # contact): it is only bounded by the factory limits. Making sure the start pose
        # can be reached smoothly from rest is up to whoever calibrates.
        for name, groups in poses.items():
            for g, jq in groups.items():
                for j, v in jq.items():
                    if name == "rest":
                        m = measured.get(j) or {}
                        flo, fhi = m["factory"] if m.get("factory") else (min(m.get("min", limits[j][0]), limits[j][0]),
                                                                          max(m.get("max", limits[j][1]), limits[j][1]))
                        if not flo - tol <= v <= fhi + tol:
                            raise CalibrationError(f"Pose rest/{j}={v:.3f} hors des limites d'usine [{flo:.3f}, {fhi:.3f}]")
                        continue
                    lo, hi = limits[j]
                    if not lo <= v <= hi:
                        raise CalibrationError(f"Pose {name}/{j}={v:.3f} hors des limites calibrées [{lo:.3f}, {hi:.3f}]")
        for c in coupled:
            for name, groups in poses.items():
                if name == "rest":
                    continue
                flat = {j: v for jq in groups.values() for j, v in jq.items()}
                if all(j in flat for j in c["terms"]):
                    val = sum(k * flat[j] for j, k in c["terms"].items())
                    if val > c["bound"]:
                        raise CalibrationError(f"Pose {name} hors de l'enveloppe mesurée ({c['description']}) : "
                                               "explorer une zone plus large autour de cette pose")
    robot = robot or {}
    version = version or f"{robot.get('name') or 'nao'}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    notes = []
    for g, d in sorted(imported.items()):
        changed = [n for n in ("rest", "start")
                   if n in d["poses"] and (poses or {}).get(n, {}).get(g) is not None
                   and any(abs(poses[n][g][j] - v) > 1e-9 for j, v in d["poses"][n].items() if j in poses[n][g])]
        notes.append(f"groupe {g} repris de la calibration {d['source']}" +
                     (f" (pose{'s' if len(changed) > 1 else ''} {' et '.join(POSE_FR[n] for n in changed)} "
                      "recapturée" + ("s" if len(changed) > 1 else "") + ")" if changed else ""))
    # Nothing explored, everything taken from the same calibration (e.g. only the rest pose changes):
    # its margin and velocities are kept as they are.
    if not samples and imported and len({d.get("margin_rad") for d in imported.values()}) == 1:
        m0 = next(iter(imported.values())).get("margin_rad")
        if m0 is not None:
            margin = float(m0)
    kinematics = _merge_kinematics(imported, set(samples))
    return Calibration(version=version, robot=robot, limits=limits, measured=measured, poses=poses or {},
                       coupled=coupled, margin_rad=margin, warnings=warnings, notes="; ".join(notes),
                       kinematics=kinematics)


POSE_FR = {"rest": "de repos", "start": "de départ"}


def _merge_kinematics(imported: Dict[str, dict], swept) -> Optional[dict]:
    """Velocities of the imported groups (a joint explored again loses its own:
    they only hold for the area where they were measured)."""
    out = {"velocity_rad_s": {}, "acceleration_rad_s2": {}}
    sources = []
    for g, d in sorted(imported.items()):
        k = d.get("kinematics")
        if not k:
            continue
        if k.get("source") and k["source"] not in sources:
            sources.append(k["source"])
        for key in out:
            for j, v in (k.get(key) or {}).items():
                if j not in swept:
                    out[key].setdefault(j, float(v))
    if not out["velocity_rad_s"] and not out["acceleration_rad_s2"]:
        return None
    return {"source": ", ".join(sources), **out}


def import_group(cal: "Calibration", group: str, joints: Sequence[str],
                 other_joints: Optional[Sequence[str]] = None) -> dict:
    """Extracts from a saved calibration everything about a group: limits,
    measurements, coupled constraints involving only its joints, and poses
    (if present). Used to keep an already calibrated group without exploring it again.
    other_joints: joints of the OTHER calibrated groups (their velocities are not taken here)."""
    if is_simulated(cal):
        raise CalibrationError("Calibration simulée : ses valeurs ne peuvent pas être reprises")
    js = list(joints)
    missing = [j for j in js if j not in cal.limits]
    if missing:
        raise CalibrationError(f"{cal.version} ne calibre pas {missing}")
    poses = {n: {j: float(v) for j, v in (cal.poses or {}).get(n, {}).get(group, {}).items()}
             for n in ("rest", "start")}
    poses = {n: p for n, p in poses.items() if set(js) <= set(p)}
    # velocities/accelerations measured on the takes (extended calibration): those of the group,
    # plus those of joints outside the calibrated groups (hands), so they are kept
    kin = None
    if cal.kinematics:
        keep = lambda j: j in js or j not in (other_joints or ())
        kin = {"source": cal.kinematics.get("source", ""),
               **{k: {j: float(v) for j, v in (cal.kinematics.get(k) or {}).items() if keep(j)}
                  for k in ("velocity_rad_s", "acceleration_rad_s2")}}
    return {"source": cal.version, "file": cal.path.name if cal.path else None,
            "margin_rad": float(cal.margin_rad), "kinematics": kin,
            "limits": {j: [float(cal.limits[j][0]), float(cal.limits[j][1])] for j in js},
            "measured": {j: dict((cal.measured or {}).get(j) or {}) for j in js},
            "coupled": [c for c in (cal.coupled or []) if set(c.get("terms", {})) <= set(js)],
            "poses": poses}


# -- procedure on the robot ------------------------------------------------------------
class CalibrationSession:
    """Guided exploration group by group. No movement command:
    the explored group is relaxed, the others are stiffened in place."""

    def __init__(self, cfg, adapter, proc: dict):
        self.cfg, self.adapter, self.proc = cfg, adapter, proc
        self.rate = float(proc.get("rate_hz", 50))
        self.sweeps: Dict[str, dict] = {}
        self.poses: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.imported: Dict[str, dict] = {}   # groups taken from a saved calibration
        self._stop = threading.Event()
        self.current: Optional[str] = None
        self.n = 0
        self.live: List[np.ndarray] = []      # measurements of the current sweep (read by the interface)
        self.live_joints: List[str] = []

    def robot_info(self) -> dict:
        info = {}
        caps = self.adapter.capabilities()
        for k in ("robot_name", "naoqi_version", "robot", "backend", "adapter"):
            if caps.get(k) is not None:
                info[k] = caps[k]
        info["name"] = caps.get("robot_name") or caps.get("adapter")
        return info

    def factory_limits(self) -> Optional[Dict[str, List[float]]]:
        fn = getattr(self.adapter, "factory_limits", None)
        return fn(self.cfg.all_joints) if fn else None

    def request_stop(self):
        self._stop.set()

    def sweep(self, group: str, max_duration_s: float = 120.0, confirm: bool = False):
        if getattr(self.adapter, "is_real", False) and not confirm:
            raise CalibrationError(f"Confirmer que le groupe {group} est tenu avant de le relâcher")
        joints = self.cfg.groups[group]
        others = [j for j in self.cfg.all_joints if j not in joints]
        self.imported.pop(group, None)  # exploring a group replaces its import
        self.adapter.set_roles(guided=joints, driven=others)
        self._stop.clear()
        self.current = group
        a = self.adapter
        qs = []
        self.live, self.live_joints = qs, list(joints)
        t0 = a.now()
        k = 0
        try:
            while not self._stop.is_set() and a.now() - t0 < max_duration_s:
                a.heartbeat()
                qs.append(a.read_angles(joints))
                self.n = len(qs)
                k += 1
                a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        finally:
            # even if the link drops, the measurements already taken are kept
            self.current = None
            self._store_sweep(group, joints, qs)
        q = self.sweeps[group]["q"]
        return {j: (float(q[:, i].min()), float(q[:, i].max())) for i, j in enumerate(joints)} if len(q) else {}

    def _store_sweep(self, group, joints, qs):
        q = np.array(qs).reshape(-1, len(joints))
        prev = self.sweeps.get(group)
        if prev is not None and len(prev["q"]):  # a new exploration adds to the previous one
            q = np.vstack([np.asarray(prev["q"]), q])
        self.sweeps[group] = {"joints": joints, "q": q}

    def clear_sweep(self, group: str):
        self.sweeps.pop(group, None)

    def import_group(self, cal: "Calibration", group: str) -> dict:
        """Imports a group from a saved calibration (replaces its exploration; its
        poses too if the file has them, otherwise capture them)."""
        others = [j for j in self.cfg.all_joints if j not in self.cfg.groups[group]]
        imp = import_group(cal, group, self.cfg.groups[group], others)
        self.sweeps.pop(group, None)
        self.imported[group] = imp
        for n, p in imp["poses"].items():
            self.poses.setdefault(n, {})[group] = dict(p)
        return imp

    def drop_import(self, group: str):
        self.imported.pop(group, None)

    def capture_pose(self, name: str, group: str, samples: int = 5, interval_s: float = 0.02) -> Dict[str, float]:
        """Average of `samples` sensor readings (no command)."""
        js = self.cfg.groups[group]
        reads = []
        for k in range(max(1, int(samples))):
            if k:
                self.adapter.sleep(interval_s)
            reads.append(self.adapter.read_angles(js))
        q = np.mean(reads, axis=0)
        self.poses.setdefault(name, {})[group] = {j: float(v) for j, v in zip(js, q)}
        return self.poses[name][group]

    def result(self, version=None) -> Calibration:
        need = [(n, g) for n in ("rest", "start") for g in self.cfg.groups]
        missing = [f"{n}/{g}" for n, g in need if g not in self.poses.get(n, {})]
        if missing:
            raise CalibrationError(f"Poses à capturer : {missing}")
        return compute_calibration(self.sweeps, self.cfg.all_joints, self.proc, self.factory_limits(),
                                   self.poses, self.robot_info(), version, imported=self.imported)


def load_procedure(config_root) -> dict:
    p = Path(config_root) / "calibration.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d or {}


def sim_sweeper(cfg, group: str, duration_s: float = 20.0, rng=None):
    """Simulated exploration demonstrator: goes through the placeholder range
    (to test the procedure without a robot; it has no measurement value)."""
    rng = rng or np.random.default_rng(0)
    js = cfg.groups[group]
    freqs = rng.uniform(0.15, 0.6, size=len(js))
    phases = rng.uniform(0, 2 * np.pi, size=len(js))

    def fn(t):
        out = {}
        for j, f, ph in zip(js, freqs, phases):
            lo, hi = cfg.limits[j]
            c, r = (lo + hi) / 2, (hi - lo) / 2 * 0.97
            out[j] = float(c + r * np.sin(2 * np.pi * f * t + ph))
        return out
    return fn
