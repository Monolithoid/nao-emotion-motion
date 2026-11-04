"""Calibration du guardian : procédure dédiée qui MESURE les limites sur le robot.

Le guardian ne s'appuie sur des limites « réelles » que si un fichier de
calibration est actif. Ce fichier est produit par une procédure à part
entière, robot par robot :

  1. identification du robot (nom, NAOqi, configuration, limites d'usine
     lues via ALMotion.getLimits quand le pont est connecté) ;
  2. exploration guidée, groupe par groupe : le groupe est relâché (lecture
     seule), les autres sont maintenus ; l'opérateur amène lentement chaque
     articulation jusqu'à ses butées utiles et parcourt les combinaisons
     (par ex. tête tournée ET baissée). Rien n'est commandé pendant cette étape ;
  3. capture des poses de repos et de départ (lecture des capteurs) ;
  4. calcul : limites = amplitude mesurée ∩ limites d'usine, moins une marge ;
     contraintes couplées = enveloppe convexe des couples explorés, simplifiée
     de façon conservatrice, moins la marge ;
  5. relecture par l'opérateur, puis enregistrement versionné et activation.

Limites de la méthode : l'enveloppe convexe suppose une zone atteignable
convexe pour chaque couple ; une zone en forme de L doit être découpée en
plusieurs couples/contraintes à la main. Les collisions entre membres (bras ↔
tête, bras ↔ bras) ne sont pas mesurables ainsi et restent des contraintes
manuelles dans guardian.yaml.
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


# -- géométrie -------------------------------------------------------------------
def convex_hull(points: np.ndarray) -> np.ndarray:
    """Enveloppe convexe 2D (chaîne monotone), sommets dans le sens trigonométrique."""
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
    """Retire les sommets de moindre aire (Visvalingam). Retirer un sommet d'un
    polygone convexe le rétrécit : la simplification reste conservatrice."""
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
    """Polygone convexe (sens trigo) -> demi-plans n·q <= b, rétrécis de `margin`."""
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
        nx, ny = d[1] / norm, -d[0] / norm  # normale extérieure
        out.append((float(nx), float(ny), float(nx * p[0] + ny * p[1] - margin)))
    return out


# -- fichier de calibration --------------------------------------------------------
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
    path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {"schema_version": 1, "kind": CAL_KIND, "version": self.version, "created_at": self.created_at,
                "robot": self.robot, "margin_rad": self.margin_rad, "unit": "rad", "notes": self.notes,
                "limits": {j: [float(a), float(b)] for j, (a, b) in self.limits.items()},
                "measured": self.measured, "poses": self.poses, "coupled_constraints": self.coupled}

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
                   created_at=str(d.get("created_at", "")), notes=d.get("notes", ""), path=path)


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


def activate(config_root, cal_path) -> Path:
    cal_path = Path(cal_path)
    Calibration.load(cal_path)  # valide avant activation
    d = Path(config_root) / "calibration"
    if cal_path.resolve().parent != d.resolve():
        raise CalibrationError("Le fichier doit se trouver dans configs/calibration/")
    (d / "ACTIVE").write_text(cal_path.name + "\n", encoding="utf-8")
    return d / "ACTIVE"


# -- calcul --------------------------------------------------------------------------
def compute_calibration(sweeps: Dict[str, Dict[str, np.ndarray]], joints: Sequence[str], proc: dict,
                        factory: Optional[Dict[str, Sequence[float]]] = None, poses=None, robot=None,
                        version: Optional[str] = None) -> Calibration:
    """sweeps : {groupe: {"joints": [...], "q": (N, J)}} mesurés pendant l'exploration."""
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
    missing = [j for j in joints if j not in samples]
    if missing:
        raise CalibrationError(f"Articulations non explorées : {missing}")
    limits, measured = {}, {}
    for j in joints:
        lo, hi = float(samples[j].min()), float(samples[j].max())
        if hi - lo < min_range:
            raise CalibrationError(f"{j} : amplitude explorée {hi - lo:.2f} rad < {min_range} rad. "
                                   "Explorer toute l'amplitude utile.")
        flo, fhi = (factory or {}).get(j, (lo, hi))[:2]
        a, b = max(lo, float(flo)) + margin, min(hi, float(fhi)) - margin
        if not a < b:
            raise CalibrationError(f"{j} : amplitude nulle après marge")
        limits[j] = (a, b)
        measured[j] = {"min": lo, "max": hi, "factory": [float(flo), float(fhi)] if factory and j in factory else None,
                       "n": int(samples[j].size)}
    coupled = []
    for a, b in proc.get("pairs", []):
        if a not in samples or b not in samples or len(samples[a]) != len(samples[b]):
            continue  # couple non exploré ensemble (groupes différents)
        pts = np.stack([samples[a], samples[b]], axis=1)
        hull = simplify_hull(convex_hull(pts), int(proc.get("max_hull_edges", 12)))
        for k, (na, nb, bound) in enumerate(hull_constraints(hull, margin)):
            coupled.append({"name": f"calib:{a}-{b}:{k}", "terms": {a: round(na, 6), b: round(nb, 6)},
                            "bound": round(bound, 6), "description": f"enveloppe mesurée {a}/{b}"})
    if poses:
        for name, groups in poses.items():
            for g, jq in groups.items():
                for j, v in jq.items():
                    lo, hi = limits[j]
                    if not lo <= v <= hi:
                        raise CalibrationError(f"Pose {name}/{j}={v:.3f} hors des limites calibrées [{lo:.3f}, {hi:.3f}]")
        for c in coupled:
            for name, groups in poses.items():
                flat = {j: v for jq in groups.values() for j, v in jq.items()}
                if all(j in flat for j in c["terms"]):
                    val = sum(k * flat[j] for j, k in c["terms"].items())
                    if val > c["bound"]:
                        raise CalibrationError(f"Pose {name} hors de l'enveloppe mesurée ({c['description']}) : "
                                               "explorer une zone plus large autour de cette pose")
    robot = robot or {}
    version = version or f"{robot.get('name') or 'nao'}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return Calibration(version=version, robot=robot, limits=limits, measured=measured, poses=poses or {},
                       coupled=coupled, margin_rad=margin)


# -- procédure sur le robot ------------------------------------------------------------
class CalibrationSession:
    """Exploration guidée groupe par groupe. Aucune commande de mouvement :
    le groupe exploré est relâché, les autres sont raidis sur place."""

    def __init__(self, cfg, adapter, proc: dict):
        self.cfg, self.adapter, self.proc = cfg, adapter, proc
        self.rate = float(proc.get("rate_hz", 50))
        self.sweeps: Dict[str, dict] = {}
        self.poses: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._stop = threading.Event()
        self.current: Optional[str] = None
        self.n = 0

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
        self.adapter.set_roles(guided=joints, driven=others)
        self._stop.clear()
        self.current = group
        a = self.adapter
        qs = []
        t0 = a.now()
        k = 0
        while not self._stop.is_set() and a.now() - t0 < max_duration_s:
            a.heartbeat()
            qs.append(a.read_angles(joints))
            self.n = len(qs)
            k += 1
            a.sleep(max(0.0, t0 + k / self.rate - a.now()))
        self.current = None
        self.sweeps[group] = {"joints": joints, "q": np.array(qs)}
        q = np.array(qs)
        return {j: (float(q[:, i].min()), float(q[:, i].max())) for i, j in enumerate(joints)} if len(qs) else {}

    def capture_pose(self, name: str, group: str) -> Dict[str, float]:
        js = self.cfg.groups[group]
        q = self.adapter.read_angles(js)
        self.poses.setdefault(name, {})[group] = {j: float(v) for j, v in zip(js, q)}
        return self.poses[name][group]

    def result(self, version=None) -> Calibration:
        need = [(n, g) for n in ("rest", "start") for g in self.cfg.groups]
        missing = [f"{n}/{g}" for n, g in need if g not in self.poses.get(n, {})]
        if missing:
            raise CalibrationError(f"Poses à capturer : {missing}")
        return compute_calibration(self.sweeps, self.cfg.all_joints, self.proc, self.factory_limits(),
                                   self.poses, self.robot_info(), version)


def load_procedure(config_root) -> dict:
    p = Path(config_root) / "calibration.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d or {}


def sim_sweeper(cfg, group: str, duration_s: float = 20.0, rng=None):
    """Démonstrateur simulé d'exploration : parcourt l'amplitude de remplacement
    (pour tester la procédure sans robot ; n'a aucune valeur de mesure)."""
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
