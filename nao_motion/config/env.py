"""Lecture de `.env` (sans dépendance) et surcharge de robot.yaml.

Ordre de priorité : variables d'environnement > fichier .env > robot.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# variable -> (chemin dans robot.yaml, conversion)
MAPPING = {
    "NAO_ADAPTER": (("adapter",), str),
    "NAO_IP": (("robot", "ip"), str),
    "NAO_PORT": (("robot", "port"), int),
    "NAO_NAME": (("robot", "name"), str),
    "NAO_MODEL": (("robot", "model"), str),
    "NAO_HEAD_VERSION": (("robot", "head_version"), str),
    "NAO_BODY_VERSION": (("robot", "body_version"), str),
    "NAOQI_VERSION": (("robot", "naoqi_version"), str),
    "NAO_PYTHON": (("robot", "python"), str),
    "NAO_BRIDGE_LOCATION": (("bridge", "location"), str),
    "NAO_BRIDGE_HOST": (("bridge", "host"), str),
    "NAO_BRIDGE_PORT": (("bridge", "port"), int),
    "NAO_BRIDGE_WATCHDOG_S": (("bridge", "watchdog_s"), float),
}


def parse_env_file(path) -> Dict[str, str]:
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def apply_env(robot: dict, env_file: Optional[Path] = None, environ=None) -> dict:
    """Retourne une copie de robot.yaml surchargée ; robot['_env_sources'] liste ce qui vient du .env."""
    environ = os.environ if environ is None else environ
    values = parse_env_file(env_file) if env_file else {}
    values.update({k: v for k, v in environ.items() if k in MAPPING})
    r = {k: (dict(v) if isinstance(v, dict) else v) for k, v in robot.items()}
    applied = []
    for key, (path, conv) in MAPPING.items():
        if key not in values or values[key] == "":
            continue
        try:
            val = conv(values[key])
        except ValueError as e:
            raise ValueError(f"{key}={values[key]!r} invalide") from e
        d = r
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = val
        applied.append(key)
    r["_env_sources"] = applied
    return r
