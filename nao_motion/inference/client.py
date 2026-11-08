"""Client d'inférence déportée : même interface `generate` que MotionEngine."""
from __future__ import annotations

import json
import urllib.request

from ..engine import Generated
from ..trajectory import Trajectory


class RemoteEngine:
    def __init__(self, cfg, url: str, timeout_s: float = 5.0):
        if not url:
            raise ValueError("inference.remote_url requis quand inference.location = remote")
        self.cfg, self.url, self.timeout = cfg, url.rstrip("/"), timeout_s

    def generate(self, emotion, intensity, event=None, head_current=None, variation=0.0, seed=None,
                 hands_current=None) -> Generated:
        body = json.dumps({"emotion": emotion, "intensity": intensity,
                           "event": event.to_dict() if event else None, "head_current": head_current,
                           "hands_current": hands_current,
                           "variation": variation, "seed": seed}).encode()
        req = urllib.request.Request(self.url + "/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        g = d["generated"]
        traj = Trajectory.from_dict(g["trajectory"])  # revalidée localement par le guardian
        with_hands = self.cfg.all_joints + self.cfg.hand_joints
        if set(traj.joints) not in (set(self.cfg.all_joints), set(with_hands)):
            raise ValueError("Trajectoire distante incompatible avec la configuration locale")
        has_hands = bool(self.cfg.hand_joints) and set(traj.joints) == set(with_hands)
        traj = traj.select(with_hands if has_hands else self.cfg.all_joints)
        arms, head = traj.select(self.cfg.arm_joints), traj.select(self.cfg.head_joints)
        hands = traj.select(self.cfg.hand_joints) if has_hands else None
        return Generated(arms, head, traj, None, None, traj.duration, g["inputs"],
                         g.get("notes", []) + [f"inférence distante : {self.url}"], hands)
