"""Serveur d'inférence minimal (bibliothèque standard) pour une inférence déportée.

POST /generate {"emotion": "joie" | {"joie":0.5,...}, "intensity": 0.7,
                "event": {...optionnel...}, "head_current": {...optionnel...}}
-> trajectoire + rapport du guardian. Le serveur n'exécute RIEN : le client
(contrôleur) revalide et exécute après confirmation.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..events import EventContext


def make_handler(wb):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "inconnu"})
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                ev = EventContext.from_dict(d["event"]) if d.get("event") else EventContext.none()
                gen = wb.get_engine().generate(d["emotion"], float(d["intensity"]), ev, d.get("head_current"),
                                               float(d.get("variation", 0)), d.get("seed"))
                self._send(200, {"generated": gen.to_dict(),
                                 "guardian": wb.guardian.validate(gen.combined).to_dict()})
            except Exception as e:
                self._send(400, {"error": str(e)})
    return H


def serve(wb, host="127.0.0.1", port=8766):
    print(f"Inférence sur http://{host}:{port}/generate (aucune exécution)")
    ThreadingHTTPServer((host, port), make_handler(wb)).serve_forever()
