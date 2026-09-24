"""A local stand-in for POST /v1/systemone in the documented response shape.
Says ask (noul 0.91) when the state mentions curl, otherwise allow (noul 0.4)."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers.get("Authorization", "").startswith("Bearer ")
        assert set(body) == {"model", "state", "questions"}, set(body)
        q = body["questions"]
        assert q["verdict"]["type"] == "choice" and set(q["verdict"]["criteria"]) in (
            {"allow", "ask"},
            {"clean", "hold"},
        )
        assert q["refuse"]["type"] == "noul"
        refuse = "curl" in body["state"]
        words = list(q["verdict"]["criteria"])
        choice = words[1] if refuse else words[0]
        p = 0.91 if refuse else 0.4
        out = {
            "model": body["model"],
            "answers": {
                "verdict": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.8,
                    "probabilities": {words[0]: 1 - p, words[1]: p},
                },
                "refuse": {"type": "noul", "noul": p},
            },
            "usage": {"input_tokens": len(body["state"]) // 4, "output_tokens": 0},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
