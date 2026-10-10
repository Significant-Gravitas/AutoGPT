"""A local stand-in for POST /v1/systemone in the documented response shape.
Answers every question in the request: a choice picks the second option and a
noul reads 0.91 when the state mentions curl, otherwise the first option and 0.4."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers.get("Authorization", "").startswith("Bearer ")
        assert set(body) == {"model", "state", "questions"}, set(body)
        assert 1 <= len(body["questions"]) <= 8
        refuse = "curl" in json.dumps(body["state"])
        p = 0.91 if refuse else 0.4
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "choice":
                words = list(q["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": words[1] if refuse else words[0],
                    "confidence": 0.8,
                    "probabilities": {words[0]: 1 - p, words[1]: p},
                }
            else:
                assert q["type"] == "noul", q
                answers[name] = {"type": "noul", "noul": p}
        out = {
            "model": body["model"],
            "answers": answers,
            "usage": {
                "input_tokens": len(json.dumps(body["state"])) // 4,
                "output_tokens": 0,
            },
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
