from __future__ import annotations

import uuid
from flask import Flask, request, jsonify

from . import ai


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/message", methods=["POST"])
    def message() -> tuple[dict[str, str], int]:
        sid = request.args.get("sid") or request.json.get("sid") or str(uuid.uuid4())
        text = request.json.get("text", "")
        ai.handle_user(sid, text)
        return jsonify({"sid": sid}), 202

    @app.get("/state")
    def state() -> tuple[dict[str, str], int]:
        sid = request.args["sid"]
        dq = ai.async_messages.get(sid)
        text = dq.popleft() if dq else "…"
        return {"text": text}, 200

    @app.route("/mode", methods=["POST"])
    def mode() -> dict[str, str]:
        mode = request.json.get("mode", "affection")
        ai.latest_thought["mode"] = f"Mode switched to {mode}"
        return jsonify({"status": "ok"})

    return app
