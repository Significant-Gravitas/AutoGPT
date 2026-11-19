"""PostToolUse hook for ``claude -p`` in a task eval run: asks the eval
process whether the turn must stop (a question was asked), as production's
hook does (``sdk/security_hooks.py``), and prints the hook's answer."""

import json
import os
import socket
import sys

event = json.load(sys.stdin)
with socket.create_connection(("127.0.0.1", int(os.environ["OTTO_EVAL_PORT"]))) as s:
    s.sendall(
        json.dumps({"kind": "post_tool", "tool": event.get("tool_name", "")}).encode()
        + b"\n"
    )
    reply = s.makefile().readline()
print(json.dumps(json.loads(reply)["output"]))
