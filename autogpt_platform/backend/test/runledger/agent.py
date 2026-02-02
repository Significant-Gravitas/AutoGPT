from __future__ import annotations

import json
import sys


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("type") == "task_start":
            send(
                {
                    "type": "tool_call",
                    "name": "format_docs_text",
                    "call_id": "c1",
                    "args": msg.get("input", {}),
                }
            )
        elif msg.get("type") == "tool_result":
            result = msg.get("result") or {}
            output = {
                "success": bool(result.get("result", {}).get("success", False)),
                "rgb": result.get("rgb"),
            }
            send({"type": "final_output", "output": output})
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
