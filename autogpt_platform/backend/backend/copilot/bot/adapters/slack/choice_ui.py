"""Block Kit builder + action-id codec for native Slack choice buttons.

Kept separate from ``adapter.py`` (already large) — the dispatch side (which
needs the adapter's client/context helpers) stays there; this module only
builds outbound blocks and parses the click payload's action_id.
"""

import re
from typing import Any

_ACTION_ID_RE = re.compile(r"^qans:([0-9a-f]{12}):(\d+)$")


def choice_blocks(text: str, token: str, options: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": option[:75]},
                    "action_id": _action_id(token, index),
                }
                for index, option in enumerate(options)
            ],
        },
    ]


def _action_id(token: str, index: int) -> str:
    return f"qans:{token}:{index}"


def parse_action_id(action_id: str) -> tuple[str, int] | None:
    match = _ACTION_ID_RE.match(action_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))
