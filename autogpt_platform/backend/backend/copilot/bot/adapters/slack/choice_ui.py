"""Block Kit builder + action-id codec for native Slack choice buttons.

Kept separate from ``adapter.py`` (already large) — the dispatch side (which
needs the adapter's client/context helpers) stays there; this module only
builds outbound blocks and parses the click payload's action_id.
"""

import re
from typing import Any

from backend.copilot.bot.choices import (
    BUTTON_KINDS,
    CARD_KIND,
    QUESTION_KIND,
    ButtonKind,
)

_ACTION_ID_RE = re.compile(r"^(qans|appr):([0-9a-f]{12}):(\d+)$")


def choice_blocks(
    text: str, token: str, options: list[str], kind: ButtonKind = QUESTION_KIND
) -> list[dict[str, Any]]:
    buttons: list[dict[str, Any]] = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": option[:75]},
            "action_id": _action_id(kind, token, index),
        }
        for index, option in enumerate(options)
    ]
    if kind == CARD_KIND and buttons:
        buttons[0]["style"] = "primary"
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "actions", "elements": buttons},
    ]


def _action_id(kind: ButtonKind, token: str, index: int) -> str:
    return f"{kind}:{token}:{index}"


def parse_action_id(action_id: str) -> tuple[ButtonKind, str, int] | None:
    match = _ACTION_ID_RE.match(action_id)
    if not match:
        return None
    return BUTTON_KINDS[match.group(1)], match.group(2), int(match.group(3))
