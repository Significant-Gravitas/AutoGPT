"""Inline-keyboard builder + callback_data codec for native Telegram choice
buttons.

Kept separate from ``adapter.py`` (already large) — the dispatch side (which
needs the adapter's client/context helpers) stays there; this module only
builds the outbound keyboard and parses the click payload's callback_data.
"""

import re
from typing import Any

from backend.copilot.bot.choices import BUTTON_KINDS, QUESTION_KIND, ButtonKind

# callback_data caps at 64 bytes on the wire; "qans:" + 12 hex chars + ":" +
# a small index stays well under that.
_CALLBACK_RE = re.compile(r"^(qans|appr):([0-9a-f]{12}):(\d+)$")


def choice_keyboard(
    token: str, options: list[str], kind: ButtonKind = QUESTION_KIND
) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": option[:64], "callback_data": f"{kind}:{token}:{index}"}]
            for index, option in enumerate(options)
        ]
    }


def parse_callback_data(data: str) -> tuple[ButtonKind, str, int] | None:
    match = _CALLBACK_RE.match(data)
    if not match:
        return None
    return BUTTON_KINDS[match.group(1)], match.group(2), int(match.group(3))
