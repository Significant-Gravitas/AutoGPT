"""Adaptive Card builder + submit-data codec for native Teams choice buttons.

Teams delivers an ``Action.Submit`` click as an ordinary inbound "message"
activity carrying ``value`` (not a separate invoke — that's the newer
Universal Actions schema this bot doesn't use), so the dispatch side stays a
small branch in ``adapter.py``'s existing message path; this module only
builds the outbound card and parses that ``value`` payload.
"""

from typing import Any


def choice_card(text: str, token: str, options: list[str]) -> dict[str, Any]:
    """A minimal Adaptive Card carrying one Action.Submit button per option.

    Pinned to schema 1.2, matching ``_link_card`` — the highest version every
    Teams client, including mobile, renders reliably.
    """
    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.2",
            "body": [{"type": "TextBlock", "text": text, "wrap": True}],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": option[:60],
                    "data": {"qans_token": token, "qans_index": index},
                }
                for index, option in enumerate(options)
            ],
        },
    }


def parse_choice_value(value: Any) -> tuple[str, int] | None:
    """Extract (token, index) from an Action.Submit activity's ``value``."""
    if not isinstance(value, dict):
        return None
    token = value.get("qans_token")
    index = value.get("qans_index")
    if not isinstance(token, str) or not token:
        return None
    # bool is an int subclass in Python -- exclude it explicitly rather than
    # let a stray True/False masquerade as index 1/0.
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return token, index
