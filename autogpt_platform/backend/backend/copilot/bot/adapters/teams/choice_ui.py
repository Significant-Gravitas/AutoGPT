"""Adaptive Card builder + submit-data codec for native Teams choice buttons.

Teams delivers an ``Action.Submit`` click as an ordinary inbound "message"
activity carrying ``value`` (not a separate invoke — that's the newer
Universal Actions schema this bot doesn't use), so the dispatch side stays a
small branch in ``adapter.py``'s existing message path; this module only
builds the outbound card and parses that ``value`` payload.
"""

from typing import Any

from backend.copilot.bot.choices import (
    BUTTON_KINDS,
    CARD_KIND,
    QUESTION_KIND,
    ButtonKind,
)


def choice_card(
    text: str, token: str, options: list[str], kind: ButtonKind = QUESTION_KIND
) -> dict[str, Any]:
    """A minimal Adaptive Card carrying one Action.Submit button per option.

    Pinned to schema 1.2, matching ``_link_card`` — the highest version every
    Teams client, including mobile, renders reliably.
    """
    actions: list[dict[str, Any]] = [
        {
            "type": "Action.Submit",
            "title": option[:60],
            "data": {"qans_token": token, "qans_index": index},
        }
        for index, option in enumerate(options)
    ]
    if kind == CARD_KIND and actions:
        for action in actions:
            action["data"]["qans_kind"] = kind
        actions[0]["style"] = "positive"
    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.2",
            "body": [{"type": "TextBlock", "text": text, "wrap": True}],
            "actions": actions,
        },
    }


def parse_choice_value(value: Any) -> tuple[ButtonKind, str, int] | None:
    """Extract (kind, token, index) from an Action.Submit activity's ``value``.

    A card sent before kinds existed carries none, and was a question.
    """
    if not isinstance(value, dict):
        return None
    token = value.get("qans_token")
    index = value.get("qans_index")
    kind = BUTTON_KINDS.get(str(value.get("qans_kind") or QUESTION_KIND))
    if not isinstance(token, str) or not token or kind is None:
        return None
    # bool is an int subclass in Python -- exclude it explicitly rather than
    # let a stray True/False masquerade as index 1/0.
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return kind, token, index
