"""What a ``review_required`` output carries besides the review itself.

The chat mounts a review's approval card from the output, and the card reads
the input from the review row, so the output only previews the input. It also
tells the model where the user finds the card: a model that cannot see the
card has sent people to look in another chat, or called again to raise it,
which only returns the same review.
"""

import json
from typing import Any

from backend.copilot.model import ChatSessionMetadata

_IN_THIS_CHAT = (
    "Its approval card is at the bottom of this chat and under Needs you on "
    "Home; if the user can't see it, a reload of this chat shows it. Never say "
    "it is in another chat, and don't repeat this call to raise it again."
)
_ON_A_CHAT_PLATFORM = (
    "No approval card is posted on this platform: the user approves it under "
    "Needs you on Home in AutoGPT, or in this conversation in the AutoGPT web "
    "app. Tell them that, and don't repeat this call to raise it again."
)
_IN_A_DELEGATED_THREAD = (
    "Its approval card is in this thread, which the user is not watching, and "
    "under Needs you on Home: say in your report that it waits for them there. "
    "Don't repeat this call to raise it again."
)

# Longest input string the output carries as is, and the most the whole input
# may take before the fields that don't fit are shown as their size.
_INPUT_STRING_CHARS = 500
_INPUT_PREVIEW_CHARS = 4_000


def review_card_location(metadata: ChatSessionMetadata | None) -> str:
    """Where the user finds a review's card raised in a session with *metadata*."""
    if metadata is None:
        return _IN_THIS_CHAT
    if metadata.source_platform:
        return _ON_A_CHAT_PLATFORM
    # A handoff makes the thread the user's own chat; a delegation does not.
    if metadata.delegated_by_session_id and not metadata.handed_off_from_expert_id:
        return _IN_A_DELEGATED_THREAD
    return _IN_THIS_CHAT


def preview_input(input_data: dict[str, Any]) -> dict[str, Any]:
    """*input_data* with each long string shown as its size, and, when that is
    still too long, each field that does not fit shown as its size: a field
    whose value (of any type) takes over ``_INPUT_STRING_CHARS``, or that would
    take the fields kept so far, keys included, past ``_INPUT_PREVIEW_CHARS``."""
    capped = {key: _sized_strings(value) for key, value in input_data.items()}
    if _chars(capped) <= _INPUT_PREVIEW_CHARS:
        return capped
    preview: dict[str, Any] = {}
    room = _INPUT_PREVIEW_CHARS - len("{}")
    for key, value in input_data.items():
        size = _chars(capped[key])
        # `"key": value, ` as json.dumps writes it.
        cost = _chars(key) + len(": ") + size + len(", ")
        fits = size <= _INPUT_STRING_CHARS and cost <= room
        preview[key] = capped[key] if fits else _size_of(value)
        room -= cost if fits else 0
    return preview


def _sized_strings(value: Any) -> Any:
    match value:
        case str() if len(value) > _INPUT_STRING_CHARS:
            return _size_of(value)
        case dict():
            return {key: _sized_strings(item) for key, item in value.items()}
        case list():
            return [_sized_strings(item) for item in value]
        case _:
            return value


def _size_of(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return f"<{len(text):,} chars>"


def _chars(value: Any) -> int:
    return len(json.dumps(value, default=str))
