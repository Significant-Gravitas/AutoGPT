"""Chat rows the platform posts for the user, which the model never reads back.

A scheduled chat that was skipped or paused gets a card in its chat saying
why, with the fix on a button. Posted as an assistant row, the next turn
would read the card as something the model had said itself, and a stale line
like "Discord still needs to be connected" in a chat's history is what kept an
every-minute monitor repeating it, run after run, without trying again. So a
row whose metadata carries ``hidden_from_model`` is dropped from every list of
messages that becomes model input, the tool that lets an expert read another
chat included. Drop them after any slicing by count or sequence: the
transcript watermark counts every row, hidden ones included. They are the
owner's, so a shared copy of the chat leaves them out as well.
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.copilot.model import ChatMessage

HIDDEN_FROM_MODEL_KEY = "hidden_from_model"


def is_hidden_from_model(metadata: Mapping[str, Any] | None) -> bool:
    return metadata is not None and metadata.get(HIDDEN_FROM_MODEL_KEY) is True


def visible_to_model(messages: "Iterable[ChatMessage]") -> "list[ChatMessage]":
    return [m for m in messages if not is_hidden_from_model(m.metadata)]
