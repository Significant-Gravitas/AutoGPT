"""Provenance of prompt rows the platform wrote on the user's behalf.

A scheduled turn that fires into an existing chat stores its prompt as an
assistant-role row (a user-role row would raise the confirm watermark that
expert proposals gate on), while the CLI session receives the same text as a
user turn. ``PromptOrigin`` marks such rows so anything that rebuilds model
context from DB rows can tell the prompt apart from a model reply.

The stamp rides in the row's ``metadata`` JSON column under
:data:`PROMPT_ORIGIN_METADATA_KEY`, but on ``ChatMessage`` it lives on a
field excluded from serialized payloads: the frontend treats any non-null
metadata on an assistant row as a reason to keep it as its own bubble, so
the key must never reach the API. :func:`split_prompt_origin` lifts it out
on the way in and :func:`with_prompt_origin` folds it back for the insert.

Being excluded, the field is also dropped by the Redis session cache and by
DatabaseManager RPC responses, so today it is readable only from the column
itself or from ``ChatMessage.from_db`` in a process with a direct Prisma
connection.
"""

import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

PROMPT_ORIGIN_METADATA_KEY = "prompt_origin"

PromptOriginKind = Literal["scheduled_followup", "scheduled_routine"]


class PromptOrigin(BaseModel):
    """Who wrote a prompt row, when, and from which schedule."""

    kind: PromptOriginKind
    schedule_id: str | None = None
    """The scheduler job id that fired the turn."""
    routine_id: str | None = None
    fired_at: datetime | None = None
    """When the scheduler dispatched the turn (UTC)."""


def with_prompt_origin(
    metadata: dict[str, Any] | None, origin: PromptOrigin | None
) -> dict[str, Any] | None:
    """*metadata* with *origin* stored under the reserved key.

    Returns *metadata* unchanged when there is no origin, and never mutates
    the caller's dict.
    """
    if origin is None:
        return metadata
    return {
        **(metadata or {}),
        PROMPT_ORIGIN_METADATA_KEY: origin.model_dump(mode="json"),
    }


def split_prompt_origin(
    metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, PromptOrigin | None]:
    """Separate the reserved key from the rest of a row's metadata.

    An otherwise empty bag comes back as ``None`` so a stamped row serializes
    exactly like an unstamped one. A malformed stamp is dropped rather than
    failing the session load.
    """
    if not metadata or PROMPT_ORIGIN_METADATA_KEY not in metadata:
        return metadata, None
    rest = {k: v for k, v in metadata.items() if k != PROMPT_ORIGIN_METADATA_KEY}
    try:
        origin = PromptOrigin.model_validate(metadata[PROMPT_ORIGIN_METADATA_KEY])
    except ValidationError:
        logger.warning("Dropping malformed prompt_origin stamp from message metadata")
        origin = None
    return rest or None, origin
