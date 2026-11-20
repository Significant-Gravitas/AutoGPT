"""Server-owned identity and metadata for an expert's first kickoff turn."""

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.copilot.model import ChatMessage, ChatSession

EXPERT_KICKOFF_KIND = "expert_kickoff"

# The kickoff the server sends for a hire no browser follows
# (``experts/kickoff.py``). Word for word the frontend's KICKOFF_PROMPT
# (``copilot/expertKickoff.ts``), so a hire greets the same way whichever side
# sends it; change both together. It stays ask-first: one click on Hire once ran
# a Gmail send (SECRT-2622).
EXPERT_KICKOFF_PROMPT = (
    "You were just hired. Call expert_onboarding once, and nothing else, this "
    "turn: a one-sentence greeting introducing yourself in your voice, with "
    "no dashes, then "
    "2-3 short questions the user can answer by tapping. Every question and "
    "every option must come from your own role and area of expertise and from "
    "the workflows installed on you — ask which outcome in your area to start "
    "with, and which of the services your work depends on you should be "
    "connected to, naming the real providers rather than asking in the "
    "abstract. Do not ask about work outside your role, whatever else you "
    "know about the user or their team. Do not run a workflow, create a "
    "schedule, or start any other work before the answers come back."
)

_MESSAGE_NAMESPACE = uuid.UUID("d0890fe0-26eb-5ee4-91b4-42a930ce29ac")
_CLIENT_MESSAGE_NAMESPACE = uuid.UUID("d70473da-722b-55cb-963f-4729e64d7b06")


def scoped_client_message_id(
    user_id: str,
    session_id: str,
    client_message_id: str,
) -> str:
    """Scope a client idempotency key to its authenticated session owner.

    Clients can choose ``client_message_id``. They cannot use it to claim an
    arbitrary global ``ChatMessage`` primary key belonging to another user.
    """
    return str(
        uuid.uuid5(
            _CLIENT_MESSAGE_NAMESPACE,
            f"{user_id}:{session_id}:{client_message_id}",
        )
    )


def expert_kickoff_message_id(
    user_id: str,
    session_id: str,
    expert_id: str,
) -> str:
    """Return the canonical message PK for one expert session's kickoff."""
    return str(
        uuid.uuid5(
            _MESSAGE_NAMESPACE,
            f"{user_id}:{session_id}:{expert_id}",
        )
    )


def expert_kickoff_metadata(expert_id: str) -> dict[str, Any]:
    """Metadata that keeps the persisted control turn off user surfaces."""
    return {
        "hidden": True,
        "kind": EXPERT_KICKOFF_KIND,
        "expert_id": expert_id,
    }


def is_hidden_chat_message(metadata: Mapping[str, Any] | None) -> bool:
    """Whether message metadata marks a row as hidden from user surfaces."""
    return metadata is not None and metadata.get("hidden") is True


def is_expert_kickoff_message(message: "ChatMessage") -> bool:
    """Whether a persisted user row is the server-written kickoff message.

    ``expert_kickoff_metadata`` is the only writer of this ``kind`` and the
    chat route never copies request fields into message metadata, so a true
    here means the server wrote the row itself.
    """
    if message.role != "user":
        return False
    metadata = message.metadata
    return metadata is not None and metadata.get("kind") == EXPERT_KICKOFF_KIND


def is_expert_kickoff_turn(session: "ChatSession") -> bool:
    """Whether the turn being served answers the hire's kickoff message.

    Nobody has asked for anything yet on that turn, which is what makes it
    safe for callers to narrow it down to the onboarding card.
    """
    for message in reversed(session.messages):
        if message.role == "user":
            return is_expert_kickoff_message(message)
    return False
