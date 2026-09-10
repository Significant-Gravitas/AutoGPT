"""Server-owned identity and metadata for an expert's first kickoff turn."""

import uuid
from collections.abc import Mapping
from typing import Any

EXPERT_KICKOFF_KIND = "expert_kickoff"

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
