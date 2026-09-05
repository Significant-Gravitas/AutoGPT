from enum import Enum
from typing import Optional

from pydantic import SecretStr

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials

__all__ = [
    "CREDENTIALS_FIELD_DESCRIPTION",
    "SHARED_PATH",
    "TEST_CREDENTIALS",
    "TEST_CREDENTIALS_INPUT",
    "WORKSPACE_PATH",
    "WorkspaceScope",
    "agent_volume_name",
    "expert_volume_name",
    "user_volume_name",
    "volume_name_for_scope",
    "workspace_volume_mounts",
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="e2b",
    api_key=SecretStr("mock-e2b-api-key"),
    title="Mock E2B API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}

CREDENTIALS_FIELD_DESCRIPTION = (
    "Enter your API key for the E2B platform. "
    "You can get it here - https://e2b.dev/dashboard"
)


class WorkspaceScope(str, Enum):
    USER = "user"
    AGENT = "agent"


def user_volume_name(user_id: str) -> str:
    """Name of the per-user durable workspace volume.

    Shared across every E2B surface for the user — desktop blocks, the CoPilot
    on-demand desktop, and the CoPilot agent shell — so they all read/write the
    same persistent ``/home/user/workspace``.
    """
    return f"autogpt-user-{user_id}"


def agent_volume_name(graph_id: str) -> str:
    """Name of the per-agent (graph) durable workspace volume."""
    return f"autogpt-agent-{graph_id}"


def expert_volume_name(expert_id: str) -> str:
    """Name of a hired expert's own durable volume.

    This is the expert's home, not the user's: it persists for the life of the
    expert regardless of which chat, delegation or scheduled kickoff happens
    to be running as them, and it outlives the expert's sandbox if that is
    ever rebuilt.
    """
    return f"autogpt-expert-{expert_id}"


def volume_name_for_scope(
    scope: WorkspaceScope, execution_context: ExecutionContext
) -> Optional[str]:
    if scope == WorkspaceScope.USER and execution_context.user_id:
        return user_volume_name(execution_context.user_id)
    if scope == WorkspaceScope.AGENT and execution_context.graph_id:
        return agent_volume_name(execution_context.graph_id)
    return None


def workspace_volume_mounts(
    user_id: Optional[str], expert_id: Optional[str] = None
) -> dict[str, str]:
    """Mount map (``path -> volume name``) for a CoPilot shell or desktop.

    Plain session: the user's volume at ``WORKSPACE_PATH``.

    Expert session: the expert's own volume at ``WORKSPACE_PATH`` — the
    persistent home it may customise freely — plus the owning user's volume
    at ``SHARED_PATH``, so deliverables dropped there show up on the user's
    desktop and in every other session on the account. Without a user there
    is nothing to share, so only the expert's home is mounted.
    """
    mounts: dict[str, str] = {}
    if expert_id:
        mounts[WORKSPACE_PATH] = expert_volume_name(expert_id)
        if user_id:
            mounts[SHARED_PATH] = user_volume_name(user_id)
    elif user_id:
        mounts[WORKSPACE_PATH] = user_volume_name(user_id)
    return mounts
