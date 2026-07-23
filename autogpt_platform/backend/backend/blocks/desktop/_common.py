from enum import Enum
from typing import Optional

from pydantic import SecretStr

from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials

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


def volume_name_for_scope(
    scope: WorkspaceScope, execution_context: ExecutionContext
) -> Optional[str]:
    if scope == WorkspaceScope.USER and execution_context.user_id:
        return user_volume_name(execution_context.user_id)
    if scope == WorkspaceScope.AGENT and execution_context.graph_id:
        return agent_volume_name(execution_context.graph_id)
    return None
