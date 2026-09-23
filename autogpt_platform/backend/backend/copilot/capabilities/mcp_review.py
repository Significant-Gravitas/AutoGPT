"""Human review for MCP calls made through ``run_capability``.

With the auto-mode gate off, every call to a server that is not in the
official catalog pauses here: a tool's name is no evidence of what it does,
and its annotations are the server's own claim.  Catalog servers run without
a pause.  With the gate on, the gate decides every MCP call on the server's
effect map instead (``gate/subject.py``) and this review is not opened.

Records reuse the block review table with a synthetic node id so the
existing approval UI, the pending-review feed and ``resume_capability`` all
work unchanged.
"""

import uuid
from typing import Any

from pydantic import BaseModel

from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_SESSION_PREFIX,
    COPILOT_SYNTHETIC_ID_PREFIX,
)
from backend.data.db_accessors import review_db

COPILOT_MCP_NODE_PREFIX = f"{COPILOT_SYNTHETIC_ID_PREFIX}mcp-"

class MCPReviewPayload(BaseModel):
    """What ``resume_capability`` needs to replay an approved MCP call."""

    server_url: str
    tool: str
    arguments: dict[str, Any]


def needs_review(*, catalog_server: bool) -> bool:
    return not catalog_server


def is_mcp_review_id(review_id: str) -> bool:
    return review_id.startswith(COPILOT_MCP_NODE_PREFIX)


def mcp_review_node_id(host: str) -> str:
    return f"{COPILOT_MCP_NODE_PREFIX}{host}"


async def open_mcp_review(
    *,
    user_id: str,
    session_id: str,
    host: str,
    payload: MCPReviewPayload,
    organization_id: str | None,
    team_id: str | None,
) -> str:
    """Store a WAITING review for the call and return its review id.

    An identical pending call (same server, tool and arguments) reuses its
    review so a retried ``run_capability`` does not stack approvals.
    """
    graph_exec_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
    node_id = mcp_review_node_id(host)
    data = payload.model_dump()
    for review in await review_db().get_pending_reviews_for_execution(
        graph_exec_id, user_id
    ):
        if (
            review.node_id == node_id
            and review.status.value == "WAITING"
            and review.payload == data
        ):
            return review.node_exec_id
    review_id = f"{node_id}{COPILOT_NODE_EXEC_ID_SEPARATOR}{uuid.uuid4().hex[:8]}"
    await review_db().get_or_create_human_review(
        user_id=user_id,
        node_exec_id=review_id,
        graph_exec_id=graph_exec_id,
        graph_id=graph_exec_id,
        graph_version=1,
        input_data=data,
        message=(
            f"Run MCP tool '{payload.tool}' on {host} with the shown arguments? "
            "This server is not in the official catalog."
        ),
        editable=True,
        organization_id=organization_id,
        team_id=team_id,
    )
    return review_id
