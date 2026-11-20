"""A node whose credential needs reconnecting fails as a user error.

The provider's refusal was reported once when it happened; logging every
later run of a scheduled graph as an unexpected failure would send Sentry an
event per run until the user reconnects.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import NodeExecutionStats
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.oauth.refresh_failure import (
    CredentialsNeedReconnectError,
    ReconnectRequired,
)
from backend.util.request import HTTPClientError


def _node_exec() -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(root_execution_id="graph-exec-1"),
    )


async def _run_failing_with(error: Exception) -> tuple[ExecutionStatus, MagicMock]:
    async def failing_node(**kwargs):
        raise error
        yield  # an async generator, like execute_node

    log_metadata = MagicMock()
    db_client = MagicMock(
        upsert_execution_output=AsyncMock(),
        get_node_execution=AsyncMock(return_value=None),
    )
    with (
        patch(
            "backend.data.graph.get_block",
            return_value=SimpleNamespace(execution_timeout_seconds=None),
        ),
        patch("backend.executor.manager.execute_node", new=failing_node),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
    ):
        _, status = await ExecutionProcessor()._on_node_execution(
            node=Node(id="node-1", block_id="block-1"),
            node_exec=_node_exec(),
            node_exec_progress=NodeExecutionProgress(),
            stats=NodeExecutionStats(),
            db_client=db_client,
            log_metadata=log_metadata,
        )
    return status, log_metadata


@pytest.mark.asyncio
async def test_a_credential_needing_reconnect_is_an_expected_failure():
    error = CredentialsNeedReconnectError(
        "twitter",
        "cred-1",
        ReconnectRequired(error_code="invalid_request", status_code=400),
    )

    status, log_metadata = await _run_failing_with(error)

    assert status == ExecutionStatus.FAILED
    log_metadata.exception.assert_not_called()
    [expected] = [
        c.args[0]
        for c in log_metadata.info.call_args_list
        if "Expected failure on node execution" in c.args[0]
    ]
    assert "reconnect the account" in expected


@pytest.mark.asyncio
async def test_any_other_provider_refusal_is_still_unexpected():
    status, log_metadata = await _run_failing_with(HTTPClientError("nope", 400))

    assert status == ExecutionStatus.FAILED
    log_metadata.exception.assert_called_once()
