"""Which account a stored AutoPilot node bills once its ChatGPT connection is
deleted: run start validation, then the executor's own run of the node, with
only the session and the turn stubbed."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.autopilot import AutoPilotBlock
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.graph import Node
from backend.executor.manager import execute_node
from backend.executor.utils import validate_exec
from backend.util.exceptions import BlockExecutionError


async def _execute_stored_node(input_default: dict) -> tuple[str, AsyncMock]:
    """Returns the error the node failed with ("" if it ran) and the session
    factory, whose arguments name the account the run bills."""
    node = Node(id="node-1", block_id=AutoPilotBlock().id, input_default=input_default)
    inputs, message = validate_exec(node, {})
    assert inputs is not None, message
    entry = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id=node.id,
        block_id=node.block_id,
        inputs=inputs,
        execution_context=ExecutionContext(
            user_id="user-1",
            graph_id="graph-1",
            graph_exec_id="graph-exec-1",
        ),
    )
    create_session = AsyncMock(return_value="session-1")
    execute_copilot = AsyncMock(
        return_value=(
            "done",
            [],
            "[]",
            "session-1",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    scope = MagicMock(_user=None, _tags={})
    with (
        patch.object(AutoPilotBlock, "create_session", create_session),
        patch.object(AutoPilotBlock, "execute_copilot", execute_copilot),
        patch("backend.executor.manager._sentry_get_current_scope", return_value=scope),
    ):
        try:
            async for _ in execute_node(
                node, entry, SimpleNamespace(creds_manager=MagicMock())
            ):
                pass
        except BlockExecutionError as e:
            return str(e), create_session
    return "", create_session


@pytest.mark.asyncio
async def test_a_legacy_node_whose_connection_was_deleted_is_not_billed_to_platform():
    """Deleting the ChatGPT connection of a node saved without a transport used
    to store `codex_credentials: {}`. Validation now reads that as "not
    configured", which on its own would run the node on platform credits, an
    account its owner never chose."""
    error, create_session = await _execute_stored_node(
        {"prompt": "do the thing", "codex_credentials": {}}
    )

    assert "needs a connected ChatGPT account" in error
    create_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_default",
    [
        {"prompt": "do the thing"},
        {"prompt": "do the thing", "transport": "platform", "codex_credentials": {}},
    ],
    ids=["never-connected", "chose-platform"],
)
async def test_a_node_that_never_chose_chatgpt_runs_on_platform_credits(
    input_default: dict,
):
    error, create_session = await _execute_stored_node(input_default)

    assert error == ""
    assert create_session.await_args.kwargs["llm_auth_provider"] == "platform"
