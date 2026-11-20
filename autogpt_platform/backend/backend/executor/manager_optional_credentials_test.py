"""#16193: a deleted optional credential must not leave its node unrunnable.

Drives the real save hook, the real queue-time `_enqueue_next_nodes` (with the
DB client mocked) and the real `execute_node` for an AnySearch Extract node
linked from an Agent Input, the shape from the issue's repro.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.io import AgentInputBlock
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.graph import Link, Node
from backend.executor.manager import _enqueue_next_nodes, execute_node
from backend.integrations.webhooks.graph_lifecycle_hooks import _before_graph_activate

DELETED_KEY = {
    "id": "deleted-cred",
    "provider": "anysearch",
    "type": "api_key",
    "title": "AnySearch key",
}
URL = "https://example.com"


def _graph(input_default: dict[str, Any]) -> tuple[Node, Node]:
    link = Link(
        source_id="input-node",
        sink_id="search-node",
        source_name="result",
        sink_name="url",
    )
    input_node = Node(
        id="input-node",
        block_id=AgentInputBlock().id,
        input_default={"name": "url"},
        output_links=[link],
    )
    search_node = Node(
        id="search-node",
        block_id=AnySearchExtractBlock().id,
        input_default=input_default,
        input_links=[link],
    )
    return input_node, search_node


async def _save_with_deleted_credential(nodes: list[Node]) -> None:
    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        await _before_graph_activate(MagicMock(nodes=nodes), "user-1")


async def _enqueue(input_node: Node, search_node: Node) -> list[NodeExecutionEntry]:
    @asynccontextmanager
    async def _no_lock(_key):
        yield

    db_client = MagicMock()
    db_client.get_node = AsyncMock(return_value=search_node)
    db_client.upsert_execution_input = AsyncMock(
        return_value=(MagicMock(node_exec_id="search-exec"), {"url": URL})
    )
    with (
        patch("backend.executor.manager.synchronized", _no_lock),
        patch("backend.executor.manager.send_async_execution_update", AsyncMock()),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            AsyncMock(),
        ),
    ):
        return await _enqueue_next_nodes(
            db_client=db_client,
            node=input_node,
            output=("result", URL),
            user_id="user-1",
            graph_exec_id="graph-exec-1",
            graph_id="graph-1",
            graph_version=1,
            log_metadata=MagicMock(),
            nodes_input_masks=None,
            execution_context=ExecutionContext(),
        )


async def _run(node: Node, entry: NodeExecutionEntry) -> list[tuple[str, Any]]:
    extract = AsyncMock(
        return_value={
            "code": 0,
            "message": "success",
            "data": {"url": URL, "title": "Example Domain", "content": "ok"},
        }
    )
    scope = MagicMock(_user=None, _tags={})
    with (
        patch.object(AnySearchExtractBlock, "_extract", extract),
        patch("backend.executor.manager._sentry_get_current_scope", return_value=scope),
    ):
        outputs = [
            item
            async for item in execute_node(
                node, entry, SimpleNamespace(creds_manager=MagicMock())
            )
        ]
    extract.assert_awaited_once_with(None, URL)
    return outputs


@pytest.mark.asyncio
async def test_anonymous_node_runs_after_save_with_deleted_credential():
    input_node, search_node = _graph({"auth": "anonymous", "credentials": DELETED_KEY})

    await _save_with_deleted_credential([input_node, search_node])
    assert "credentials" not in search_node.input_default

    entries = await _enqueue(input_node, search_node)
    assert len(entries) == 1

    outputs = await _run(search_node, entries[0])
    assert ("title", "Example Domain") in outputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_default",
    [{"auth": "anonymous", "credentials": {}}, {"credentials": {}}],
    ids=["anonymous", "default-auth"],
)
async def test_anonymous_node_with_stored_empty_credentials_runs(input_default):
    """Graphs saved before the fix already store `{}`; they run without a
    migration."""
    input_node, search_node = _graph(input_default)

    entries = await _enqueue(input_node, search_node)
    assert len(entries) == 1

    outputs = await _run(search_node, entries[0])
    assert ("title", "Example Domain") in outputs


@pytest.mark.asyncio
async def test_api_key_node_with_deleted_credential_fails_visibly():
    """In api_key mode the node is queued and fails with a clear error instead
    of being skipped without one."""
    input_node, search_node = _graph({"auth": "api_key", "credentials": DELETED_KEY})

    await _save_with_deleted_credential([input_node, search_node])
    entries = await _enqueue(input_node, search_node)
    assert len(entries) == 1

    scope = MagicMock(_user=None, _tags={})
    with patch(
        "backend.executor.manager._sentry_get_current_scope", return_value=scope
    ):
        with pytest.raises(ValueError, match="API key credentials are required"):
            async for _ in execute_node(
                search_node, entries[0], SimpleNamespace(creds_manager=MagicMock())
            ):
                pass
