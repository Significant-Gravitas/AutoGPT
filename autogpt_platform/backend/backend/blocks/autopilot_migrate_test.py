from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma import Json

from backend.blocks.autopilot_migrate import (
    AUTOPILOT_BLOCK_ID,
    migrate_autopilot_transport,
)


def _node(node_id: str, constant_input: dict):
    node = MagicMock()
    node.id = node_id
    node.constantInput = constant_input
    return node


def _patched_prisma(nodes: list):
    client = MagicMock()
    client.find_many = AsyncMock(return_value=nodes)
    client.update = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_backfill_writes_json_not_a_raw_dict():
    """Regression: prisma rejects a plain dict for a Json column with
    "constantInput should be of any of the following types: JsonNullValueInput,
    Json". The first live run of this migration failed on exactly that."""
    client = _patched_prisma(
        [
            _node(
                "node-1",
                {"codex_credentials": {"id": "cred-1", "provider": "codex"}},
            )
        ]
    )

    with patch("backend.blocks.autopilot_migrate.AgentNode") as agent_node:
        agent_node.prisma.return_value = client
        assert await migrate_autopilot_transport(apply=True) == 1

    payload = client.update.await_args.kwargs["data"]["constantInput"]
    assert isinstance(payload, Json), f"raw {type(payload).__name__} would be rejected"


@pytest.mark.asyncio
async def test_dry_run_reports_without_writing():
    client = _patched_prisma([_node("node-1", {"codex_credentials": {"id": "cred-1"}})])

    with patch("backend.blocks.autopilot_migrate.AgentNode") as agent_node:
        agent_node.prisma.return_value = client
        assert await migrate_autopilot_transport(apply=False) == 1

    client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_migrated_and_credential_free_nodes_are_skipped():
    """Idempotent — it runs on every boot."""
    client = _patched_prisma(
        [
            _node("no-credential", {"prompt": "hi"}),
            _node("id-less-meta", {"codex_credentials": {"provider": "codex"}}),
            _node(
                "already-done",
                {
                    "transport": "codex_app_server",
                    "codex_credentials": {"id": "cred-1"},
                },
            ),
        ]
    )

    with patch("backend.blocks.autopilot_migrate.AgentNode") as agent_node:
        agent_node.prisma.return_value = client
        assert await migrate_autopilot_transport(apply=True) == 0

    client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_autopilot_nodes_are_considered():
    client = _patched_prisma([])

    with patch("backend.blocks.autopilot_migrate.AgentNode") as agent_node:
        agent_node.prisma.return_value = client
        await migrate_autopilot_transport(apply=True)

    assert client.find_many.await_args.kwargs["where"] == {
        "agentBlockId": AUTOPILOT_BLOCK_ID
    }
