import uuid
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.data.autopilot_migrate import (
    AUTOPILOT_BLOCK_ID,
    migrate_autopilot_transport,
)


@pytest.mark.asyncio
async def test_dry_run_counts_without_writing():
    with patch(
        "backend.data.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(return_value=[{"count": 3}]),
    ), patch(
        "backend.data.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(),
    ) as execute:
        assert await migrate_autopilot_transport(apply=False) == 3

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_writes_the_transport_value():
    with patch(
        "backend.data.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(),
    ) as query, patch(
        "backend.data.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(return_value=1),
    ) as execute:
        assert await migrate_autopilot_transport(apply=True) == 1

    query.assert_not_awaited()
    sql, transport, block_id = execute.await_args.args
    assert transport == "codex_app_server"
    assert block_id == AUTOPILOT_BLOCK_ID
    # Atomic single-key write, not a whole-blob replace: two booting pods must
    # not be able to clobber each other or a concurrent user edit.
    assert "jsonb_set" in sql
    assert "UPDATE" in sql


@pytest.mark.asyncio
async def test_apply_nothing_pending_is_a_silent_no_op():
    with patch(
        "backend.data.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(),
    ) as query, patch(
        "backend.data.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(return_value=0),
    ) as execute:
        assert await migrate_autopilot_transport(apply=True) == 0

    query.assert_not_awaited()
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_dry_run_nothing_pending_is_a_silent_no_op():
    with patch(
        "backend.data.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(return_value=[{"count": 0}]),
    ), patch(
        "backend.data.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(),
    ) as execute:
        assert await migrate_autopilot_transport(apply=False) == 0

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_predicate_scopes_to_autopilot_nodes_needing_the_backfill():
    """The predicate is the idempotency guarantee — it must exclude nodes that
    have a concrete transport, and id-less credential metas (nothing selected)."""
    query = AsyncMock(return_value=[{"count": 0}])
    with patch("backend.data.autopilot_migrate.query_raw_with_schema", new=query):
        await migrate_autopilot_transport(apply=False)

    sql, block_id = query.await_args.args
    assert block_id == AUTOPILOT_BLOCK_ID
    assert "\"constantInput\"->>'transport' IS NULL" in sql
    assert "'codex_credentials'->>'id' IS NOT NULL" in sql


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_is_idempotent_against_database(server):
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={
            "id": user_id,
            "email": f"autopilot-migrate-{user_id}@example.com",
            "name": "AutoPilot Migration Test",
        }
    )
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "autopilot-migrate-test",
            "description": "autopilot-migrate-test",
            "userId": user_id,
            "isActive": True,
        }
    )

    autopilot_block = await prisma.models.AgentBlock.prisma().find_unique(
        where={"id": AUTOPILOT_BLOCK_ID}
    )
    assert autopilot_block is not None
    other_blocks = await prisma.models.AgentBlock.prisma().find_many()
    other_block = next(
        block for block in other_blocks if block.id != AUTOPILOT_BLOCK_ID
    )

    async def seed(block_id: str, constant_input: dict) -> str:
        node = await prisma.models.AgentNode.prisma().create(
            data={
                "agentBlockId": block_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "constantInput": prisma.Json(constant_input),
            }
        )
        return node.id

    try:
        matching = await seed(
            AUTOPILOT_BLOCK_ID,
            {"codex_credentials": {"id": "cred-1"}},
        )
        already_migrated = await seed(
            AUTOPILOT_BLOCK_ID,
            {
                "transport": "platform",
                "codex_credentials": {"id": "cred-2"},
            },
        )
        null_transport = await seed(
            AUTOPILOT_BLOCK_ID,
            {
                "transport": None,
                "codex_credentials": {"id": "cred-null"},
            },
        )
        idless = await seed(
            AUTOPILOT_BLOCK_ID,
            {"codex_credentials": {"provider": "codex", "type": "oauth2"}},
        )
        other = await seed(
            other_block.id,
            {"codex_credentials": {"id": "cred-3"}},
        )

        assert await migrate_autopilot_transport(apply=True) == 2
        assert await migrate_autopilot_transport(apply=True) == 0

        async def constant_input(node_id: str) -> dict:
            node = await prisma.models.AgentNode.prisma().find_unique(
                where={"id": node_id}
            )
            assert node is not None
            return dict(node.constantInput or {})

        assert (await constant_input(matching))["transport"] == "codex_app_server"
        assert (await constant_input(null_transport))["transport"] == "codex_app_server"
        assert (await constant_input(already_migrated))["transport"] == "platform"
        assert "transport" not in await constant_input(idless)
        assert "transport" not in await constant_input(other)
    finally:
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
        await prisma.models.User.prisma().delete_many(where={"id": user_id})


def test_query_templates_survive_schema_formatting():
    """These templates are run through str.format() to inject the schema
    prefix, so any literal brace becomes a format field. A '{transport}'
    jsonb_set path raised KeyError at apply time — the second way this
    migration crashed only when actually run."""
    from backend.data.autopilot_migrate import _COUNT_QUERY, _UPDATE_QUERY

    for template in (_COUNT_QUERY, _UPDATE_QUERY):
        template.format(schema_prefix="platform.", schema="platform")
