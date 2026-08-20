from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.autopilot_migrate import (
    AUTOPILOT_BLOCK_ID,
    migrate_autopilot_transport,
)


@pytest.mark.asyncio
async def test_dry_run_counts_without_writing():
    with patch(
        "backend.blocks.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(return_value=[{"count": 3}]),
    ), patch(
        "backend.blocks.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(),
    ) as execute:
        assert await migrate_autopilot_transport(apply=False) == 3

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_writes_the_transport_value():
    with patch(
        "backend.blocks.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(),
    ) as query, patch(
        "backend.blocks.autopilot_migrate.execute_raw_with_schema",
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
        "backend.blocks.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(),
    ) as query, patch(
        "backend.blocks.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(return_value=0),
    ) as execute:
        assert await migrate_autopilot_transport(apply=True) == 0

    query.assert_not_awaited()
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_dry_run_nothing_pending_is_a_silent_no_op():
    with patch(
        "backend.blocks.autopilot_migrate.query_raw_with_schema",
        new=AsyncMock(return_value=[{"count": 0}]),
    ), patch(
        "backend.blocks.autopilot_migrate.execute_raw_with_schema",
        new=AsyncMock(),
    ) as execute:
        assert await migrate_autopilot_transport(apply=False) == 0

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_predicate_scopes_to_autopilot_nodes_needing_the_backfill():
    """The predicate is the idempotency guarantee — it must exclude nodes that
    already have a transport, and id-less credential metas (nothing selected)."""
    query = AsyncMock(return_value=[{"count": 0}])
    with patch("backend.blocks.autopilot_migrate.query_raw_with_schema", new=query):
        await migrate_autopilot_transport(apply=False)

    sql, block_id = query.await_args.args
    assert block_id == AUTOPILOT_BLOCK_ID
    assert "NOT (\"constantInput\" ? 'transport')" in sql
    assert "'codex_credentials'->>'id' IS NOT NULL" in sql


def test_query_templates_survive_schema_formatting():
    """These templates are run through str.format() to inject the schema
    prefix, so any literal brace becomes a format field. A '{transport}'
    jsonb_set path raised KeyError at apply time — the second way this
    migration crashed only when actually run."""
    from backend.blocks.autopilot_migrate import _COUNT_QUERY, _UPDATE_QUERY

    for template in (_COUNT_QUERY, _UPDATE_QUERY):
        template.format(schema_prefix="platform.", schema="platform")
