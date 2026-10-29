"""Real-DB tests for the model retirement CLI/library.

Seeds minimal User/AgentGraph/AgentNode rows, exercises retire → revert
against the actual `{schema_prefix}` SQL, and verifies the catalog-based
validation and the active-migration partial unique index.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import prisma.models
import pytest
from prisma.errors import UniqueViolationError

import backend.data.llm_registry.registry as reg
from backend.data.llm_registry.registry import get_all_models, load_catalog
from backend.data.llm_registry.retire import (
    count_model_usage,
    list_model_migrations,
    retire_model,
    revert_model_migration,
)

# Prisma's client is bound to the session-scoped server fixture's loop.
pytestmark = pytest.mark.asyncio(loop_scope="session")

_TEST_USER_ID = "5e53486c-cf57-477e-ba2a-cb02dc82ret1"


@pytest.fixture(autouse=True)
async def retirement_env(server):
    """Load the catalog, clean records, and tear down created graph rows."""
    load_catalog()
    await prisma.models.LlmModelMigration.prisma().delete_many()
    created_graph_ids: list[str] = []

    try:
        await prisma.models.User.prisma().create(
            data={
                "id": _TEST_USER_ID,
                "email": f"retire-{_TEST_USER_ID}@example.com",
                "name": "Retire Test",
            }
        )
    except UniqueViolationError:
        pass

    yield created_graph_ids

    await prisma.models.LlmModelMigration.prisma().delete_many()
    for graph_id in created_graph_ids:
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})


async def _seed_node(created_graph_ids: list[str], model_value: str) -> str:
    """Create a graph + one node whose constantInput uses *model_value*."""
    block = await prisma.models.AgentBlock.prisma().find_first()
    assert block is not None, "blocks must be initialized by the server fixture"
    graph_id = str(uuid.uuid4())
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "retire-test",
            "description": "retire-test",
            "userId": _TEST_USER_ID,
            "isActive": True,
        }
    )
    created_graph_ids.append(graph_id)
    node = await prisma.models.AgentNode.prisma().create(
        data={
            "agentBlockId": block.id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "constantInput": prisma.Json({"model": model_value}),
        }
    )
    return node.id


def _two_catalog_slugs() -> tuple[str, str]:
    models = [m for m in get_all_models() if m.is_enabled]
    assert len(models) >= 2
    return models[0].slug, models[1].slug


async def _node_model(node_id: str) -> str:
    node = await prisma.models.AgentNode.prisma().find_unique(where={"id": node_id})
    assert node is not None
    constant_input = dict(node.constantInput or {})
    return str(constant_input.get("model"))


def _node_value(slug: str) -> str:
    """AgentNodes store FULL enum values — identity with catalog slugs."""
    return slug


async def test_usage_counts_seeded_nodes(retirement_env):
    source, _ = _two_catalog_slugs()
    await _seed_node(retirement_env, _node_value(source))

    assert await count_model_usage(source) == 1


async def test_retire_rewrites_nodes_and_records(retirement_env):
    source, replacement = _two_catalog_slugs()
    node_id = await _seed_node(retirement_env, _node_value(source))

    result = await retire_model(source, replacement, reason="test retirement")

    assert result.nodes_migrated == 1
    assert result.migration_id is not None
    assert await _node_model(node_id) == _node_value(replacement)

    record = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": result.migration_id}
    )
    assert record is not None
    assert record.nodeCount == 1
    assert node_id in list(record.migratedNodeIds or [])
    assert record.reason == "test retirement"


async def test_revert_restores_nodes(retirement_env):
    source, replacement = _two_catalog_slugs()
    node_id = await _seed_node(retirement_env, _node_value(source))
    retired = await retire_model(source, replacement)
    assert retired.migration_id is not None

    reverted = await revert_model_migration(retired.migration_id)

    assert reverted.nodes_reverted == 1
    assert reverted.nodes_already_changed == 0
    assert await _node_model(node_id) == _node_value(source)
    record = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": retired.migration_id}
    )
    assert record is not None
    assert record.isReverted is True
    assert record.revertedAt is not None
    assert await list_model_migrations() == []  # active-only view is empty


async def test_unknown_replacement_rejected_without_writes(retirement_env):
    source, _ = _two_catalog_slugs()
    node_id = await _seed_node(retirement_env, _node_value(source))

    with pytest.raises(ValueError, match="not in the catalog"):
        await retire_model(source, "no/such-model")

    assert await _node_model(node_id) == _node_value(source)
    assert await prisma.models.LlmModelMigration.prisma().count() == 0


async def test_disabled_replacement_rejected(retirement_env):
    source, replacement = _two_catalog_slugs()
    await _seed_node(retirement_env, _node_value(source))

    original = reg._dynamic_models[replacement]
    reg._dynamic_models[replacement] = original.model_copy(update={"is_enabled": False})
    try:
        with pytest.raises(ValueError, match="disabled in the catalog"):
            await retire_model(source, replacement)
    finally:
        reg._dynamic_models[replacement] = original


async def test_second_active_migration_rejected(retirement_env):
    source, replacement = _two_catalog_slugs()
    await _seed_node(retirement_env, _node_value(source))
    first = await retire_model(source, replacement)
    assert first.migration_id is not None

    # New node appears on the retired model (e.g. imported graph) — a second
    # retire must be rejected while the first migration is still active.
    await _seed_node(retirement_env, _node_value(source))
    with pytest.raises(ValueError, match="active migration"):
        await retire_model(source, replacement)


async def test_retire_with_no_referencing_nodes_records_nothing(retirement_env):
    source, replacement = _two_catalog_slugs()

    result = await retire_model(source, replacement)

    assert result.nodes_migrated == 0
    assert result.migration_id is None
    assert await prisma.models.LlmModelMigration.prisma().count() == 0


async def test_prefixed_model_nodes_are_found_and_rewritten(retirement_env):
    """Provider-prefixed enum values (e.g. moonshotai/kimi-k2.5) are stored
    verbatim on nodes; an earlier revision stripped the prefix and silently
    no-opped on exactly these models."""
    source = "moonshotai/kimi-k2.5"
    replacement = "moonshotai/kimi-k2.6"
    node_id = await _seed_node(retirement_env, source)

    assert await count_model_usage(source) == 1
    result = await retire_model(source, replacement, reason="prefix regression")
    assert result.nodes_migrated == 1
    assert await _node_model(node_id) == replacement

    revert = await revert_model_migration(result.migration_id)
    assert revert.nodes_reverted == 1
    assert await _node_model(node_id) == source


@pytest.mark.asyncio(loop_scope="session")
async def test_cli_dry_run_writes_nothing(retirement_env, mocker):
    """The destructive CLI defaults to dry-run: without --yes it must
    report and exit 1 having written no node changes and no record."""
    from backend.data.llm_registry.retire import _build_parser, _run_cli

    source, replacement = _two_catalog_slugs()
    node_id = await _seed_node(retirement_env, _node_value(source))

    mocker.patch("backend.data.db.connect", new=AsyncMock())
    args = _build_parser().parse_args([source, "--replacement", replacement])
    assert await _run_cli(args) == 1

    assert await _node_model(node_id) == _node_value(source)
    assert await prisma.models.LlmModelMigration.prisma().count() == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_revert_leaves_manually_repointed_nodes_alone(retirement_env):
    """The documented revert guard: nodes a user manually repointed at a
    third model since the migration are NOT touched by the revert, and the
    result reports them as already-changed."""
    source, replacement = _two_catalog_slugs()
    node_a = await _seed_node(retirement_env, _node_value(source))
    node_b = await _seed_node(retirement_env, _node_value(source))

    result = await retire_model(source, replacement)
    assert result.nodes_migrated == 2

    third = "user/hand-picked-model"
    node = await prisma.models.AgentNode.prisma().find_unique(where={"id": node_b})
    assert node is not None
    assert isinstance(node.constantInput, dict)
    ci = dict(node.constantInput)
    ci["model"] = third
    await prisma.models.AgentNode.prisma().update(
        where={"id": node_b}, data={"constantInput": prisma.Json(ci)}
    )

    migrations = await list_model_migrations()
    revert = await revert_model_migration(migrations[0].id)

    assert revert.nodes_reverted == 1
    assert revert.nodes_already_changed == 1
    assert await _node_model(node_a) == _node_value(source)
    assert await _node_model(node_b) == third


@pytest.mark.asyncio(loop_scope="session")
async def test_cli_replacement_defaults_from_catalog_fallback(retirement_env, mocker):
    """--replacement pre-fills from the retired model's fallback_model_slug."""
    from backend.data.llm_registry import retire as retire_mod
    from backend.data.llm_registry.retire import _build_parser, _run_cli

    source, replacement = _two_catalog_slugs()
    node_id = await _seed_node(retirement_env, _node_value(source))

    payload = retire_mod.get_catalog()
    patched = payload.model_copy(
        update={
            "models": [
                (
                    m.model_copy(update={"fallback_model_slug": replacement})
                    if m.slug == source
                    else m
                )
                for m in payload.models
            ]
        }
    )
    mocker.patch.object(retire_mod, "get_catalog", return_value=patched)
    mocker.patch("backend.data.db.connect", new=AsyncMock())

    args = _build_parser().parse_args([source])  # no --replacement
    assert await _run_cli(args) == 1  # dry run banner path
    assert args.replacement == replacement
    assert await _node_model(node_id) == _node_value(source)  # nothing written


@pytest.mark.asyncio(loop_scope="session")
async def test_double_revert_refused_by_atomic_claim(retirement_env):
    """The second revert of the same migration must refuse — the guarded
    in-transaction claim matches zero rows (the TOCTOU race guard)."""
    source, replacement = _two_catalog_slugs()
    await _seed_node(retirement_env, _node_value(source))
    await retire_model(source, replacement)
    migration_id = (await list_model_migrations())[0].id

    await revert_model_migration(migration_id)
    with pytest.raises(ValueError, match="already been reverted"):
        await revert_model_migration(migration_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_revert_unknown_migration_id_refused(retirement_env):
    with pytest.raises(ValueError, match="not found"):
        await revert_model_migration("no-such-migration-id")
