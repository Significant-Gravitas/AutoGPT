"""Real-DB tests for the model retirement CLI/library.

Seeds minimal User/AgentGraph/AgentNode/AgentPreset rows, exercises retire →
revert against the actual `{schema_prefix}` SQL, and verifies the catalog-based
validation and the active-migration partial unique index.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock

import prisma.models
import pytest
from prisma.errors import UniqueViolationError

import backend.data.llm_registry.registry as reg
from backend.blocks.ai_image_generator_block import AIImageGeneratorBlock
from backend.blocks.codex import CodeGenerationBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.blocks.stagehand.blocks import StagehandObserveBlock
from backend.data.graph import (
    NODE_INPUT_MASK_PREFIX,
    find_llm_model_fields,
    migrate_llm_models,
)
from backend.data.llm_registry.llm_models import (
    DEFAULT_LLM_MODEL,
    LEGACY_MODEL_MAPPINGS,
    LLMModel,
)
from backend.data.llm_registry.registry import get_all_models, load_catalog
from backend.data.llm_registry.retire import (
    list_model_migrations,
    model_usage,
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
    # Presets restrict their graph's deletion, and their input rows only
    # unlink when a preset goes, so both go first.
    created_presets = {"agentGraphId": {"in": created_graph_ids}}
    await prisma.models.AgentNodeExecutionInputOutput.prisma().delete_many(
        where={"AgentPreset": {"is": created_presets}}
    )
    await prisma.models.AgentPreset.prisma().delete_many(where=created_presets)
    for graph_id in created_graph_ids:
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})


_LLM_BLOCK_ID = AITextGeneratorBlock().id
_STAGEHAND_BLOCK_ID = StagehandObserveBlock().id
_CODEX_BLOCK_ID = CodeGenerationBlock().id
_IMAGE_BLOCK_ID = AIImageGeneratorBlock().id


async def _seed_node(
    created_graph_ids: list[str],
    model_value: str,
    block_id: str = _LLM_BLOCK_ID,
) -> str:
    """Create a graph + one node of *block_id* whose constantInput uses
    *model_value*. Defaults to an LLM block, the kind retirement rewrites."""
    _, node_id = await _seed_graph(created_graph_ids, model_value, block_id)
    return node_id


async def _seed_graph(
    created_graph_ids: list[str],
    model_value: str,
    block_id: str = _LLM_BLOCK_ID,
) -> tuple[str, str]:
    """``_seed_node``, returning the graph id too."""
    block = await prisma.models.AgentBlock.prisma().find_unique(where={"id": block_id})
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
    return graph_id, node.id


async def _seed_preset(graph_id: str, inputs: dict[str, Any]) -> dict[str, str]:
    """Create a preset of *graph_id* holding *inputs*, one row per name as the
    library stores them. Returns each row's id by name."""
    preset = await prisma.models.AgentPreset.prisma().create(
        data={
            "name": "retire-test",
            "description": "retire-test",
            "userId": _TEST_USER_ID,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "InputPresets": {
                "create": [
                    {"name": name, "data": prisma.Json(data)}
                    for name, data in inputs.items()
                ]
            },
        },
        include={"InputPresets": True},
    )
    return {row.name: row.id for row in preset.InputPresets or []}


def _override_key(node_id: str) -> str:
    """The row name a preset stores one node's input overrides under."""
    return f"{NODE_INPUT_MASK_PREFIX}{node_id.split('-')[0]}"


async def _preset_input(row_id: str) -> Any:
    row = await prisma.models.AgentNodeExecutionInputOutput.prisma().find_unique(
        where={"id": row_id}
    )
    assert row is not None
    return row.data


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

    assert (await model_usage(source)).nodes == 1


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

    assert (await model_usage(source)).nodes == 1
    result = await retire_model(source, replacement, reason="prefix regression")
    assert result.nodes_migrated == 1
    assert await _node_model(node_id) == replacement

    revert = await revert_model_migration(result.migration_id)
    assert revert.nodes_reverted == 1
    assert await _node_model(node_id) == source


async def test_retire_leaves_other_blocks_model_inputs_alone(
    retirement_env, mocker, capsys
):
    """Stagehand lists Claude Sonnet 4.6 but not 5.5. Retiring Sonnet 4.5 onto
    5.5 moves the LLM block's node and must leave the Stagehand node's value,
    which 5.5 would make invalid, as it is. The dry run says so."""
    from backend.data.llm_registry.retire import _build_parser, _run_cli

    source, replacement = "claude-sonnet-4-5-20250929", "claude-sonnet-5-5"
    llm_node = await _seed_node(retirement_env, source)
    stagehand_node = await _seed_node(
        retirement_env, source, block_id=_STAGEHAND_BLOCK_ID
    )

    usage = await model_usage(source)
    assert (usage.nodes, usage.other_block_nodes) == (1, 1)
    mocker.patch("backend.data.db.connect", new=AsyncMock())
    args = _build_parser().parse_args([source, "--replacement", replacement])
    assert await _run_cli(args) == 1
    dry_run = capsys.readouterr().out
    assert "would migrate 1 node(s) from" in dry_run
    assert "1 node(s) of other blocks also name" in dry_run

    result = await retire_model(source, replacement)

    assert result.nodes_migrated == 1
    assert await _node_model(llm_node) == replacement
    assert await _node_model(stagehand_node) == source


async def test_boot_pass_moves_shut_down_models_once_and_nothing_else(
    retirement_env, caplog
):
    """The next boot moves an LLM node off a model its provider has shut down,
    in either stored spelling, o1-mini included, and logs what it moved. It
    leaves a model still being served (the retire CLI moves those), a live
    model and other blocks' values alone: the Codex node's ``gpt-5.1-codex``
    (which the Codex block reads as its own ``gpt-5.3-codex``) and an image
    node's model outside
    ``LLMModel`` too. A second boot changes nothing and logs no move."""
    nodes = {
        "o3-mini": await _seed_node(retirement_env, "o3-mini"),
        "openai/o3-mini": await _seed_node(retirement_env, "openai/o3-mini"),
        "o1-mini": await _seed_node(retirement_env, "o1-mini"),
        "still served": await _seed_node(retirement_env, "gpt-5-2025-08-07"),
        "live": await _seed_node(retirement_env, "gpt-6.1-sol"),
        "stagehand": await _seed_node(
            retirement_env, "claude-sonnet-4-5-20250929", _STAGEHAND_BLOCK_ID
        ),
        "codex": await _seed_node(retirement_env, "gpt-5.1-codex", _CODEX_BLOCK_ID),
        # Outside LLMModel, so only the catch-all's block scope keeps it.
        "image": await _seed_node(retirement_env, "Flux 1.1 Pro", _IMAGE_BLOCK_ID),
    }
    expected = {
        "o3-mini": "gpt-6.1-terra",
        "openai/o3-mini": "gpt-6.1-terra",
        "o1-mini": "gpt-6.1-terra",
        "still served": "gpt-5-2025-08-07",
        "live": "gpt-6.1-sol",
        "stagehand": "claude-sonnet-4-5-20250929",
        "codex": "gpt-5.1-codex",
        "image": "Flux 1.1 Pro",
    }
    first_boot_moves = {
        f"LLM model migration moved 1 node(s) and 0 preset override(s) from "
        f"'{stored}' to 'gpt-6.1-terra'"
        for stored in ("o3-mini", "openai/o3-mini", "o1-mini")
    }

    # The real boot pass, over every row in the test database as at app
    # start; only this test's rows are asserted.
    for boot in range(2):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="backend.data.graph"):
            await migrate_llm_models(DEFAULT_LLM_MODEL)
        assert {
            name: await _node_model(node_id) for name, node_id in nodes.items()
        } == expected
        moves = {
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("LLM model migration moved")
        }
        if boot == 0:
            assert first_boot_moves <= moves
        else:
            # The first boot moved every row; nothing is left to move or log.
            assert moves == set()


async def test_boot_pass_rewrites_only_llm_block_preset_overrides(
    retirement_env, caplog
):
    """A preset's override of an LLM node moves off a model OpenAI has shut
    down. The same value in a Codex node's override (the Codex block reads
    gpt-5.1-codex as its own gpt-5.3-codex), an image node's override, and
    objects a workflow input
    holds stay as they are, also under the catch-all that sends values outside
    ``LLMModel`` to the default model."""
    llm_graph, llm_node = await _seed_graph(retirement_env, "gpt-6.1-sol")
    codex_graph, codex_node = await _seed_graph(
        retirement_env, "gpt-5.1-codex", _CODEX_BLOCK_ID
    )
    image_graph, image_node = await _seed_graph(
        retirement_env, "Flux 1.1 Pro", _IMAGE_BLOCK_ID
    )
    llm_rows = await _seed_preset(
        llm_graph,
        {
            _override_key(llm_node): {"model": "gpt-5.1-codex", "prompt": "hi"},
            "settings": {"model": "gpt-5.1-codex"},
            "image_settings": {"model": "Flux 1.1 Pro"},
        },
    )
    codex_rows = await _seed_preset(
        codex_graph, {_override_key(codex_node): {"model": "gpt-5.1-codex"}}
    )
    image_rows = await _seed_preset(
        image_graph, {_override_key(image_node): {"model": "Flux 1.1 Pro"}}
    )
    expected = {
        llm_rows[_override_key(llm_node)]: {"model": "gpt-6.1-sol", "prompt": "hi"},
        llm_rows["settings"]: {"model": "gpt-5.1-codex"},
        llm_rows["image_settings"]: {"model": "Flux 1.1 Pro"},
        codex_rows[_override_key(codex_node)]: {"model": "gpt-5.1-codex"},
        image_rows[_override_key(image_node)]: {"model": "Flux 1.1 Pro"},
    }

    # The real boot pass, over every row in the test database as at app
    # start; only this test's rows are asserted.
    for boot in range(2):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="backend.data.graph"):
            await migrate_llm_models(DEFAULT_LLM_MODEL)
        assert {row: await _preset_input(row) for row in expected} == expected
        assert await _node_model(codex_node) == "gpt-5.1-codex"
        moves = {
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("LLM model migration moved")
        }
        # The LLM node's override counts as a preset move, not a node move.
        preset_move = (
            "LLM model migration moved 0 node(s) and 1 preset override(s) from "
            "'gpt-5.1-codex' to 'gpt-6.1-sol'"
        )
        assert (preset_move in moves) == (boot == 0)


async def test_boot_pass_covers_every_llm_block(retirement_env):
    """A node of every block with an LLMModel field moves, not only the text generator's."""
    block_ids = sorted(find_llm_model_fields())
    assert len(block_ids) > 1
    nodes = {
        block_id: await _seed_node(retirement_env, "o3-mini", block_id=block_id)
        for block_id in block_ids
    }

    await migrate_llm_models(DEFAULT_LLM_MODEL)

    assert {block_id: await _node_model(node) for block_id, node in nodes.items()} == {
        block_id: "gpt-6.1-terra" for block_id in block_ids
    }


async def test_boot_pass_sends_a_retired_slug_outside_llmmodel_to_its_family(
    retirement_env,
):
    """Most mapped spellings are no longer LLMModel members, so only the targeted statement,
    run before the catch-all, lands them on their family's replacement and not the default.
    """
    keys = {
        "claude-3-haiku-20240307": "claude-3-haiku-20240307",
        "anthropic/claude-opus-4-1": "claude-opus-4-1",
        "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
        "z-ai/glm-4.5v": "z-ai/glm-4.5v",
    }
    expected = {
        stored: LEGACY_MODEL_MAPPINGS[key].value for stored, key in keys.items()
    }
    assert not set(expected) & {member.value for member in LLMModel}
    assert DEFAULT_LLM_MODEL.value not in expected.values()
    nodes = {stored: await _seed_node(retirement_env, stored) for stored in expected}
    graph, llm_node = await _seed_graph(retirement_env, "gpt-6.1-sol")
    override = _override_key(llm_node)
    rows = await _seed_preset(
        graph, {override: {"model": "claude-opus-4-1-20250805", "prompt": "hi"}}
    )

    await migrate_llm_models(DEFAULT_LLM_MODEL)

    assert {
        stored: await _node_model(node) for stored, node in nodes.items()
    } == expected
    assert await _preset_input(rows[override]) == {
        "model": LEGACY_MODEL_MAPPINGS["claude-opus-4-1-20250805"].value,
        "prompt": "hi",
    }


async def test_boot_pass_sends_a_value_outside_llmmodel_to_the_default(
    retirement_env, caplog
):
    """The catch-alls: an LLM node, and a preset's override of one, naming a value no
    LLMModel member has run the default model, and the boot log says so."""
    node = await _seed_node(retirement_env, "not-a-real-model")
    graph, llm_node = await _seed_graph(retirement_env, "gpt-6.1-sol")
    override = _override_key(llm_node)
    rows = await _seed_preset(
        graph, {override: {"model": "also-not-real", "prompt": "hi"}}
    )

    with caplog.at_level(logging.INFO, logger="backend.data.graph"):
        await migrate_llm_models(DEFAULT_LLM_MODEL)

    assert await _node_model(node) == DEFAULT_LLM_MODEL.value
    assert await _preset_input(rows[override]) == {
        "model": DEFAULT_LLM_MODEL.value,
        "prompt": "hi",
    }
    outside = f"from a value outside LLMModel to '{DEFAULT_LLM_MODEL.value}'"
    moves = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith("LLM model migration moved")
        and message.endswith(outside)
        and "0 node(s)" not in message
        and "0 preset override(s)" not in message
        for message in moves
    )


async def test_boot_pass_leaves_every_other_llmmodel_member_alone(retirement_env):
    """The catch-all moves a value only when it is outside LLMModel: a node on any member the
    map does not retire stays where it is."""
    stay = [m.value for m in LLMModel if m.value not in LEGACY_MODEL_MAPPINGS]
    nodes = {slug: await _seed_node(retirement_env, slug) for slug in stay}

    await migrate_llm_models(DEFAULT_LLM_MODEL)

    assert {slug: await _node_model(node) for slug, node in nodes.items()} == {
        slug: slug for slug in stay
    }


async def test_boot_pass_logs_every_row_it_moves_per_pair(retirement_env, caplog):
    """The boot log adds up every row a statement moved for one (stored value, replacement) pair."""
    for _ in range(3):
        await _seed_node(retirement_env, "o3-mini")
    graph, llm_node = await _seed_graph(retirement_env, "gpt-6.1-sol")
    for _ in range(2):
        await _seed_preset(graph, {_override_key(llm_node): {"model": "o3-mini"}})

    with caplog.at_level(logging.INFO, logger="backend.data.graph"):
        await migrate_llm_models(DEFAULT_LLM_MODEL)

    assert (
        "LLM model migration moved 3 node(s) and 2 preset override(s) from "
        "'o3-mini' to 'gpt-6.1-terra'"
    ) in {record.getMessage() for record in caplog.records}


async def test_retire_counts_preset_values_and_leaves_them_alone(
    retirement_env, mocker, capsys
):
    """The CLI moves the LLM node and leaves every preset row as it is: a
    preset's override of that node, a Codex node's override, a workflow
    input's value, and a workflow input holding an object with a ``model``
    key (named after no node, so no override). The dry run counts each kind;
    an LLM block given one runs the model's catalog fallback instead
    (``Block._execute``)."""
    from backend.data.llm_registry.retire import _build_parser, _run_cli

    source, replacement = "gpt-5.1-codex", "gpt-6.1-sol"
    llm_graph, llm_node = await _seed_graph(retirement_env, source)
    codex_graph, codex_node = await _seed_graph(retirement_env, source, _CODEX_BLOCK_ID)
    llm_rows = await _seed_preset(
        llm_graph,
        {
            _override_key(llm_node): {"model": source, "sys_prompt": "be brief"},
            "model_choice": source,
            "settings": {"model": source},
        },
    )
    codex_rows = await _seed_preset(
        codex_graph, {_override_key(codex_node): {"model": source}}
    )
    untouched = {
        llm_rows[_override_key(llm_node)]: {"model": source, "sys_prompt": "be brief"},
        llm_rows["model_choice"]: source,
        llm_rows["settings"]: {"model": source},
        codex_rows[_override_key(codex_node)]: {"model": source},
    }

    usage = await model_usage(source)
    assert usage.model_dump() == {
        "nodes": 1,
        "other_block_nodes": 1,
        "preset_overrides": 1,
        "other_preset_inputs": 3,
    }
    mocker.patch("backend.data.db.connect", new=AsyncMock())
    args = _build_parser().parse_args([source, "--replacement", replacement])
    assert await _run_cli(args) == 1
    dry_run = capsys.readouterr().out
    assert "would migrate 1 node(s)" in dry_run
    assert "1 preset override(s) of an LLM node name 'gpt-5.1-codex'" in dry_run
    assert "3 other preset input(s) name 'gpt-5.1-codex'" in dry_run
    assert (
        "An LLM block given 'gpt-5.1-codex' runs 'gpt-6.1-sol', its catalog "
        "fallback, instead." in dry_run
    )

    args = _build_parser().parse_args([source, "--replacement", replacement, "--yes"])
    assert await _run_cli(args) == 0
    applied = capsys.readouterr().out

    assert '"nodes_migrated": 1' in applied
    assert "1 preset override(s) of an LLM node name 'gpt-5.1-codex'" in applied
    # The catalog has switched it off already, so there is nothing to remind.
    assert "Reminder" not in applied
    assert await _node_model(llm_node) == replacement
    assert await _node_model(codex_node) == source
    assert {row: await _preset_input(row) for row in untouched} == untouched


async def test_the_left_alone_note_names_the_fallback_and_its_date(capsys):
    """A model still served runs as itself until its shutdown date, so the
    note names the fallback and the day it starts; with no date, the fallback
    runs at once."""
    from backend.data.llm_registry.retire import ModelUsage, _print_left_alone

    usage = ModelUsage(
        nodes=0, other_block_nodes=0, preset_overrides=1, other_preset_inputs=0
    )
    _print_left_alone("gpt-5-2025-08-07", usage)
    _print_left_alone("o3-mini", usage)
    note = capsys.readouterr().out

    assert (
        "An LLM block given 'gpt-5-2025-08-07' runs 'gpt-6.1-sol', its catalog "
        "fallback, from 2026-12-11." in note
    )
    assert (
        "An LLM block given 'o3-mini' runs 'gpt-6.1-terra', its catalog "
        "fallback, instead." in note
    )


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
