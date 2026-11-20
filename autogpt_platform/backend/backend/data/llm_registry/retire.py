"""Retire catalog models: migrate AgentNodes off a model, revertably.

Retirement is about GRAPH NODES — it does not edit the catalog file. The
operator flips the model's ``enabled: False`` in a catalog PR separately;
this module rewrites existing workflow references and records the operation
in ``LlmModelMigration`` (per-install runtime state) so it can be reverted.

Only nodes of blocks whose ``model`` input is an ``LLMModel`` are rewritten.
Other blocks name an input ``model`` too, with lists of their own (Stagehand
offers a few Claude and GPT models, Codex its own GPT models), and a catalog
replacement can be a value their schema refuses. Those nodes are counted and
left alone.

A preset can override one node's inputs, in a row named with that node's mask
key (``graph.NODE_INPUT_MASK_PREFIX``). Such a row is rewritten too when its
node is one of those LLM blocks, and recorded for revert. Other preset rows
that name the model, most often a workflow input's value, are counted and left
alone.

CLI::

    python -m backend.data.llm_registry.retire --usage <slug>
    python -m backend.data.llm_registry.retire <slug> --replacement <slug> \
        [--reason "..."] [--yes]
    python -m backend.data.llm_registry.retire --revert <migration-id>
    python -m backend.data.llm_registry.retire --list [--include-reverted]

Without ``--yes`` the retire command prints a dry-run summary (usage count,
replacement validation) and exits non-zero so scripts can't retire by
accident.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, LiteralString, cast

import prisma.models
from prisma.errors import UniqueViolationError
from pydantic import BaseModel

import backend.data.db
from backend.data.db import get_database_schema, query_raw_with_schema, transaction
from backend.data.graph import NODE_INPUT_MASK_PREFIX, find_llm_model_fields
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.registry import get_model, load_catalog

logger = logging.getLogger(__name__)


class MigrationResult(BaseModel):
    source_model_slug: str
    target_model_slug: str
    nodes_migrated: int
    preset_inputs_migrated: int
    migration_id: str | None  # None when nothing referenced the model


class RevertResult(BaseModel):
    migration_id: str
    source_model_slug: str
    target_model_slug: str
    nodes_reverted: int
    # Nodes that were manually changed to a third model after the migration
    # are left alone (the revert UPDATE is guarded on the migrated value).
    nodes_already_changed: int
    preset_inputs_reverted: int
    preset_inputs_already_changed: int


class ModelUsage(BaseModel):
    """What a retirement of one model rewrites, and what it leaves alone."""

    nodes: int
    preset_inputs: int
    # Nodes of other blocks whose own ``model`` list holds the same value.
    other_block_nodes: int
    # Preset rows naming the model that are not an LLM node's override, most
    # often a workflow input's value.
    other_preset_inputs: int


def _schema_format(query_template: str) -> LiteralString:
    """Format a ``{schema_prefix}`` query for transaction-scoped raw SQL.

    ``query_raw_with_schema``/``execute_raw_with_schema`` can't run inside an
    existing transaction client, so FOR UPDATE reads format the template with
    the same prefix logic and go through ``tx.query_raw``.
    """
    schema = get_database_schema()
    schema_prefix = f'"{schema}".' if schema != "public" else ""
    # cast: prisma types raw queries as LiteralString; the template is a
    # module-level literal and the prefix derives from DATABASE_URL config.
    return cast(LiteralString, query_template.format(schema_prefix=schema_prefix))


# AgentNode.constantInput stores FULL enum values — mixed bare and
# provider-prefixed forms (``claude-opus-4-7``, ``moonshotai/kimi-k2.5``),
# exactly the catalog slugs; graph.migrate_llm_models compares against
# ``LLMModel.value`` unmodified. Node values therefore map to catalog slugs
# by IDENTITY — stripping the provider prefix (as an earlier revision did)
# both misses prefixed models and writes out-of-enum values that the
# startup migration would stomp to the global fallback.


def _validate_replacement(slug: str) -> None:
    """Replacement must be a known, enabled catalog model (kill-switch aware)."""
    model = get_model(slug)
    if model is None:
        raise ValueError(f"Replacement model '{slug}' is not in the catalog")
    if not model.is_enabled:
        raise ValueError(
            f"Replacement model '{slug}' is disabled in the catalog — "
            f"retiring onto a dead model would strand the nodes again"
        )


@functools.cache
def _llm_model_block_ids_json() -> str:
    """JSON array of the blocks whose ``model`` input is an ``LLMModel``.
    Built once per process: it instantiates every block, and blocks don't
    change while a process runs."""
    return json.dumps(
        sorted(
            block_id
            for block_id, field_name in find_llm_model_fields().items()
            if field_name == "model"
        )
    )


# A preset's override of one LLM-block node: the row named with that node's
# mask key, holding the node's ``model``. $1 is the model slug, $2 a JSON
# array of the LLM block ids, $3 ``NODE_INPUT_MASK_PREFIX``.
_LLM_PRESET_INPUTS = """
    FROM {schema_prefix}"AgentNodeExecutionInputOutput" AS io
    JOIN {schema_prefix}"AgentPreset" AS p ON p."id" = io."agentPresetId"
    JOIN {schema_prefix}"AgentNode" AS n
        ON n."agentGraphId" = p."agentGraphId"
        AND n."agentGraphVersion" = p."agentGraphVersion"
        AND io."name" = $3::text || split_part(n."id", '-', 1)
    WHERE io."data"::jsonb->>'model' = $1
    AND n."agentBlockId" IN (SELECT jsonb_array_elements_text($2::jsonb))
"""


async def model_usage(slug: str) -> ModelUsage:
    """What a retirement of *slug* would rewrite, and what it leaves alone."""
    rows = await query_raw_with_schema(
        'WITH llm_presets AS (SELECT io."id"'
        + _LLM_PRESET_INPUTS
        + """)
        SELECT
            (SELECT COUNT(*) FROM {schema_prefix}"AgentNode"
             WHERE "constantInput"::jsonb->>'model' = $1
             AND "agentBlockId" IN (SELECT jsonb_array_elements_text($2::jsonb))
            ) AS nodes,
            (SELECT COUNT(*) FROM llm_presets) AS preset_inputs,
            (SELECT COUNT(*) FROM {schema_prefix}"AgentNode"
             WHERE "constantInput"::jsonb->>'model' = $1
             AND "agentBlockId" NOT IN (SELECT jsonb_array_elements_text($2::jsonb))
            ) AS other_block_nodes,
            (SELECT COUNT(*) FROM {schema_prefix}"AgentNodeExecutionInputOutput" AS io
             WHERE io."agentPresetId" IS NOT NULL
             AND (io."data"::jsonb->>'model' = $1 OR io."data"::jsonb #>> '{{}}' = $1)
             AND io."id" NOT IN (SELECT "id" FROM llm_presets)
            ) AS other_preset_inputs
        """,
        slug,
        _llm_model_block_ids_json(),
        NODE_INPUT_MASK_PREFIX,
    )
    return ModelUsage(**{name: int(count) for name, count in rows[0].items()})


async def _migrate_nodes_in_tx(
    tx, source_value: str, target_value: str, llm_block_ids_json: str
) -> list[str]:
    """Lock + rewrite AgentNode model references. Returns migrated node ids."""
    node_ids_result = await tx.query_raw(
        _schema_format(
            """
            SELECT id
            FROM {schema_prefix}"AgentNode"
            WHERE "constantInput"::jsonb->>'model' = $1
            AND "agentBlockId" IN (SELECT jsonb_array_elements_text($2::jsonb))
            FOR UPDATE
            """
        ),
        source_value,
        llm_block_ids_json,
    )
    migrated_node_ids = [row["id"] for row in node_ids_result or []]
    if migrated_node_ids:
        await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNode"
                SET "constantInput" = JSONB_SET(
                    "constantInput"::jsonb,
                    '{{model}}',
                    to_jsonb($1::text)
                )
                WHERE id::text IN (
                    SELECT jsonb_array_elements_text($2::jsonb)
                )
                """
            ),
            target_value,
            json.dumps(migrated_node_ids),
        )
    return migrated_node_ids


async def _migrate_preset_inputs_in_tx(
    tx, source_value: str, target_value: str, llm_block_ids_json: str
) -> list[str]:
    """Lock + rewrite the presets' overrides of LLM-block nodes. Returns the
    rewritten row ids."""
    rows = await tx.query_raw(
        _schema_format('SELECT io."id"' + _LLM_PRESET_INPUTS + "FOR UPDATE OF io"),
        source_value,
        llm_block_ids_json,
        NODE_INPUT_MASK_PREFIX,
    )
    row_ids = [row["id"] for row in rows or []]
    if row_ids:
        await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNodeExecutionInputOutput"
                SET "data" = JSONB_SET("data"::jsonb, '{{model}}', to_jsonb($1::text))
                WHERE "id" IN (SELECT jsonb_array_elements_text($2::jsonb))
                """
            ),
            target_value,
            json.dumps(row_ids),
        )
    return row_ids


async def retire_model(
    slug: str, replacement_slug: str, reason: str | None = None
) -> MigrationResult:
    """Rewrite every LLM-block node referencing *slug*, and every preset
    override of such a node, to *replacement_slug*.

    Records an ``LlmModelMigration`` row when anything was touched; the
    partial unique index rejects a second active migration for the same
    source (revert the first one before retrying).
    """
    if slug == replacement_slug:
        raise ValueError("Replacement must differ from the retired model")
    _validate_replacement(replacement_slug)
    source = get_model(slug)
    if source is not None and source.is_enabled:
        logger.warning(
            "Model '%s' is still enabled in the catalog — remember to flip "
            "enabled: False in a catalog PR, or new graphs will keep "
            "selecting it",
            slug,
        )

    # Built before the transaction opens, so the first build holds no locks.
    llm_block_ids_json = _llm_model_block_ids_json()
    try:
        async with transaction() as tx:
            migrated_node_ids = await _migrate_nodes_in_tx(
                tx, slug, replacement_slug, llm_block_ids_json
            )
            migrated_preset_input_ids = await _migrate_preset_inputs_in_tx(
                tx, slug, replacement_slug, llm_block_ids_json
            )
            migration_id: str | None = None
            if migrated_node_ids or migrated_preset_input_ids:
                record = await tx.llmmodelmigration.create(
                    data={
                        "sourceModelSlug": slug,
                        "targetModelSlug": replacement_slug,
                        "reason": reason,
                        "migratedNodeIds": json.dumps(migrated_node_ids),
                        "nodeCount": len(migrated_node_ids),
                        "migratedPresetInputIds": json.dumps(migrated_preset_input_ids),
                    }
                )
                migration_id = record.id
    except UniqueViolationError as exc:
        raise ValueError(
            f"An active migration for '{slug}' already exists — revert it "
            f"first (see --list)"
        ) from exc

    return MigrationResult(
        source_model_slug=slug,
        target_model_slug=replacement_slug,
        nodes_migrated=len(migrated_node_ids),
        preset_inputs_migrated=len(migrated_preset_input_ids),
        migration_id=migration_id,
    )


async def revert_model_migration(migration_id: str) -> RevertResult:
    """Restore the nodes and preset inputs a migration touched to the
    original model."""
    migration = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": migration_id}
    )
    if not migration:
        raise ValueError(f"Migration '{migration_id}' not found")
    if migration.isReverted:
        raise ValueError(f"Migration '{migration_id}' has already been reverted")

    migrated_node_ids = _recorded_ids(migration.migratedNodeIds)
    migrated_preset_input_ids = _recorded_ids(migration.migratedPresetInputIds)
    if not migrated_node_ids and not migrated_preset_input_ids:
        raise ValueError("Nothing to revert in this migration")

    async with transaction() as tx:
        # Atomic claim: the guarded update IS the lock — a concurrent revert
        # of the same migration matches zero rows and aborts here without
        # touching any nodes (fixes the check-then-act race on isReverted).
        claimed = await tx.llmmodelmigration.update_many(
            where={"id": migration_id, "isReverted": False},
            data={"isReverted": True, "revertedAt": datetime.now(timezone.utc)},
        )
        if claimed == 0:
            raise ValueError(f"Migration '{migration_id}' has already been reverted")
        # Guarded on the migrated value: nodes manually repointed at a third
        # model since the migration are left alone.
        result = await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNode"
                SET "constantInput" = JSONB_SET(
                    "constantInput"::jsonb,
                    '{{model}}',
                    to_jsonb($1::text)
                )
                WHERE id::text IN (
                    SELECT jsonb_array_elements_text($2::jsonb)
                )
                AND "constantInput"::jsonb->>'model' = $3
                """
            ),
            migration.sourceModelSlug,
            json.dumps(migrated_node_ids),
            migration.targetModelSlug,
        )
        nodes_reverted = result if isinstance(result, int) else 0
        result = await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNodeExecutionInputOutput"
                SET "data" = JSONB_SET("data"::jsonb, '{{model}}', to_jsonb($1::text))
                WHERE "id" IN (SELECT jsonb_array_elements_text($2::jsonb))
                AND "data"::jsonb->>'model' = $3
                """
            ),
            migration.sourceModelSlug,
            json.dumps(migrated_preset_input_ids),
            migration.targetModelSlug,
        )
        preset_inputs_reverted = result if isinstance(result, int) else 0

    if get_model(migration.sourceModelSlug) is None:
        logger.warning(
            "Reverted nodes now reference '%s', which is not in the catalog — "
            "re-add it in a catalog PR if this revert is meant to be permanent",
            migration.sourceModelSlug,
        )

    return RevertResult(
        migration_id=migration_id,
        source_model_slug=migration.sourceModelSlug,
        target_model_slug=migration.targetModelSlug,
        nodes_reverted=nodes_reverted,
        nodes_already_changed=len(migrated_node_ids) - nodes_reverted,
        preset_inputs_reverted=preset_inputs_reverted,
        preset_inputs_already_changed=(
            len(migrated_preset_input_ids) - preset_inputs_reverted
        ),
    )


def _recorded_ids(value: object) -> list[str]:
    """An id list from a migration record's JSON column, which Prisma returns
    parsed or as text."""
    return value if isinstance(value, list) else json.loads(str(value))


class MigrationRow(BaseModel):
    """One retirement record, as listed by the CLI."""

    id: str
    source_model_slug: str
    target_model_slug: str
    reason: str | None
    node_count: int
    preset_input_count: int
    is_reverted: bool
    reverted_at: str | None
    created_at: str


async def list_model_migrations(include_reverted: bool = False) -> list[MigrationRow]:
    """Recent migrations, newest first."""
    where: Any = None if include_reverted else {"isReverted": False}
    records = await prisma.models.LlmModelMigration.prisma().find_many(
        where=where, order={"createdAt": "desc"}
    )
    return [
        MigrationRow(
            id=r.id,
            source_model_slug=r.sourceModelSlug,
            target_model_slug=r.targetModelSlug,
            reason=r.reason,
            node_count=r.nodeCount,
            preset_input_count=len(_recorded_ids(r.migratedPresetInputIds)),
            is_reverted=r.isReverted,
            reverted_at=r.revertedAt.isoformat() if r.revertedAt else None,
            created_at=r.createdAt.isoformat(),
        )
        for r in records
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.data.llm_registry.retire",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("slug", nargs="?", help="model slug to retire")
    parser.add_argument(
        "--replacement",
        help="catalog slug nodes migrate to (defaults to the retired model's "
        "catalog fallback_model_slug when set)",
    )
    parser.add_argument("--reason", help="recorded on the migration row")
    parser.add_argument("--yes", action="store_true", help="execute (no dry-run)")
    parser.add_argument(
        "--usage", metavar="SLUG", help="print what retiring SLUG rewrites and exit"
    )
    parser.add_argument("--revert", metavar="MIGRATION_ID", help="revert a migration")
    parser.add_argument("--list", action="store_true", help="list migrations")
    parser.add_argument("--include-reverted", action="store_true")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    # One action per invocation — a retire positional combined with
    # --usage/--revert/--list would silently ignore the positional.
    modes = [bool(args.slug), bool(args.usage), bool(args.revert), args.list]
    if sum(modes) != 1:
        print(
            "Pick exactly one action: <slug> --replacement …, --usage, "
            "--revert, or --list",
            file=sys.stderr,
        )
        return 2

    await backend.data.db.connect()
    load_catalog()

    if args.usage:
        usage = await model_usage(args.usage)
        print(
            f"{args.usage}: {usage.nodes} node(s), "
            f"{usage.preset_inputs} preset input(s)"
        )
        _print_left_alone(args.usage, usage)
        return 0
    if args.revert:
        result = await revert_model_migration(args.revert)
        print(result.model_dump_json(indent=2))
        return 0
    if args.list:
        for row in await list_model_migrations(args.include_reverted):
            print(row.model_dump_json())
        return 0
    if args.slug and not args.replacement:
        # Standing replacement pointer: the catalog's fallback_model_slug
        # pre-fills --replacement so routine retirements are one argument.
        by_slug = {m.slug: m for m in get_catalog().models}
        entry = by_slug.get(args.slug)
        if entry is not None and entry.fallback_model_slug:
            args.replacement = entry.fallback_model_slug
            print(
                f"--replacement defaulted from catalog fallback: "
                f"'{args.replacement}'"
            )
    if not args.slug or not args.replacement:
        print(
            "slug and --replacement are required to retire (no catalog "
            "fallback_model_slug is set for this model)",
            file=sys.stderr,
        )
        return 2

    usage = await model_usage(args.slug)
    if not args.yes:
        _validate_replacement(args.replacement)
        print(
            f"DRY RUN: would migrate {usage.nodes} node(s) and "
            f"{usage.preset_inputs} preset input(s) from '{args.slug}' to "
            f"'{args.replacement}'. Re-run with --yes to execute."
        )
        _print_left_alone(args.slug, usage)
        return 1
    result = await retire_model(args.slug, args.replacement, reason=args.reason)
    print(result.model_dump_json(indent=2))
    _print_left_alone(args.slug, usage)
    print(
        "Reminder: retirement only rewrites graph nodes — flip "
        f"'{args.slug}' to enabled=False in a catalog PR to stop new use."
    )
    return 0


def _print_left_alone(slug: str, usage: ModelUsage) -> None:
    if usage.other_block_nodes:
        print(
            f"{usage.other_block_nodes} node(s) of other blocks also name "
            f"'{slug}' in their own model list and are left alone."
        )
    if usage.other_preset_inputs:
        print(
            f"{usage.other_preset_inputs} preset input(s) name '{slug}' "
            "without overriding an LLM block's model (most often a workflow "
            "input's value) and are left alone."
        )


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_cli(_build_parser().parse_args())))
