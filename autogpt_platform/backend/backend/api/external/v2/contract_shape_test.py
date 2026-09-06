"""
Rules the v2 contract holds across every model, checked against the schema it
publishes rather than the source it is written in.

The v1 surface accumulated four spellings of "credentials for a run" and money
in three units because each model was reviewed alone. These tests read the whole
OpenAPI document, so a new model cannot reintroduce a spelling the API retired.
"""

from typing import Any, Iterator, get_args

import prisma.enums
import pytest

import backend.blocks._base as block_types
from backend.data.model import CREDENTIALS_ADAPTER

# `node_exec_id` is deliberately absent: v2 renames graph executions to runs,
# and a node execution is not one.
RETIRED_FIELD_NAMES = {
    "input_data": "inputs",
    "output_data": "outputs",
    "credentials": "credentials_inputs",
    "credential_inputs": "credentials_inputs",
    "agent_credentials": "credentials_inputs",
    "execution_id": "run_id",
    "cost_amount": "cost_cents",
    "duration": "duration_seconds",
}

# A property whose name contains one of these holds an amount of money.
MONEY_WORDS = ("cost", "balance", "amount", "price")

# ...unless it is one of these, which are not amounts.
NOT_AMOUNTS = {"cost_type", "cost_filter", "costs"}

# A property whose name ends with one of these is a point in time.
TIME_SUFFIXES = ("_at", "_time", "_end", "_start")


def test_no_model_uses_a_retired_field_name() -> None:
    offenders = [
        f"{model}.{field} (use {RETIRED_FIELD_NAMES[field]})"
        for model, field, _ in _properties()
        if field in RETIRED_FIELD_NAMES
    ]
    assert not offenders, f"retired field names in the v2 contract: {offenders}"


def test_every_money_amount_is_named_in_cents() -> None:
    """Three units and four spellings is how v1's billing fields ended up."""
    offenders = [
        f"{model}.{field}"
        for model, field, _ in _properties()
        if field not in NOT_AMOUNTS
        and any(word in field for word in MONEY_WORDS)
        and not field.endswith("_cents")
        and not field.endswith("_count")
    ]
    assert not offenders, f"money fields without their unit: {offenders}"


def test_every_duration_is_named_in_seconds() -> None:
    offenders = [
        f"{model}.{field}"
        for model, field, _ in _properties()
        if "duration" in field and not field.endswith(("_seconds", "_count"))
    ]
    assert not offenders, f"durations without their unit: {offenders}"


def test_every_point_in_time_is_a_timestamp_not_a_number() -> None:
    """`current_period_end` was unix seconds beside ISO datetimes in one model."""
    offenders = [
        f"{model}.{field}: {schema}"
        for model, field, schema in _properties()
        if field.endswith(TIME_SUFFIXES) and _is_numeric(schema)
    ]
    assert not offenders, f"times published as numbers: {offenders}"


def test_the_schema_has_models_to_check() -> None:
    """Guards every test above against passing on an empty document."""
    assert len(list(_properties())) > 300


def _properties() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Every (model, field, schema) the v2 OpenAPI document publishes."""
    from .app import v2_app

    schemas = v2_app.openapi()["components"]["schemas"]
    for model, schema in schemas.items():
        for field, field_schema in (schema.get("properties") or {}).items():
            yield model, field, field_schema


def _is_numeric(schema: dict[str, Any]) -> bool:
    variants = schema.get("anyOf") or schema.get("oneOf") or [schema]
    return any(variant.get("type") in ("integer", "number") for variant in variants)


@pytest.fixture(scope="module", autouse=True)
def _quiet_mcp_registration() -> None:
    """Importing the app registers the MCP tools; nothing here needs the log."""
    import logging

    logging.getLogger("backend.api.external.v2.mcp_server").setLevel(logging.WARNING)


# ============================================================================
# Enums: v2 publishes its own, and they cannot drift from what they mirror
# ============================================================================

# v2 value set -> the internal value set it must equal, and why anything differs.
MIRRORED_ENUMS: dict[str, tuple[set[str], set[str], str]] = {}


def _register() -> None:
    from . import models

    MIRRORED_ENUMS.update(
        {
            "RunStatus": (
                set(get_args(models.RunStatus)),
                {s.value for s in prisma.enums.AgentExecutionStatus},
                "",
            ),
            "AgentRunReviewStatus": (
                set(get_args(models.AgentRunReviewStatus)),
                {s.value for s in prisma.enums.ReviewStatus},
                "",
            ),
            "TransactionType": (
                set(get_args(models.TransactionType)),
                {t.value for t in prisma.enums.CreditTransactionType},
                "",
            ),
            "SubscriptionTierValue": (
                set(get_args(models.SubscriptionTierValue)),
                {t.value for t in prisma.enums.SubscriptionTier},
                "",
            ),
            "SubmissionStatus": (
                set(get_args(models.SubmissionStatus)),
                {s.value for s in prisma.enums.SubmissionStatus},
                "",
            ),
            "CredentialType": (
                set(get_args(models.CredentialType)),
                _credential_union_types(),
                "",
            ),
            "SearchContentType": (
                {t.value for t in models.SearchContentType},
                {t.value for t in prisma.enums.ContentType} - {"CHAT_SESSION"},
                "chat sessions have no v2 surface and no scope that grants them",
            ),
            "BlockCostType": (
                {t.value for t in models.BlockCostType},
                {t.value for t in block_types.BlockCostType},
                "",
            ),
            "BlockType": (
                {t.name for t in models.BlockType},
                {t.name for t in block_types.BlockType},
                "compared by member name: the internal values are UI labels",
            ),
        }
    )


def test_no_v2_enum_has_drifted_from_the_enum_it_mirrors() -> None:
    """A hand-copied enum is silent when its source grows: the new value simply
    fails response validation in production. These pin every copy."""
    _register()
    drifted = {
        name: {
            "missing from v2": sorted(source - published),
            "not in the source": sorted(published - source),
            "deliberately": reason,
        }
        for name, (published, source, reason) in MIRRORED_ENUMS.items()
        if published != source
    }
    assert not drifted, f"v2 enums out of step with their sources: {drifted}"


def test_every_mirrored_enum_is_registered() -> None:
    """Guards the test above against an empty table."""
    _register()
    assert len(MIRRORED_ENUMS) == 9


def _credential_union_types() -> set[str]:
    """The `type` discriminator values a `Credentials` object can actually carry.

    Not `data.model.CredentialsType`, which also lists `device_code` — no member
    of the union declares it, and a device-code grant stores an
    `OAuth2Credentials`.
    """
    return set(CREDENTIALS_ADAPTER.core_schema["choices"])


# ============================================================================
# Credential requirements: read the schema graphs actually publish
# ============================================================================


async def test_credential_requirements_are_read_from_a_real_graph_schema() -> None:
    """The parser looked for `provider` or an `anyOf` of them; graphs publish
    `credentials_provider` as a list, so both requirement endpoints answered 200
    with an empty list for every agent that needs a credential."""
    from unittest import mock

    from .integrations import helpers

    schema = _one_block_graph_credentials_schema()
    assert schema["properties"], "fixture graph publishes no credentials field"

    with mock.patch.object(
        helpers.creds_manager.store,
        "get_all_creds",
        new=mock.AsyncMock(return_value=[]),
    ):
        requirements = await helpers.get_credential_requirements(schema, "user-1")

    assert requirements, f"no requirements parsed from {schema}"
    assert requirements[0].provider
    assert requirements[0].field_name in schema["properties"]
    assert requirements[0].supported_types


def _one_block_graph_credentials_schema() -> dict[str, Any]:
    """`credentials_input_schema` of a graph holding one credential-taking block."""
    from datetime import datetime, timezone

    from backend.blocks import get_blocks
    from backend.data.graph import GraphModel, NodeModel

    block_id = next(
        block_id
        for block_id, block in get_blocks().items()
        if any(
            "credentials" in field
            for field in block().input_schema.jsonschema().get("properties", {})
        )
    )
    graph = GraphModel(
        id="g",
        version=1,
        name="fixture",
        description="",
        user_id="u",
        created_at=datetime.now(timezone.utc),
        nodes=[
            NodeModel(
                id="n1",
                block_id=block_id,
                input_default={},
                graph_id="g",
                graph_version=1,
            )
        ],
        links=[],
    )
    return graph.credentials_input_schema
