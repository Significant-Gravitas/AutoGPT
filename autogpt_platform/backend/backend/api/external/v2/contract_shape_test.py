"""
Rules the v2 contract holds across every model, checked against the schema it
publishes rather than the source it is written in.

The v1 surface accumulated four spellings of "credentials for a run" and money
in three units because each model was reviewed alone. These tests read the whole
OpenAPI document, so a new model cannot reintroduce a spelling the API retired.
"""

from typing import Any, Iterator

import pytest

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
