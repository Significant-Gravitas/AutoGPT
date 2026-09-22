"""Tests for DataForB2B filter-building helpers (coerce_scalar, build_slot_condition,
finalize_filters)."""

import pytest

from backend.blocks.dataforb2b._enums import FilterOperator
from backend.blocks.dataforb2b._filters import (
    build_slot_condition,
    coerce_scalar,
    finalize_filters,
)


def test_coerce_scalar_bool():
    assert coerce_scalar("true") is True
    assert coerce_scalar("False") is False
    assert coerce_scalar(True) is True


def test_coerce_scalar_number():
    assert coerce_scalar("42") == 42
    assert isinstance(coerce_scalar("42"), int)
    assert coerce_scalar("3.5") == 3.5
    assert isinstance(coerce_scalar("3.5"), float)


def test_coerce_scalar_passthrough_string():
    assert coerce_scalar("software engineer") == "software engineer"


def test_build_slot_condition_empty_returns_none():
    assert build_slot_condition(None, FilterOperator.EQUALS, "x") is None
    assert build_slot_condition("industry", FilterOperator.EQUALS, None) is None
    assert build_slot_condition("industry", FilterOperator.EQUALS, "   ") is None


def test_build_slot_condition_default_operator():
    cond = build_slot_condition("industry", None, "software")
    assert cond == {"column": "industry", "type": "=", "value": "software"}


def test_build_slot_condition_in_operator():
    cond = build_slot_condition(
        "industry", FilterOperator.IN, "software, finance, , retail"
    )
    assert cond == {
        "column": "industry",
        "type": "in",
        "value": ["software", "finance", "retail"],
    }


def test_build_slot_condition_not_in_operator():
    cond = build_slot_condition("industry", FilterOperator.NOT_IN, "software, finance")
    assert cond == {
        "column": "industry",
        "type": "not_in",
        "value": ["software", "finance"],
    }


def test_build_slot_condition_in_operator_empty_returns_none():
    assert build_slot_condition("industry", FilterOperator.IN, "  , , ") is None


def test_build_slot_condition_between_operator():
    cond = build_slot_condition("employee_count", FilterOperator.BETWEEN, "10,50")
    assert cond == {
        "column": "employee_count",
        "type": "between",
        "value": 10,
        "value2": 50,
    }


def test_build_slot_condition_between_operator_wrong_count_raises():
    """Thread ED7c/DxBb: 'between' with != 2 values must raise, not silently drop."""
    with pytest.raises(ValueError, match="exactly two"):
        build_slot_condition("employee_count", FilterOperator.BETWEEN, "10,50,90")

    with pytest.raises(ValueError, match="exactly two"):
        build_slot_condition("employee_count", FilterOperator.BETWEEN, "10")


def test_build_slot_condition_like_operator_keeps_raw_string():
    cond = build_slot_condition("company_name", FilterOperator.LIKE, "acme corp")
    assert cond == {"column": "company_name", "type": "like", "value": "acme corp"}


def test_build_slot_condition_coerces_scalar_for_other_operators():
    cond = build_slot_condition("employee_count", FilterOperator.GREATER_THAN, "100")
    assert cond == {"column": "employee_count", "type": ">", "value": 100}


def test_finalize_filters_none_when_no_conditions_or_advanced():
    assert finalize_filters([], "and", None) is None


def test_finalize_filters_conditions_only():
    conditions = [{"column": "industry", "type": "=", "value": "software"}]
    assert finalize_filters(conditions, "and", None) == {
        "op": "and",
        "conditions": conditions,
    }


def test_finalize_filters_invalid_match_defaults_to_and():
    conditions = [{"column": "industry", "type": "=", "value": "software"}]
    assert finalize_filters(conditions, "invalid", None)["op"] == "and"


def test_finalize_filters_or_match():
    conditions = [{"column": "industry", "type": "=", "value": "software"}]
    assert finalize_filters(conditions, "or", None) == {
        "op": "or",
        "conditions": conditions,
    }


def test_finalize_filters_advanced_only_dict_with_conditions():
    advanced = {"op": "and", "conditions": [{"column": "x", "type": "=", "value": 1}]}
    assert finalize_filters([], "and", advanced) == advanced


def test_finalize_filters_advanced_only_list():
    advanced = [{"column": "x", "type": "=", "value": 1}]
    assert finalize_filters([], "and", advanced) == {
        "op": "and",
        "conditions": advanced,
    }


def test_finalize_filters_advanced_only_bare_condition_dict():
    advanced = {"column": "x", "type": "=", "value": 1}
    assert finalize_filters([], "and", advanced) == {
        "op": "and",
        "conditions": [advanced],
    }


def test_finalize_filters_merges_slots_and_advanced_with_and():
    conditions = [{"column": "industry", "type": "=", "value": "software"}]
    advanced = {"column": "x", "type": "=", "value": 1}
    result = finalize_filters(conditions, "and", advanced)
    assert result == {
        "op": "and",
        "conditions": [
            {"op": "and", "conditions": conditions},
            {"op": "and", "conditions": [advanced]},
        ],
    }
