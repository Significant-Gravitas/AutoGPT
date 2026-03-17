"""Tests for business understanding merge and format logic."""

from datetime import datetime, timezone
from typing import Any

from backend.data.understanding import (
    BusinessUnderstanding,
    BusinessUnderstandingInput,
    format_understanding_for_prompt,
    merge_business_understanding_data,
)


def _make_input(**kwargs: Any) -> BusinessUnderstandingInput:
    """Create a BusinessUnderstandingInput with only the specified fields."""
    return BusinessUnderstandingInput.model_validate(kwargs)


# ─── merge_business_understanding_data: suggested_prompts ─────────────


def test_merge_suggested_prompts_overwrites_existing():
    """New suggested_prompts should fully replace existing ones (not append)."""
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": ["Old prompt 1", "Old prompt 2"],
    }
    input_data = _make_input(
        suggested_prompts=["New prompt A", "New prompt B", "New prompt C"],
    )

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == [
        "New prompt A",
        "New prompt B",
        "New prompt C",
    ]


def test_merge_suggested_prompts_none_preserves_existing():
    """When input has suggested_prompts=None, existing prompts are preserved."""
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": ["Keep me"],
    }
    input_data = _make_input(industry="Finance")

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == ["Keep me"]
    assert result["business"]["industry"] == "Finance"


def test_merge_suggested_prompts_added_to_empty_data():
    """Suggested prompts are set at top level even when starting from empty data."""
    existing: dict[str, Any] = {}
    input_data = _make_input(suggested_prompts=["Prompt 1"])

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == ["Prompt 1"]


def test_merge_suggested_prompts_empty_list_overwrites():
    """An explicit empty list should overwrite existing prompts."""
    existing: dict[str, Any] = {
        "suggested_prompts": ["Old prompt"],
        "business": {"version": 1},
    }
    input_data = _make_input(suggested_prompts=[])

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == []


# ─── format_understanding_for_prompt: excludes suggested_prompts ──────


def test_format_understanding_excludes_suggested_prompts():
    """suggested_prompts is UI-only and must NOT appear in the system prompt."""
    understanding = BusinessUnderstanding(
        id="test-id",
        user_id="user-1",
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
        user_name="Alice",
        industry="Technology",
        suggested_prompts=["Automate reports", "Set up alerts", "Track KPIs"],
    )

    formatted = format_understanding_for_prompt(understanding)

    assert "Alice" in formatted
    assert "Technology" in formatted
    assert "suggested_prompts" not in formatted
    assert "Automate reports" not in formatted
    assert "Set up alerts" not in formatted
    assert "Track KPIs" not in formatted
