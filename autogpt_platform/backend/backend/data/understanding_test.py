"""Tests for business understanding merge and format logic."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from backend.data.understanding import (
    BusinessUnderstanding,
    BusinessUnderstandingInput,
    _json_to_themed_prompts,
    format_understanding_for_prompt,
    merge_business_understanding_data,
)


def _make_input(**kwargs: Any) -> BusinessUnderstandingInput:
    """Create a BusinessUnderstandingInput with only the specified fields."""
    return BusinessUnderstandingInput.model_validate(kwargs)


# ─── merge_business_understanding_data: list fields ─────────────────────


def test_merge_list_field_appends_new_items_once_in_order():
    existing = {"business": {"pain_points": ["a", {"legacy": 1}, "b"]}}

    result = merge_business_understanding_data(
        existing, _make_input(pain_points=["b", "c", "a", "c", "d"])
    )

    assert result["business"]["pain_points"] == ["a", {"legacy": 1}, "b", "c", "d"]


def test_merge_list_field_does_not_compare_each_item_with_every_stored_one():
    comparisons = 0

    class CountingStr(str):
        def __eq__(self, other: object) -> bool:
            nonlocal comparisons
            comparisons += 1
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    item_count = 2_000
    stored = [CountingStr(f"stored {i}") for i in range(item_count)]
    new = [f"new {i}" for i in range(item_count)]

    result = merge_business_understanding_data(
        {"business": {"pain_points": stored}}, _make_input(pain_points=new)
    )

    assert comparisons < item_count
    assert result["business"]["pain_points"] == [*stored, *new]


# ─── merge_business_understanding_data: themed prompts ─────────────────


def test_merge_themed_prompts_overwrites_existing():
    """New themed prompts should fully replace existing ones (not merge)."""
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": {
            "Learn": ["Old learn prompt"],
            "Create": ["Old create prompt"],
        },
    }
    new_prompts = {
        "Automate": ["Schedule daily reports", "Set up email alerts"],
        "Organize": ["Sort inbox by priority"],
    }
    input_data = _make_input(suggested_prompts=new_prompts)

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == new_prompts


def test_merge_themed_prompts_none_preserves_existing():
    """When input has suggested_prompts=None, existing themed prompts are preserved."""
    existing_prompts = {
        "Learn": ["How to automate?"],
        "Create": ["Build a chatbot"],
    }
    existing = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": existing_prompts,
    }
    input_data = _make_input(industry="Finance")

    result = merge_business_understanding_data(existing, input_data)

    assert result["suggested_prompts"] == existing_prompts
    assert result["business"]["industry"] == "Finance"


# ─── from_db: themed prompts deserialization ───────────────────────────


def test_from_db_themed_prompts():
    """from_db correctly deserializes a themed dict for suggested_prompts."""
    themed = {
        "Learn": ["What can I automate?"],
        "Create": ["Build a workflow"],
    }
    db_record = MagicMock()
    db_record.id = "test-id"
    db_record.userId = "user-1"
    db_record.createdAt = datetime.now(tz=timezone.utc)
    db_record.updatedAt = datetime.now(tz=timezone.utc)
    db_record.data = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": themed,
    }

    result = BusinessUnderstanding.from_db(db_record)

    assert result.suggested_prompts == themed


def test_from_db_legacy_list_prompts_preserved_under_general():
    """from_db preserves legacy list[str] prompts under a 'General' key."""
    db_record = MagicMock()
    db_record.id = "test-id"
    db_record.userId = "user-1"
    db_record.createdAt = datetime.now(tz=timezone.utc)
    db_record.updatedAt = datetime.now(tz=timezone.utc)
    db_record.data = {
        "name": "Alice",
        "business": {"industry": "Tech", "version": 1},
        "suggested_prompts": ["Old prompt 1", "Old prompt 2"],
    }

    result = BusinessUnderstanding.from_db(db_record)

    assert result.suggested_prompts == {"General": ["Old prompt 1", "Old prompt 2"]}


# ─── _json_to_themed_prompts helper ───────────────────────────────────


def test_json_to_themed_prompts_with_dict():
    value = {"Learn": ["a", "b"], "Create": ["c"]}
    assert _json_to_themed_prompts(value) == {"Learn": ["a", "b"], "Create": ["c"]}


def test_json_to_themed_prompts_with_list_returns_general():
    assert _json_to_themed_prompts(["a", "b"]) == {"General": ["a", "b"]}


def test_json_to_themed_prompts_with_none_returns_empty():
    assert _json_to_themed_prompts(None) == {}


# ─── format_understanding_for_prompt: excludes themed prompts ──────────


def test_format_understanding_excludes_themed_prompts():
    """Themed suggested_prompts are UI-only and must NOT appear in the system prompt."""
    understanding = BusinessUnderstanding(
        id="test-id",
        user_id="user-1",
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
        user_name="Alice",
        industry="Technology",
        suggested_prompts={
            "Learn": ["Automate reports"],
            "Create": ["Set up alerts", "Track KPIs"],
        },
    )

    formatted = format_understanding_for_prompt(understanding)

    assert "Alice" in formatted
    assert "Technology" in formatted
    assert "suggested_prompts" not in formatted
    assert "Automate reports" not in formatted
    assert "Set up alerts" not in formatted
    assert "Track KPIs" not in formatted


# ─── format_understanding_for_prompt: newest items only ─────────────────


def _profile(**lists: list[str]) -> BusinessUnderstanding:
    now = datetime.now(tz=timezone.utc)
    return BusinessUnderstanding(
        id="test-id", user_id="user-1", created_at=now, updated_at=now, **lists
    )


def test_format_shows_each_lists_newest_items_and_how_many_it_leaves_out():
    pain_points = [f"pain-{i:03d}" for i in range(57)]
    goals = [f"goal-{i:03d}" for i in range(23)]

    formatted = format_understanding_for_prompt(
        _profile(
            pain_points=pain_points, automation_goals=goals, current_software=["Slack"]
        )
    )

    assert [p for p in pain_points if p in formatted] == pain_points[-20:]
    assert [g for g in goals if g in formatted] == goals[-20:]
    assert (
        '(and 37 more: `tool:read_understanding` with field "pain_points")' in formatted
    )
    assert (
        '\n- (and 3 more: `tool:read_understanding` with field "automation_goals")'
        in formatted
    )
    assert "Current Software: Slack\n" in f"{formatted}\n"


def test_format_cuts_a_long_item_at_its_first_2000_characters():
    formatted = format_understanding_for_prompt(
        _profile(pain_points=["a" * 1999 + "b" + "c" * 500])
    )

    assert "a" * 1999 + "b…" in formatted
    assert "c" not in formatted.split("Pain Points: ")[1]
