"""Tests for the SDK turn's system-prompt assembly."""

import pytest

from backend.copilot.prompting import get_sdk_supplement

from .system_prompt import SystemPromptParts, assemble_system_prompt


def _parts(**overrides) -> SystemPromptParts:
    fields = {
        "base": "<base/>",
        "use_e2b": False,
        "expert_session": False,
        "delegation": "<delegation/>",
        "oversight": "<oversight/>",
        "team_building": "<team_building/>",
        "chat_platform": "<chat_platform/>",
        "graphiti": "<graphiti/>",
        "auto_mode": "<auto_mode/>",
        "expert_identity": "<expert/>",
    }
    return SystemPromptParts(**(fields | overrides))


def test_order_of_the_pieces():
    prompt = assemble_system_prompt(_parts(), building_suffix="<guide/>")

    assert prompt == (
        "<base/>"
        + get_sdk_supplement(use_e2b=False, expert_session=False)
        + "<delegation/><oversight/><team_building/><chat_platform/>"
        + "<graphiti/><auto_mode/><guide/><expert/>"
    )


def test_building_suffix_extends_the_prompt_without_the_guide():
    """A plain session's restart prompt is its launch prompt plus the guide."""
    parts = _parts(expert_identity="")
    launch = assemble_system_prompt(parts, building_suffix="")

    restart = assemble_system_prompt(parts, building_suffix="<guide/>")

    assert restart == launch + "<guide/>"


@pytest.mark.parametrize(
    "use_e2b, expert_session, expected",
    [(True, False, 1), (True, True, 0), (False, False, 0)],
)
def test_computer_note_only_for_a_plain_e2b_session(use_e2b, expert_session, expected):
    """An expert session's computer block rides its first user message, so
    the system prompt must not add a second one."""
    prompt = assemble_system_prompt(
        _parts(use_e2b=use_e2b, expert_session=expert_session), building_suffix=""
    )

    assert prompt.count("### Your computer") == expected
