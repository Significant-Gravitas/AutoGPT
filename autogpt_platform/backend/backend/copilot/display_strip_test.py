"""The display strip's tag list: frozen, and apart from the inbound sanitizer's."""

import pytest

from backend.copilot.expert_context import OWNED_BLOCK_TAGS
from backend.copilot.prompting import VOICE_TURN_PREFIX, VOICE_TURN_TAG
from backend.copilot.service import (
    ENV_CONTEXT_TAG,
    MEMORY_CONTEXT_TAG,
    SERVER_INJECTED_BLOCK_TAGS,
    SESSION_CONTEXT_TAG,
    SKILLS_CONTEXT_TAG,
    STORED_ROW_BLOCK_TAGS,
    USER_CONTEXT_TAG,
    strip_injected_context_for_display,
)

# Every tag the platform has written at the start of a stored user row, found
# with ``git log -S`` on each name. The strip as it stood before #14763 listed
# the first eleven: the first turn's context blocks, the skills-drift notice
# (which some early rows carry), the voice instruction since #14322, and the
# expert identity, workflows and team blocks. #14763 moved the expert blocks to
# one registry and added routines, the expert's computer and standing work.
TAGS_WRITTEN_TO_STORED_ROWS = (
    "user_context",
    "memory_context",
    "env_context",
    "budget_context",
    "session_context",
    "available_skills",
    "skills_update",
    "voice_turn",
    "expert_identity",
    "expert_workflows",
    "team_context",
    "routines",
    "expert_computer",
    "standing_work",
)

# Blocks that only ever go to the model, never into a stored row: the inbound
# sanitizer reserves them, the display strip has nothing to peel.
MODEL_ONLY_BLOCK_TAGS = {
    "seen_capabilities",
    "budget_status",
    "conversation_history",
    "builder_context",
    "user_follow_up",
}


@pytest.mark.parametrize("tag", TAGS_WRITTEN_TO_STORED_ROWS)
def test_a_stored_row_shows_without_any_tag_the_platform_ever_wrote(tag: str):
    row = f"<{tag}>\nserver text\n</{tag}>\n\nwhat the user typed"

    assert strip_injected_context_for_display(row) == "what the user typed"


def test_the_list_holds_every_tag_ever_written_and_nothing_else():
    assert sorted(STORED_ROW_BLOCK_TAGS) == sorted(TAGS_WRITTEN_TO_STORED_ROWS)


def test_every_block_written_into_a_first_row_today_is_on_the_list():
    """A new block that lands in the stored first row must join the frozen
    list the day it ships, or that row shows it as the user's own words."""
    written_today = {
        USER_CONTEXT_TAG,
        MEMORY_CONTEXT_TAG,
        ENV_CONTEXT_TAG,
        SESSION_CONTEXT_TAG,
        SKILLS_CONTEXT_TAG,
        VOICE_TURN_TAG,
        *OWNED_BLOCK_TAGS,
    }

    assert written_today <= set(STORED_ROW_BLOCK_TAGS)


def test_the_sanitizer_differs_from_the_strip_only_by_the_model_only_blocks():
    """The model reads old rows as trusted, so a user may forge none of their
    tags; it also reads blocks no row ever holds, which only the sanitizer
    needs."""
    stored, reserved = set(STORED_ROW_BLOCK_TAGS), set(SERVER_INJECTED_BLOCK_TAGS)

    assert stored <= reserved
    assert reserved - stored == MODEL_ONLY_BLOCK_TAGS


def test_a_leading_model_only_block_is_kept_like_any_unknown_one():
    """No stored row starts with one, so the strip treats it as text it does
    not own, and still peels the stored blocks after it."""
    row = (
        "<budget_status>\n$1 left\n</budget_status>\n\n"
        + VOICE_TURN_PREFIX
        + "what the user typed"
    )

    assert strip_injected_context_for_display(row) == (
        "<budget_status>\n$1 left\n</budget_status>\n\nwhat the user typed"
    )
