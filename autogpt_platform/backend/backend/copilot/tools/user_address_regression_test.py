"""SECRT-3161 regression: Otto and an expert called the user "he".

The model picked a pronoun because nothing it was given said how to refer to
the user: no rule in Otto's prompt or in the expert's protected rules, and the
task Otto handed the expert called the user "this user". These tests check
those inputs on the real code paths. They use only APIs that predate the fix,
so the same file runs against the code before it and fails on the behaviour,
not on an import.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert
from backend.copilot.expert_context import render_expert_identity_suffix
from backend.copilot.service import _CACHEABLE_SYSTEM_PROMPT

from . import delegate_to_expert_test as delegation_test
from .delegate_to_expert import DelegateToExpertTool

# The delegation tests' doubles: an in-memory roster, a recorded queue turn and
# fake session storage. Bound here so pytest finds them as this module's.
roster = delegation_test.roster
mock_turn = delegation_test.mock_turn
mock_sessions = delegation_test.mock_sessions
_session = delegation_test._session


def _maya_hired_before_the_rule() -> Expert:
    return Expert(
        id="expert-maya",
        name="Maya",
        avatar_url=None,
        role="Marketing",
        tagline=None,
        bio=None,
        skills=[],
        identity="You plan launches.",
        voice_preferences="Plain.",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def test_otto_is_told_not_to_guess_the_users_gender():
    talking = _CACHEABLE_SYSTEM_PROMPT.split("# Talking to the user", 1)[1]
    talking = talking.split("\n# ", 1)[0]
    assert "gender" in talking, "Otto's prompt says nothing about the user's gender"
    assert '"they"' in talking


def test_every_expert_is_told_not_to_guess_the_users_gender():
    suffix = render_expert_identity_suffix(_maya_hired_before_the_rule())
    rules = suffix.split("<protected_rules>", 1)[1].split("</protected_rules>", 1)[0]
    assert any("gender" in rule for rule in PROTECTED_SOUL_RULES)
    assert "gender" in rules, "no protected rule tells the expert about gender"


@pytest.mark.asyncio
async def test_the_task_otto_hands_an_expert_names_the_user(
    roster, mock_turn, mock_sessions, monkeypatch
):
    profile = MagicMock()
    profile.get_business_understanding = AsyncMock(
        return_value=MagicMock(user_name="Priya Raman")
    )
    # raising=False: the lookup seam is part of the fix, absent before it.
    monkeypatch.setattr(
        "backend.copilot.tools.expert_delegation.understanding_db",
        MagicMock(return_value=profile),
        raising=False,
    )

    await DelegateToExpertTool()._execute(
        user_id="alice",
        session=_session(expert_id=None),
        expert_id="expert-b",
        prompt="Plan the serum launch.",
        wait_for_result=0,
    )

    preamble = mock_turn.await_args.kwargs["message"].split("\n\n", 1)[0]
    assert "this user's team" not in preamble, preamble
    assert "Priya's team" in preamble, preamble
