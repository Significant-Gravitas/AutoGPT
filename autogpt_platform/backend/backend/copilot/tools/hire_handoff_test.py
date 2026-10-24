"""A hire proposed for a task the user already described hires AND hands it
off on one approval (SECRT-3138).

The principle under test, end to end through the real tools: ``hire_expert``
with a task offers two one-time ids, "hire and hand off" and "hire only";
confirming the first hires exactly the previewed template and starts the task
in a thread run by the new expert, and says where; confirming the second only
hires. Either one burns the other, so the user's single answer can never hire
twice. Only the queue, the session store and the roster are faked.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatMessage
from backend.copilot.sdk.session_waiter import SessionResult

from ._test_data import make_session
from .confirm_expert_change import ConfirmExpertChangeTool
from .expert_proposal import ExpertChangeProposal, proposal_key
from .hire_expert import HireExpertTool
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangePreview,
    ExpertChangeProposedResponse,
)

_USER = "user-hire-handoff"
_TITLE = "the Printify order pipeline"
_BRIEF = (
    "Own the Printify order pipeline: move new Etsy orders into Printify and "
    "sync tracking numbers back to Etsy every morning."
)


class _FakeRedis:
    """The setex/get/delete/set-NX surface the proposal store uses."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def delete(self, *keys):
        return sum(1 for key in keys if self.store.pop(key, None) is not None)


def _template():
    return SimpleNamespace(
        id="tpl-james",
        name="James",
        role="Operations",
        tagline="Keeps orders, stock and suppliers moving.",
        identity="You run operations.",
        boundaries="You never pay a supplier yourself.",
        voice_preferences="Plain and brief.",
        avatar_url=None,
        color="amber-300",
    )


def _hired():
    return SimpleNamespace(
        id="exp-james",
        name="James",
        role="Operations",
        tagline="Keeps orders, stock and suppliers moving.",
        identity="You run operations.",
        boundaries="You never pay a supplier yourself.",
        voice_preferences="Plain and brief.",
        weekly_budget=None,
        avatar_url=None,
        color="amber-300",
        is_archived=False,
        schedules_paused_at=None,
        autopilot_mode=None,
    )


@contextmanager
def _env(*, turn_outcome: str = "running"):
    db = MagicMock()
    db.list_templates = AsyncMock(return_value=[_template()])
    db.count_active_experts = AsyncMock(return_value=0)
    db.hire_expert = AsyncMock(return_value=SimpleNamespace(expert=_hired()))
    db.get_expert = AsyncMock(return_value=_hired())
    db.list_experts = AsyncMock(return_value=[_hired()])
    redis = AsyncMock(return_value=_FakeRedis())
    turn = AsyncMock(return_value=(turn_outcome, SessionResult()))
    created = []

    async def create_session(user_id, **kwargs):
        created.append(kwargs)
        return SimpleNamespace(session_id=f"thread-{len(created)}")

    with (
        patch("backend.copilot.tools.hire_expert.experts_db", return_value=db),
        patch("backend.copilot.tools.expert_proposal.experts_db", return_value=db),
        patch("backend.copilot.tools.expert_delegation.experts_db", return_value=db),
        patch("backend.copilot.tools.hire_expert.get_redis_async", redis),
        patch("backend.copilot.tools.confirm_expert_change.get_redis_async", redis),
        patch(
            "backend.copilot.tools.delegate_to_expert.create_chat_session",
            create_session,
        ),
        patch(
            "backend.copilot.tools.delegate_to_expert.run_copilot_turn_via_queue", turn
        ),
        patch(
            "backend.copilot.tools.delegate_to_expert.discard_unused_sub_session",
            AsyncMock(return_value=turn_outcome == "rejected_concurrent_turn_cap"),
        ),
        patch(
            "backend.copilot.tools.delegate_to_expert.build_spawn_state_note",
            AsyncMock(return_value=""),
        ),
    ):
        yield SimpleNamespace(db=db, turn=turn, created=created)


def _approve(session):
    session.messages.append(
        ChatMessage(role="user", content="approved", sequence=len(session.messages))
    )
    return session


async def _hire(session, **kwargs):
    return await HireExpertTool()._execute(user_id=_USER, session=session, **kwargs)


async def _confirm(session, confirmation_id):
    return await ConfirmExpertChangeTool()._execute(
        user_id=_USER, session=session, confirmation_id=confirmation_id
    )


async def _propose_with_task(session) -> ExpertChangeProposedResponse:
    preview = await _hire(
        session,
        template_id="tpl-james",
        handoff_title=_TITLE,
        handoff_task=_BRIEF,
    )
    assert isinstance(preview, ExpertChangeProposedResponse)
    return preview


class TestHireAndHandOffInOneApproval:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_card_offers_hire_and_hand_off_beside_hire_only(self):
        with _env() as env:
            preview = await _propose_with_task(make_session(_USER))
        assert preview.handoff is not None
        assert preview.handoff.title == _TITLE
        assert preview.handoff.brief == _BRIEF
        assert preview.handoff_confirmation_id
        assert preview.handoff_confirmation_id != preview.confirmation_id
        env.db.hire_expert.assert_not_called()
        env.turn.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_approving_the_hand_off_hires_and_starts_the_task(self):
        with _env() as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            assert preview.handoff_confirmation_id
            resp = await _confirm(_approve(session), preview.handoff_confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        env.db.hire_expert.assert_awaited_once_with(
            _USER, "tpl-james", "James", "copilot"
        )
        # The task went to the new hire, in a thread this chat delegated.
        assert env.created == [
            {
                "dry_run": False,
                "llm_auth_provider": "platform",
                "llm_credential_id": None,
                "expert_id": "exp-james",
                "delegated_by_expert_id": None,
                "delegated_by_session_id": session.session_id,
                "origin": "interactive",
                "inherited_autopilot_mode": env.created[0]["inherited_autopilot_mode"],
            }
        ]
        turn = env.turn.await_args.kwargs
        assert turn["session_id"] == "thread-1"
        assert _BRIEF in turn["message"]
        assert turn["timeout"] == 0
        # The card and Otto both learn where the work is happening.
        assert resp.handoff is not None
        assert resp.handoff.status == "started"
        assert resp.handoff.title == _TITLE
        assert resp.handoff.sub_session_id == "thread-1"
        assert resp.handoff.link == "/copilot?sessionId=thread-1"
        assert "/copilot?sessionId=thread-1" in resp.message
        assert "ask" in resp.message.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_only_hires_without_handing_anything_off(self):
        with _env() as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            resp = await _confirm(_approve(session), preview.confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        env.db.hire_expert.assert_awaited_once()
        env.turn.assert_not_called()
        assert resp.handoff is None
        assert "hire only" in resp.message.lower()
        assert _TITLE in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_one_answer_never_hires_twice(self):
        """Both ids name the same hire, so confirming one burns the other."""
        with _env() as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            assert preview.handoff_confirmation_id
            first = await _confirm(_approve(session), preview.confirmation_id)
            second = await _confirm(_approve(session), preview.handoff_confirmation_id)

        assert isinstance(first, ExpertChangeAppliedResponse)
        assert isinstance(second, ErrorResponse)
        assert "already confirmed" in second.message
        assert env.db.hire_expert.await_count == 1
        env.turn.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_hand_off_that_cannot_start_still_reports_the_hire(self):
        with _env(turn_outcome="rejected_concurrent_turn_cap") as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            assert preview.handoff_confirmation_id
            resp = await _confirm(_approve(session), preview.handoff_confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        env.db.hire_expert.assert_awaited_once()
        assert resp.handoff is not None
        assert resp.handoff.status == "failed"
        assert resp.handoff.link is None
        # Already approved on the card: retry it, don't ask again in prose.
        assert "delegate_to_expert" in resp.message
        assert "exp-james" in resp.message


class TestHandOffTaskValidation:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_task_without_a_title_is_refused(self):
        with _env():
            resp = await _hire(
                make_session(_USER), template_id="tpl-james", handoff_task=_BRIEF
            )
        assert isinstance(resp, ErrorResponse)
        assert "handoff_title" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_title_without_a_task_is_refused(self):
        with _env():
            resp = await _hire(
                make_session(_USER), template_id="tpl-james", handoff_title=_TITLE
            )
        assert isinstance(resp, ErrorResponse)
        assert "handoff_task" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_title_is_one_line_so_it_fits_the_button(self):
        with _env():
            resp = await _hire(
                make_session(_USER),
                template_id="tpl-james",
                handoff_title="  the Printify\n\norder   pipeline ",
                handoff_task=_BRIEF,
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.handoff is not None
        assert resp.handoff.title == "the Printify order pipeline"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_overlong_title_is_refused(self):
        with _env():
            resp = await _hire(
                make_session(_USER),
                template_id="tpl-james",
                handoff_title="x" * 200,
                handoff_task=_BRIEF,
            )
        assert isinstance(resp, ErrorResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_hire_without_a_task_offers_one_id_as_before(self):
        with _env():
            resp = await _hire(make_session(_USER), template_id="tpl-james")
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.handoff is None
        assert resp.handoff_confirmation_id is None


class TestProposalsParkedBeforeTheDeploy:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_old_hire_proposal_still_confirms_as_a_plain_hire(self):
        """A proposal parked by the previous build has none of the hand-off
        fields; it must still confirm, as the hire it always was."""
        redis = _FakeRedis()
        session = make_session(_USER)
        redis.store[proposal_key("old-id")] = json.dumps(
            {
                "user_id": _USER,
                "session_id": session.session_id,
                "preview": {
                    "kind": "hire",
                    "name": "James",
                    "template_id": "tpl-james",
                },
                "expert_id": None,
                "expected_soul": None,
                "user_turn_watermark": -1,
            }
        )
        assert ExpertChangeProposal.model_validate_json(
            redis.store[proposal_key("old-id")]
        ).preview == ExpertChangePreview(
            kind="hire", name="James", template_id="tpl-james"
        )
        with _env() as env, patch(
            "backend.copilot.tools.confirm_expert_change.get_redis_async",
            AsyncMock(return_value=redis),
        ):
            resp = await _confirm(_approve(session), "old-id")
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.handoff is None
        env.db.hire_expert.assert_awaited_once()
        env.turn.assert_not_called()
