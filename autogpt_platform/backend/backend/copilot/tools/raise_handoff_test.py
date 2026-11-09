"""A new expert created for a task the user already described is created AND
handed the task on one approval, like a hire (SECRT-3401).

Through the real tools, end to end: ``raise_expert`` with a task parks the
"create and start" id beside the plain one; confirming it creates exactly the
previewed charter and starts the task in a thread run by the new expert, and
what Otto is told is worded for a created expert (created, not hired, and no
workflows installing). Only the queue, the session store and the roster are
faked.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor.utils import TurnPublishUnconfirmed
from backend.copilot.model import ChatMessage
from backend.copilot.sdk.session_waiter import SessionResult, TurnNotDispatched

from ._test_data import make_session
from .confirm_expert_change import ConfirmExpertChangeTool
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeProposedResponse,
)
from .raise_expert import RaiseExpertTool

_USER = "user-raise-handoff"
_TITLE = "the weekly SEO report"
_BRIEF = (
    "Every Monday, pull last week's search rankings for our ten core keywords "
    "and write a one-page summary of what moved and why."
)
_CHARTER = {
    "name": "Nadia",
    "role": "SEO & Content",
    "tagline": "Keeps the site ranking for what customers search.",
    "color": "teal-300",
    "about": "You own organic search: rankings, content gaps and fixes.",
    "boundaries": "You never publish a page yourself; you hand drafts over.",
}


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


def _created():
    return SimpleNamespace(
        id="exp-nadia",
        name="Nadia",
        role="SEO & Content",
        tagline=_CHARTER["tagline"],
        identity=_CHARTER["about"],
        boundaries=_CHARTER["boundaries"],
        voice_preferences="",
        weekly_budget=None,
        avatar_url=None,
        color="teal-300",
        is_archived=False,
        schedules_paused_at=None,
        autopilot_mode=None,
    )


@contextmanager
def _env(
    *,
    turn_outcome: str = "running",
    turn_error: Exception | None = None,
    refusal: str = "",
):
    created_expert = _created()
    db = MagicMock()
    db.count_active_experts = AsyncMock(return_value=0)
    db.count_raised_experts = AsyncMock(return_value=0)
    # Nobody on the team yet, so the duplicate-name check passes.
    db.list_experts = AsyncMock(return_value=[])
    db.create_raised_expert = AsyncMock(
        return_value=SimpleNamespace(expert=created_expert, failed_attachments=[])
    )
    db.get_expert = AsyncMock(return_value=created_expert)
    fake_redis = _FakeRedis()
    redis = AsyncMock(return_value=fake_redis)
    turn = AsyncMock(
        side_effect=turn_error,
        return_value=(turn_outcome, SessionResult(refusal=refusal)),
    )
    threads = []
    deleted = AsyncMock()

    async def create_session(user_id, **kwargs):
        threads.append(kwargs)
        return SimpleNamespace(session_id=f"thread-{len(threads)}")

    delegate = "backend.copilot.tools.delegate_to_expert"
    with (
        patch("backend.copilot.tools.raise_expert.experts_db", return_value=db),
        patch("backend.copilot.tools.expert_proposal.experts_db", return_value=db),
        patch("backend.copilot.tools.expert_delegation.experts_db", return_value=db),
        patch("backend.copilot.tools.expert_handoff_proposal.get_redis_async", redis),
        patch("backend.copilot.tools.confirm_expert_change.get_redis_async", redis),
        patch(f"{delegate}.create_chat_session", create_session),
        patch(f"{delegate}.run_copilot_turn_via_queue", turn),
        patch(
            f"{delegate}.discard_unused_sub_session",
            AsyncMock(return_value=turn_outcome == "refused"),
        ),
        patch(f"{delegate}.build_spawn_state_note", AsyncMock(return_value="")),
        patch(f"{delegate}.delete_chat_session", deleted),
    ):
        yield SimpleNamespace(db=db, turn=turn, threads=threads, deleted=deleted)


def _approve(session):
    session.messages.append(
        ChatMessage(role="user", content="approved", sequence=len(session.messages))
    )
    return session


async def _raise(session, **kwargs):
    return await RaiseExpertTool()._execute(
        user_id=_USER, session=session, **{**_CHARTER, **kwargs}
    )


async def _confirm(session, confirmation_id):
    return await ConfirmExpertChangeTool()._execute(
        user_id=_USER, session=session, confirmation_id=confirmation_id
    )


async def _propose_with_task(session) -> ExpertChangeProposedResponse:
    preview = await _raise(session, handoff_title=_TITLE, handoff_task=_BRIEF)
    assert isinstance(preview, ExpertChangeProposedResponse)
    assert preview.handoff_confirmation_id
    return preview


async def _create_and_start(session) -> ExpertChangeAppliedResponse:
    preview = await _propose_with_task(session)
    assert preview.handoff_confirmation_id
    resp = await _confirm(_approve(session), preview.handoff_confirmation_id)
    assert isinstance(resp, ExpertChangeAppliedResponse)
    return resp


class TestCreateAndStartInOneApproval:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_new_expert_for_a_task_parks_create_and_start(self):
        with _env() as env:
            preview = await _propose_with_task(make_session(_USER))
        assert preview.preview.kind == "raise"
        assert preview.handoff is not None
        assert preview.handoff.title == _TITLE
        assert preview.handoff.brief == _BRIEF
        assert preview.handoff_confirmation_id != preview.confirmation_id
        env.db.create_raised_expert.assert_not_called()
        env.turn.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_otto_is_told_the_card_has_two_buttons(self):
        with _env():
            preview = await _propose_with_task(make_session(_USER))
        message = preview.message.lower()
        assert "nothing created yet" in message
        assert "two buttons: create and start the task, or not now" in message
        assert "approve and decline" not in message
        assert "call tool:confirm_expert_change with handoff_confirmation_id" in (
            message
        )
        assert "only if they ask to create the expert without starting" in message
        assert "never ask in text whether to hand the task over" in message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_approving_creates_and_sends_the_task(self):
        with _env() as env:
            session = make_session(_USER)
            resp = await _create_and_start(session)

        env.db.create_raised_expert.assert_awaited_once()
        assert env.db.create_raised_expert.await_args.args[1] == "Nadia"
        assert [t["expert_id"] for t in env.threads] == ["exp-nadia"]
        assert env.threads[0]["delegated_by_session_id"] == session.session_id
        turn = env.turn.await_args.kwargs
        assert turn["session_id"] == "thread-1"
        assert turn["timeout"] == 0
        assert _BRIEF in turn["message"]
        assert resp.kind == "raise"
        assert resp.handoff is not None
        assert resp.handoff.status == "sent"
        assert resp.handoff.link == "/copilot?sessionId=thread-1"
        assert resp.handoff.retry is False
        # Created, not hired: no roster template, so nothing installs.
        assert resp.message.startswith("Nadia is created and ")
        assert "hired" not in resp.message
        assert "installing" not in resp.message
        assert "/copilot?sessionId=thread-1" in resp.message
        assert "don't ask whether to hand it over" in resp.message.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_first_task_opens_with_an_introduction_and_asks_first(self):
        """The thread is the user's first look at the new expert, and they
        approved the task on Otto's card, not the expert's plan for it."""
        with _env() as env:
            await _create_and_start(make_session(_USER))

        message = env.turn.await_args.kwargs["message"]
        assert message.startswith("[Delegated task from Otto")
        assert "open with one line introducing yourself" in message
        assert (
            "Don't run a workflow, create a schedule or send anything on this "
            "first turn" in message
        )
        assert message.index("introducing yourself") < message.index(_BRIEF)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_task_waiting_for_a_slot_is_queued(self):
        with _env(turn_outcome="queued_for_slot"):
            resp = await _create_and_start(make_session(_USER))

        assert resp.handoff is not None
        assert resp.handoff.status == "queued"
        assert resp.handoff.link == "/copilot?sessionId=thread-1"
        assert resp.message.startswith("Nadia is created and ")
        assert "started" not in resp.message
        assert "installing" not in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_send_that_failed_before_dispatch_offers_try_again(self):
        with _env(turn_error=TurnNotDispatched("redis blip")) as env:
            resp = await _create_and_start(make_session(_USER))

        assert resp.handoff is not None
        assert resp.handoff.status == "failed"
        assert resp.handoff.retry is True
        assert resp.handoff.link is None
        assert resp.message.startswith(
            f"Nadia is created and on the team, but {_TITLE} did not reach them"
        )
        assert "hired" not in resp.message
        assert "Try again on the card sends it" in resp.message
        # The card links no chat for a task that didn't start.
        assert "Never tell the user to open or look for a chat with Nadia" in (
            resp.message
        )
        env.deleted.assert_awaited_once_with("thread-1", _USER)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_refused_task_is_not_retried(self):
        with _env(turn_outcome="refused", refusal="This tree is out of budget."):
            resp = await _create_and_start(make_session(_USER))

        assert resp.handoff is not None
        assert resp.handoff.status == "failed"
        assert resp.handoff.retry is False
        assert "out of budget" in resp.message
        assert "don't ask them again" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_send_that_may_have_landed_is_reported_unconfirmed(self):
        with _env(turn_error=TurnPublishUnconfirmed("confirm lost")) as env:
            resp = await _create_and_start(make_session(_USER))

        assert resp.handoff is not None
        assert resp.handoff.status == "unconfirmed"
        assert resp.handoff.retry is False
        assert resp.handoff.sub_session_id == "thread-1"
        assert resp.message.startswith("Nadia is created. ")
        assert "tool:get_sub_session_result (sub_session_id: thread-1)" in (
            resp.message
        )
        env.deleted.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_plain_id_only_creates(self):
        """A user who asks in chat to create without starting the task."""
        with _env() as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            resp = await _confirm(_approve(session), preview.confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        env.db.create_raised_expert.assert_awaited_once()
        env.turn.assert_not_called()
        assert resp.handoff is None
        assert "The user chose create only" in resp.message
        assert _TITLE in resp.message
        # No task went anywhere, so the first chat is the card's button.
        assert "Chat with Nadia button" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_one_answer_never_creates_twice(self):
        with _env() as env:
            session = make_session(_USER)
            preview = await _propose_with_task(session)
            assert preview.handoff_confirmation_id
            first = await _confirm(_approve(session), preview.handoff_confirmation_id)
            second = await _confirm(_approve(session), preview.confirmation_id)

        assert isinstance(first, ExpertChangeAppliedResponse)
        assert isinstance(second, ErrorResponse)
        assert "already confirmed" in second.message
        assert env.db.create_raised_expert.await_count == 1
        assert env.turn.await_count == 1


class TestANewExpertWithNoTask:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_offers_one_id_as_before(self):
        with _env():
            resp = await _raise(make_session(_USER))
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.handoff is None
        assert resp.handoff_confirmation_id is None
        assert "Approve and Decline buttons" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_otto_points_at_the_chat_button_not_a_thread(self):
        """SECRT-3401: told only who joined, Otto told the user to open the
        new expert's thread, and no chat with them existed yet."""
        with _env() as env:
            session = make_session(_USER)
            preview = await _raise(session)
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(_approve(session), preview.confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        env.turn.assert_not_called()
        assert resp.handoff is None
        assert resp.message.startswith("Nadia is created and on the team.")
        assert "raised" not in resp.message
        assert "Nadia has no chat yet" in resp.message
        assert "Chat with Nadia button" in resp.message
        assert "Never say a thread or chat with them already exists" in resp.message


class TestAChatPlatformConversation:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_is_pointed_at_autogpt_not_at_a_card(self):
        """A chat-platform bot renders no card, so there is no button to
        point the user at there."""
        with _env():
            session = make_session(_USER)
            session.metadata.source_platform = "slack"
            preview = await _raise(session)
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(_approve(session), preview.confirmation_id)

        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert "Nadia has no chat yet" in resp.message
        assert "The user can start it in AutoGPT" in resp.message
        assert "button" not in resp.message


class TestHandOffTaskValidation:
    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        "fields, missing",
        [
            pytest.param({"handoff_task": _BRIEF}, "handoff_title", id="no-title"),
            pytest.param({"handoff_title": _TITLE}, "handoff_task", id="no-task"),
        ],
    )
    async def test_both_or_neither(self, fields, missing):
        with _env() as env:
            resp = await _raise(make_session(_USER), **fields)
        assert isinstance(resp, ErrorResponse)
        assert missing in resp.message
        env.db.count_active_experts.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param(
                {"handoff_title": "x" * 81, "handoff_task": _BRIEF}, id="title"
            ),
            pytest.param(
                {"handoff_title": _TITLE, "handoff_task": "x" * 4001}, id="task"
            ),
        ],
    )
    async def test_the_same_bounds_as_a_hire(self, fields):
        with _env():
            resp = await _raise(make_session(_USER), **fields)
        assert isinstance(resp, ErrorResponse)
        assert "Invalid task" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_title_is_one_line_so_it_fits_the_card(self):
        with _env():
            resp = await _raise(
                make_session(_USER),
                handoff_title="  the weekly\n\nSEO   report ",
                handoff_task=f"  {_BRIEF}\n",
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.handoff is not None
        assert resp.handoff.title == _TITLE
        assert resp.handoff.brief == _BRIEF

    def test_the_tool_schema_offers_the_task_fields(self):
        properties = RaiseExpertTool().parameters["properties"]
        assert "handoff_title" in properties
        assert "handoff_task" in properties
        assert "handoff_title" not in RaiseExpertTool().parameters["required"]
