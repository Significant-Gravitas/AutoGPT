"""Tests for the confirm-gated hire/raise/update flow.

The contract under test is the gate itself: a preview must never write, the
confirmation_id must be single-use and bound to the Autopilot session that
produced it, confirm must apply exactly what was previewed, and only a
session a human is actually driving may reach any of it.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.errors import (
    ACTIVE_EXPERT_LIMIT,
    LIFETIME_RAISED_EXPERT_LIMIT,
    ExpertLimitExceededError,
    ExpertTemplateNotFoundError,
    RaisedExpertLifetimeLimitExceededError,
)
from backend.api.features.experts.models import EXPERT_NAME_MAX_LENGTH, ExpertSoulUpdate
from backend.copilot.model import ChatMessage, ChatSessionMetadata
from backend.util.exceptions import ExpertNotFoundError, ExpertWriteNotReadableError

from ._test_data import make_session
from .confirm_expert_change import MAX_BATCH_CONFIRMATIONS, ConfirmExpertChangeTool
from .expert_proposal import ExpertChangeProposal, apply_proposal, proposal_key
from .hire_expert import HireExpertTool
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeBatchAppliedResponse,
    ExpertChangePreview,
    ExpertChangeProposedResponse,
)
from .raise_expert import RaiseExpertTool
from .update_expert import UpdateExpertTool

_USER = "user-expert-change"
_HIRE_MODULE = "backend.copilot.tools.hire_expert"
_RAISE_MODULE = "backend.copilot.tools.raise_expert"
_UPDATE_MODULE = "backend.copilot.tools.update_expert"
_PROPOSAL_MODULE = "backend.copilot.tools.expert_proposal"
_CONFIRM_MODULE = "backend.copilot.tools.confirm_expert_change"

_CHARTER = {
    "name": "Otto",
    "role": "Inbox triage",
    "color": "violet-300",
    "about": "You group the morning inbox and draft routine replies.",
    "boundaries": "You never send a reply yourself.",
}


class _FakeRedis:
    """Minimal async Redis: the setex/get/delete surface the tools use."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def get(self, key):
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


def _template(template_id: str = "tpl-scout"):
    return SimpleNamespace(
        id=template_id,
        name="Scout",
        role="Market research",
        identity="You track competitors.",
        boundaries="You never contact anyone outside the team.",
        voice_preferences="Short and factual.",
        avatar_url=None,
        color="sky-300",
    )


def _created(name: str = "Scout", expert_id: str = "exp-1"):
    return SimpleNamespace(
        id=expert_id,
        name=name,
        role="Market research",
        avatar_url=None,
        color="sky-300",
    )


def _hired_otto():
    """The existing team expert the update flow edits."""
    return SimpleNamespace(
        id="exp-2",
        name="Otto",
        role="Inbox triage",
        identity="You group the morning inbox.",
        boundaries="You never send a reply yourself.",
        voice_preferences="Plain sentences.",
        avatar_url=None,
        color="violet-300",
        weekly_budget=2000,
    )


@contextmanager
def _env(
    *,
    redis: _FakeRedis | None = None,
    templates: list | None = None,
    active_count: int = 0,
    raised_count: int = 0,
    hire_result=None,
    raise_result=None,
    hire_error: Exception | None = None,
    raise_error: Exception | None = None,
):
    db = MagicMock()
    db.list_templates = AsyncMock(return_value=templates or [_template()])
    db.count_active_experts = AsyncMock(return_value=active_count)
    db.count_raised_experts = AsyncMock(return_value=raised_count)
    db.hire_expert = AsyncMock(
        side_effect=hire_error,
        return_value=hire_result
        or SimpleNamespace(expert=_created(), failed_preloads=[]),
    )
    db.create_raised_expert = AsyncMock(
        side_effect=raise_error,
        return_value=raise_result
        or SimpleNamespace(expert=_created("Otto", "exp-2"), failed_attachments=[]),
    )
    db.get_expert = AsyncMock(return_value=_hired_otto())
    db.update_soul_if_current = AsyncMock(return_value=_created("Otto", "exp-2"))
    shared_redis = AsyncMock(return_value=redis or _FakeRedis())
    with (
        patch(f"{_HIRE_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_UPDATE_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_PROPOSAL_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_HIRE_MODULE}.get_redis_async", shared_redis),
        patch(f"{_RAISE_MODULE}.get_redis_async", shared_redis),
        patch(f"{_UPDATE_MODULE}.get_redis_async", shared_redis),
        patch(f"{_CONFIRM_MODULE}.get_redis_async", shared_redis),
    ):
        yield db


async def _hire(session, **kwargs):
    return await HireExpertTool()._execute(user_id=_USER, session=session, **kwargs)


async def _raise(session, **kwargs):
    return await RaiseExpertTool()._execute(user_id=_USER, session=session, **kwargs)


async def _confirm(session, **kwargs):
    return await ConfirmExpertChangeTool()._execute(
        user_id=_USER, session=session, **kwargs
    )


async def _update(session, **kwargs):
    return await UpdateExpertTool()._execute(user_id=_USER, session=session, **kwargs)


def _approve(session):
    """Record the user's "yes" — the human turn a confirm has to answer."""
    session.messages.append(
        ChatMessage(role="user", content="yes", sequence=len(session.messages))
    )
    return session


def _automation_session():
    """A session an AutoPilotBlock opened inside a graph run: no expert_id,
    but no human typing into it either."""
    session = make_session(_USER)
    session.metadata.origin = "automation"
    return session


def _legacy_session():
    """A session persisted before ``origin`` existed.

    Built by parsing metadata JSON that genuinely lacks the key, because the
    legacy state IS a deserialization default — assigning ``origin = None`` to
    a constructed model would pass even if the field still defaulted to
    ``"interactive"``, which is the bug this pins.
    """
    session = make_session(_USER)
    session.metadata = ChatSessionMetadata.model_validate_json('{"dry_run": false}')
    return session


class TestPreviewNeverWrites:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_preview_stores_proposal_and_creates_nothing(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            resp = await _hire(make_session(_USER), template_id="tpl-scout")
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.applied is False
        assert resp.preview.kind == "hire"
        assert resp.preview.name == "Scout"
        assert len(redis.store) == 1
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_preview_carries_the_whole_charter(self):
        with _env() as db:
            resp = await _raise(make_session(_USER), **_CHARTER)
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.applied is False
        assert resp.preview.about == _CHARTER["about"]
        assert resp.preview.boundaries == _CHARTER["boundaries"]
        assert resp.preview.color == _CHARTER["color"]
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_renames_the_template(self):
        with _env():
            resp = await _hire(
                make_session(_USER), template_id="tpl-scout", name="Recon"
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.name == "Recon"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_template_is_refused(self):
        with _env(templates=[]) as db:
            resp = await _hire(make_session(_USER), template_id="tpl-gone")
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_without_boundaries_is_refused(self):
        with _env() as db:
            resp = await _raise(
                make_session(_USER),
                **{**_CHARTER, "boundaries": "   "},
            )
        assert isinstance(resp, ErrorResponse)
        assert "boundaries" in resp.message
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_without_about_is_refused(self):
        with _env():
            resp = await _raise(make_session(_USER), **{**_CHARTER, "about": ""})
        assert isinstance(resp, ErrorResponse)
        assert "charter" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_rejects_an_invalid_color(self):
        """``color`` is free-text model output, not a constrained enum on the
        wire — an out-of-palette value must be refused before any proposal
        is stored, not merely coerced or ignored."""
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            resp = await _raise(
                make_session(_USER), **{**_CHARTER, "color": "not-a-real-color"}
            )
        assert isinstance(resp, ErrorResponse)
        assert "color" in resp.message
        assert len(redis.store) == 0
        db.create_raised_expert.assert_not_called()


class TestPreviewTimeLimits:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_active_limit_surfaces_before_the_user_approves(self):
        with _env(active_count=ACTIVE_EXPERT_LIMIT) as db:
            resp = await _hire(make_session(_USER), template_id="tpl-scout")
        assert isinstance(resp, ErrorResponse)
        assert str(ACTIVE_EXPERT_LIMIT) in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_lifetime_raised_limit_surfaces_at_preview(self):
        with _env(raised_count=LIFETIME_RAISED_EXPERT_LIMIT) as db:
            resp = await _raise(make_session(_USER), **_CHARTER)
        assert isinstance(resp, ErrorResponse)
        assert str(LIFETIME_RAISED_EXPERT_LIMIT) in resp.message
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_lifetime_limit_does_not_block_a_hire(self):
        with _env(raised_count=LIFETIME_RAISED_EXPERT_LIMIT):
            resp = await _hire(make_session(_USER), template_id="tpl-scout")
        assert isinstance(resp, ExpertChangeProposedResponse)


class TestUpdate:
    """``identity``/``boundaries`` are injected into that expert's system
    prompt on every later turn and hand-written boundaries have no undo, so
    an update goes through the same preview + confirm gate as hire/raise
    rather than writing on the model's say-so."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_preview_merges_over_the_current_soul_and_writes_nothing(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            resp = await _update(make_session(_USER), expert_id="exp-2", name="Nick")
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.applied is False
        assert resp.preview.kind == "update"
        otto = _hired_otto()
        assert resp.preview.name == "Nick"
        assert resp.preview.about == otto.identity
        assert resp.preview.boundaries == otto.boundaries
        assert len(redis.store) == 1
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_writes_exactly_the_previewed_soul(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.kind == "update"
        otto = _hired_otto()
        db.update_soul_if_current.assert_awaited_once_with(
            _USER,
            "exp-2",
            ExpertSoulUpdate(
                name="Nick",
                identity=otto.identity,
                boundaries=otto.boundaries,
                voice_preferences=otto.voice_preferences,
            ),
            expected_name=otto.name,
            expected_identity=otto.identity,
            expected_voice_preferences=otto.voice_preferences,
            expected_boundaries=otto.boundaries,
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_blank_boundaries_keep_the_stored_value(self):
        """Unlike ``voice_preferences``, boundaries has no documented
        empty-string clearing and a raise requires them — a whitespace-only
        edit must not silently wipe the existing boundaries."""
        with _env():
            resp = await _update(
                make_session(_USER), expert_id="exp-2", boundaries="   "
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.boundaries == _hired_otto().boundaries

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_fields_is_refused(self):
        with _env() as db:
            resp = await _update(make_session(_USER), expert_id="exp-2")
        assert isinstance(resp, ErrorResponse)
        db.get_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_expert_is_refused(self):
        with _env() as db:
            db.get_expert.return_value = None
            resp = await _update(make_session(_USER), expert_id="nope", name="Nick")
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expert_vanishing_before_the_confirm_is_reported(self):
        with _env() as db:
            db.update_soul_if_current.side_effect = ExpertNotFoundError("exp-2")
            session = make_session(_USER)
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "gone" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_soul_edited_elsewhere_refuses_instead_of_reverting(self):
        """The confirm writes back the whole soul as the preview snapshotted
        it, so an unconditional write would silently revert an edit made from
        the team UI in between. The compare-and-set must refuse instead."""
        with _env() as db:
            db.update_soul_if_current.return_value = None
            session = make_session(_USER)
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "edited somewhere else" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_landed_edit_is_never_reported_as_nothing_changed(self):
        """The compare-and-set committed, then the row went before the
        read-back. Folding that into the stale error would tell the model to
        re-preview an edit that already applied."""
        with _env() as db:
            db.update_soul_if_current.side_effect = ExpertWriteNotReadableError("exp-2")
            session = make_session(_USER)
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "was saved" in resp.message
        assert "nothing was changed" not in resp.message
        assert "Do not re-preview" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_name_whitespace_is_collapsed(self):
        """``expert_context`` renders one roster line per teammate, and
        ``escape_prompt_xml_tags`` neutralizes angle brackets but not
        newlines — a name carrying one forges extra roster entries."""
        with _env():
            resp = await _update(
                make_session(_USER), expert_id="exp-2", name="  Nick\n\n- Mallory  "
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.name == "Nick - Mallory"


class TestConfirm:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_hires_exactly_what_was_previewed(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout", name="Recon")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.applied is True
        assert resp.kind == "hire"
        db.hire_expert.assert_awaited_once_with(_USER, "tpl-scout", "Recon")

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_raises_exactly_what_was_previewed(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _raise(session, **_CHARTER, weekly_budget=2000)
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.kind == "raise"
        assert resp.expert.name == "Otto"
        db.create_raised_expert.assert_awaited_once_with(
            _USER,
            _CHARTER["name"],
            _CHARTER["role"],
            None,
            color=_CHARTER["color"],
            about=_CHARTER["about"],
            boundaries=_CHARTER["boundaries"],
            weekly_budget=2000,
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_second_confirm_says_the_change_is_already_done(self):
        """A double "yes" is a normal thing for a user to say — it must read
        as "already done", not as the same message an expired preview gets."""
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            await _confirm(_approve(session), confirmation_id=preview.confirmation_id)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "already confirmed" in resp.message
        assert db.hire_expert.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expired_or_unknown_id_fails_and_discloses_the_window(self):
        with _env() as db:
            resp = await _confirm(make_session(_USER), confirmation_id="nope")
        assert isinstance(resp, ErrorResponse)
        assert "expired" in resp.message
        assert "15 minutes" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_proposal_from_another_session_is_refused(self):
        with _env() as db:
            preview = await _hire(make_session(_USER), template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                make_session(_USER), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "different chat" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_rejects_inline_field_values(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                session,
                confirmation_id=preview.confirmation_id,
                name="Someone else",
            )
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_apply_time_limit_error_is_typed_not_string_matched(self):
        with _env(hire_error=ExpertLimitExceededError(ACTIVE_EXPERT_LIMIT)):
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert str(ACTIVE_EXPERT_LIMIT) in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_apply_time_template_loss_is_reported(self):
        with _env(hire_error=ExpertTemplateNotFoundError("tpl-scout")):
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert "template" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_apply_time_lifetime_limit_is_reported(self):
        error = RaisedExpertLifetimeLimitExceededError(LIFETIME_RAISED_EXPERT_LIMIT)
        with _env(raise_error=error):
            session = make_session(_USER)
            preview = await _raise(session, **_CHARTER)
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        assert str(LIFETIME_RAISED_EXPERT_LIMIT) in resp.message


class TestConfirmNeedsAHumanTurn:
    """The preview hands the model the confirmation_id in its own output, so
    without a watermark the same assistant turn can preview and confirm and
    the seam the user is supposed to approve at never happens."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_in_the_same_turn_as_the_preview_is_refused(self):
        with _env() as db:
            session = _approve(make_session(_USER))
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        assert "not answered" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_confirm_in_the_same_turn_is_refused(self):
        with _env() as db:
            session = _approve(make_session(_USER))
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_premature_confirm_does_not_burn_the_proposal(self):
        """The user still gets to say yes: rejecting must not consume the id,
        or an eager model would cost them the preview they were reading."""
        with _env() as db:
            session = _approve(make_session(_USER))
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            await _confirm(session, confirmation_id=preview.confirmation_id)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        db.hire_expert.assert_awaited_once()


class TestExpertSessionsCannotStaff:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_is_refused_inside_an_expert_chat(self):
        with _env() as db:
            resp = await _hire(
                make_session(_USER, expert_id="exp-1"), template_id="tpl-scout"
            )
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_is_refused_inside_an_expert_chat(self):
        with _env() as db:
            resp = await _raise(make_session(_USER, expert_id="exp-1"), **_CHARTER)
        assert isinstance(resp, ErrorResponse)
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_is_refused_inside_an_expert_chat(self):
        with _env() as db:
            resp = await _update(
                make_session(_USER, expert_id="exp-1"),
                expert_id="exp-2",
                name="Nick",
            )
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_is_refused_inside_an_expert_chat(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                make_session(_USER, expert_id="exp-1"),
                confirmation_id=preview.confirmation_id,
            )
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()


class TestHireNameValidation:
    """hire_expert's rename field must be bounded and normalized the same
    way raise_expert already bounds its name — an unbounded name is
    interpolated into every expert session's <team_context> roster, where
    escape_prompt_xml_tags neutralizes angle brackets but not newlines."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_overlong_name_is_refused(self):
        with _env() as db:
            resp = await _hire(
                make_session(_USER),
                template_id="tpl-scout",
                name="x" * (EXPERT_NAME_MAX_LENGTH + 1),
            )
        assert isinstance(resp, ErrorResponse)
        assert "name" in resp.message.lower()
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_name_whitespace_is_collapsed(self):
        with _env():
            resp = await _hire(
                make_session(_USER),
                template_id="tpl-scout",
                name="  Bea\n\nNewman   Ops  ",
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.name == "Bea Newman Ops"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_omitted_name_falls_back_to_template_name(self):
        with _env():
            resp = await _hire(make_session(_USER), template_id="tpl-scout")
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.name == "Scout"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_collapses_name_and_role_whitespace(self):
        """Both fields land on the same roster line, so both need the same
        collapsing hire_expert already does — stripping alone leaves interior
        newlines that forge extra roster entries."""
        with _env():
            resp = await _raise(
                make_session(_USER),
                **{
                    **_CHARTER,
                    "name": " Otto\n- Mallory ",
                    "role": " Inbox\ntriage ",
                },
            )
        assert isinstance(resp, ExpertChangeProposedResponse)
        assert resp.preview.name == "Otto - Mallory"
        assert resp.preview.role == "Inbox triage"


class TestApplyProposalDispatch:
    """apply_proposal must be exhaustive over ExpertChangeKind and must
    refuse a proposal that lost the target it referred to, rather than
    falling through to a branch that CREATES an expert instead."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_without_a_target_expert_is_refused(self):
        with _env() as db:
            proposal = ExpertChangeProposal(
                user_id=_USER,
                session_id="s1",
                preview=ExpertChangePreview(kind="update", name="Nope"),
            )
            resp = await apply_proposal(_USER, "s1", proposal)
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_without_a_template_is_refused(self):
        with _env() as db:
            proposal = ExpertChangeProposal(
                user_id=_USER,
                session_id="s1",
                preview=ExpertChangePreview(kind="hire", name="Nope"),
            )
            resp = await apply_proposal(_USER, "s1", proposal)
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()


class TestPartialHire:
    """A hire whose workflows failed to install leaves an expert that cannot
    do part of its job — the message must say so, or the user only finds out
    when the work silently doesn't happen."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_failed_workflows_are_named_in_the_message(self):
        partial = SimpleNamespace(
            expert=_created(),
            failed_preloads=["Inbox triage", "Daily digest"],
        )
        with _env(hire_result=partial):
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.failed_workflows == ["Inbox triage", "Daily digest"]
        assert "Inbox triage" in resp.message
        assert "Daily digest" in resp.message
        assert "is hired and on the team." not in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_clean_hire_still_reads_as_a_clean_hire(self):
        with _env():
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.failed_workflows == []
        assert "could not be installed" not in resp.message


class TestLegacySessionsCannotStaff:
    """A session persisted before ``origin`` existed reads back as ``None``.

    The guard matches ``interactive`` positively, so an unknown origin is
    refused: it cannot prove a human is here, and the cost is that a chat
    older than this deploy needs a new one before it can staff. The
    AutoPilotBlock resume path deliberately takes the other side of the same
    unknown — refusing legacy sessions there would break live graphs.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_is_refused_in_a_legacy_session(self):
        with _env() as db:
            resp = await _hire(_legacy_session(), template_id="tpl-scout")
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_is_refused_in_a_legacy_session(self):
        with _env() as db:
            resp = await _raise(_legacy_session(), **_CHARTER)
        assert isinstance(resp, ErrorResponse)
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_is_refused_in_a_legacy_session(self):
        with _env() as db:
            resp = await _update(_legacy_session(), expert_id="exp-2", name="Nick")
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_fresh_session_is_never_born_legacy(self):
        """The guard's cost is bounded only because no row written today can
        look legacy — ``ChatSession.new`` always supplies a concrete origin."""
        assert make_session(_USER).metadata.origin == "interactive"


class TestAutomationSessionsCannotStaff:
    """``expert_id is None`` means "not an expert chat", not "a human is
    typing". An AutoPilotBlock session inside a graph run satisfies the
    former while its prompt may be assembled from untrusted upstream data,
    so the team tools gate on the interactive origin instead."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_hire_is_refused_in_a_block_origin_session(self):
        with _env() as db:
            resp = await _hire(_automation_session(), template_id="tpl-scout")
        assert isinstance(resp, ErrorResponse)
        assert "automation" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raise_is_refused_in_a_block_origin_session(self):
        with _env() as db:
            resp = await _raise(_automation_session(), **_CHARTER)
        assert isinstance(resp, ErrorResponse)
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_is_refused_in_a_block_origin_session(self):
        with _env() as db:
            resp = await _update(_automation_session(), expert_id="exp-2", name="Nick")
        assert isinstance(resp, ErrorResponse)
        db.update_soul_if_current.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_is_refused_in_a_block_origin_session(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            block_session = _automation_session()
            block_session.session_id = session.session_id
            resp = await _confirm(
                block_session, confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()


def _legacy_proposal_json(session_id: str) -> str:
    """A proposal exactly as pre-watermark code parked it in Redis.

    Raw JSON on purpose: the missing watermark IS a deserialization default,
    so constructing the model and clearing the field would still pass if the
    default went back to a number.
    """
    return (
        f'{{"user_id": "{_USER}", "session_id": "{session_id}", '
        '"preview": {"kind": "hire", "name": "Scout", '
        '"template_id": "tpl-scout"}}'
    )


class TestProposalsInFlightAcrossTheDeploy:
    """A proposal written before the watermark existed reads back without one.

    Defaulting it to a number would make any session with one sequenced user
    message clear the gate, so an absent watermark refuses instead.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_proposal_without_a_watermark_deserializes_to_none(self):
        proposal = ExpertChangeProposal.model_validate_json(
            _legacy_proposal_json("session-x")
        )
        assert proposal.user_turn_watermark is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_proposal_without_a_watermark_is_refused(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = _approve(make_session(_USER))
            redis.store[proposal_key("pre-deploy-id")] = _legacy_proposal_json(
                session.session_id
            )
            resp = await _confirm(session, confirmation_id="pre-deploy-id")
        assert isinstance(resp, ErrorResponse)
        assert "fresh preview" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_refusal_does_not_burn_the_proposal(self):
        """Same courtesy a premature confirm gets: the id survives so the
        re-preview the model is told to run is the only thing that replaces
        it."""
        redis = _FakeRedis()
        with _env(redis=redis):
            session = _approve(make_session(_USER))
            redis.store[proposal_key("pre-deploy-id")] = _legacy_proposal_json(
                session.session_id
            )
            await _confirm(session, confirmation_id="pre-deploy-id")
        assert proposal_key("pre-deploy-id") in redis.store


async def _two_previews(session) -> tuple[str, str]:
    """A hire and a raise the user is about to approve in one breath."""
    hire = await _hire(session, template_id="tpl-scout")
    raised = await _raise(session, **_CHARTER)
    assert isinstance(hire, ExpertChangeProposedResponse)
    assert isinstance(raised, ExpertChangeProposedResponse)
    return hire.confirmation_id, raised.confirmation_id


class TestBatchConfirm:
    """A user who approves three previews at once should cost one call, not
    three round-trips. Every id is still checked against the same gate, and
    each one reports its own outcome so a single bad id cannot silently void
    the changes beside it."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_all_valid_batch_applies_every_id(self):
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(_approve(session), confirmation_ids=[first, second])
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is True
        assert [result.confirmation_id for result in resp.results] == [first, second]
        assert all(result.outcome == "applied" for result in resp.results)
        assert [expert.name for expert in resp.experts] == ["Scout", "Otto"]
        assert "Scout" in resp.message and "Otto" in resp.message
        db.hire_expert.assert_awaited_once()
        db.create_raised_expert.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_one_bad_id_does_not_void_the_rest(self):
        """The whole point of the batch: an id that went stale between the
        preview and the "yes" must be reported, not turned into a refusal
        that drops the experts the user actually approved."""
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(
                _approve(session), confirmation_ids=[first, "nope", second]
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is True
        assert [result.outcome for result in resp.results] == [
            "applied",
            "failed",
            "applied",
        ]
        failed = resp.results[1]
        assert failed.confirmation_id == "nope"
        assert failed.reason == "expired"
        assert failed.error is not None
        assert "expired" in failed.error
        assert failed.expert is None
        assert [expert.name for expert in resp.experts] == ["Scout", "Otto"]
        assert db.hire_expert.await_count == 1
        assert db.create_raised_expert.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_replayed_id_reads_as_already_done_beside_a_fresh_one(self):
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            await _confirm(_approve(session), confirmation_id=first)
            resp = await _confirm(_approve(session), confirmation_ids=[first, second])
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        replayed = resp.results[0]
        assert replayed.outcome == "already_applied"
        assert replayed.reason == "already_applied"
        assert replayed.error is not None
        assert "already confirmed" in replayed.error
        assert resp.results[1].outcome == "applied"
        # The replay must not hire a second Scout.
        assert db.hire_expert.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_of_replays_reads_as_done_not_as_a_total_failure(self):
        """ "Nothing was applied" about changes that are all on the team is the
        worst thing this tool can say: the model repeats it, and the user is
        told their team is empty while looking at it."""
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            await _confirm(_approve(session), confirmation_ids=[first, second])
            resp = await _confirm(_approve(session), confirmation_ids=[first, second])
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is True
        assert [result.outcome for result in resp.results] == [
            "already_applied",
            "already_applied",
        ]
        assert "Nothing was applied" not in resp.message
        assert "already applied" in resp.message
        # The replay must not hire a second Scout or raise a second Otto.
        assert db.hire_expert.await_count == 1
        assert db.create_raised_expert.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_saved_but_unreadable_edit_is_not_reported_as_not_added(self):
        """``_applied_but_unreadable_error`` says the edit "was saved". The
        batch must agree with it instead of rendering the same change as a
        failure."""
        with _env() as db:
            db.update_soul_if_current.side_effect = ExpertWriteNotReadableError("exp-2")
            session = make_session(_USER)
            preview = await _update(session, expert_id="exp-2", name="Nick")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_ids=[preview.confirmation_id]
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is True
        assert resp.results[0].outcome == "already_applied"
        assert resp.results[0].reason == "applied_but_expert_gone"
        assert "Nothing was applied" not in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_replay_beside_a_failure_still_counts_as_landed(self):
        with _env():
            session = make_session(_USER)
            first, second = await _two_previews(session)
            await _confirm(_approve(session), confirmation_id=first)
            resp = await _confirm(
                _approve(session), confirmation_ids=[first, second, "nope"]
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert [result.outcome for result in resp.results] == [
            "already_applied",
            "applied",
            "failed",
        ]
        assert "2 of 3" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_id_from_another_chat_is_refused_inside_a_batch(self):
        with _env() as db:
            other = make_session(_USER)
            stranger = await _hire(other, template_id="tpl-scout")
            assert isinstance(stranger, ExpertChangeProposedResponse)
            session = make_session(_USER)
            mine = await _raise(session, **_CHARTER)
            assert isinstance(mine, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session),
                confirmation_ids=[stranger.confirmation_id, mine.confirmation_id],
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.results[0].outcome == "failed"
        assert resp.results[0].reason == "wrong_chat"
        assert resp.results[0].error is not None
        assert "different chat" in resp.results[0].error
        assert resp.results[1].outcome == "applied"
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_where_nothing_lands_says_so(self):
        with _env() as db:
            resp = await _confirm(
                _approve(make_session(_USER)), confirmation_ids=["nope", "also-nope"]
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is False
        assert resp.experts == []
        assert "Nothing was applied" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_cap_reached_mid_batch_is_reported_per_id(self):
        """``capacity_error`` only runs at preview time, so N previews that
        each fit can exceed the cap together. The creation transaction is the
        real enforcement point — the overflow must surface as that id's error
        rather than as an over-hired team."""
        with _env() as db:
            db.hire_expert.side_effect = [
                SimpleNamespace(expert=_created(), failed_preloads=[]),
                ExpertLimitExceededError(ACTIVE_EXPERT_LIMIT),
            ]
            session = make_session(_USER)
            first = await _hire(session, template_id="tpl-scout")
            second = await _hire(session, template_id="tpl-scout")
            assert isinstance(first, ExpertChangeProposedResponse)
            assert isinstance(second, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session),
                confirmation_ids=[first.confirmation_id, second.confirmation_id],
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.results[0].outcome == "applied"
        assert resp.results[1].outcome == "failed"
        assert resp.results[1].reason == "limit_reached"
        assert resp.results[1].error is not None
        assert str(ACTIVE_EXPERT_LIMIT) in resp.results[1].error
        assert len(resp.experts) == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_partial_hire_inside_a_batch_still_names_its_workflows(self):
        """The single-id path spells the failed workflows out in its message;
        folding N applies into one message must not drop that warning."""
        partial = SimpleNamespace(
            expert=_created(),
            failed_preloads=["Inbox triage"],
        )
        with _env(hire_result=partial):
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(_approve(session), confirmation_ids=[first, second])
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.results[0].failed_workflows == ["Inbox triage"]
        assert "Inbox triage" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_of_one_is_still_a_batch(self):
        """The response shape follows the parameter the model used, so a card
        built for confirmation_ids never has to handle two payload shapes."""
        with _env():
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_ids=[preview.confirmation_id]
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert len(resp.results) == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_single_confirmation_id_keeps_the_old_response_shape(self):
        """Backward compatibility: the single param must still return the
        applied response the existing card and prompts were written for."""
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(
                _approve(session), confirmation_id=preview.confirmation_id
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.kind == "hire"
        assert resp.expert.name == "Scout"
        db.hire_expert.assert_awaited_once()


class TestBatchParameterValidation:
    """Both parameters mean the model is guessing about what the user
    approved, and neither means it has nothing to apply — either way the
    right answer is to refuse before any proposal is consumed."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_supplying_both_parameters_is_refused(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(
                _approve(session),
                confirmation_id=first,
                confirmation_ids=[second],
            )
        assert isinstance(resp, ErrorResponse)
        assert "never both" in resp.message
        assert proposal_key(first) in redis.store
        assert proposal_key(second) in redis.store
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_null_confirmation_id_beside_a_batch_still_applies_it(self):
        """Filling both keys and nulling the unused one is a routine tool-call
        shape. It must read as "no single id", not crash out of the tool and
        leave the model unable to tell whether the ids were consumed."""
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(
                _approve(session),
                confirmation_id=None,
                confirmation_ids=[first, second],
            )
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert len(resp.results) == 2
        db.hire_expert.assert_awaited_once()
        db.create_raised_expert.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_null_confirmation_id_on_its_own_is_refused(self):
        with _env() as db:
            resp = await _confirm(_approve(make_session(_USER)), confirmation_id=None)
        assert isinstance(resp, ErrorResponse)
        assert "confirmation_ids" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.parametrize(
        "params",
        [
            {"confirmation_id": 7},
            {"confirmation_id": ["c-1"]},
            {"confirmation_ids": "c-1"},
            {"confirmation_ids": {"id": "c-1"}},
            {"confirmation_ids": ["c-1", None]},
            {"confirmation_ids": ["c-1", 7]},
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_id_of_the_wrong_shape_is_refused_not_raised(self, params):
        """Every malformed shape has to come back as the actionable refusal;
        a traceback becomes "an error occurred" and tells the model nothing
        about whether its approvals survived."""
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            first, _ = await _two_previews(session)
            resp = await _confirm(_approve(session), **params)
        assert isinstance(resp, ErrorResponse)
        assert str(MAX_BATCH_CONFIRMATIONS) in resp.message
        assert proposal_key(first) in redis.store
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_supplying_neither_parameter_is_refused(self):
        with _env() as db:
            resp = await _confirm(_approve(make_session(_USER)))
        assert isinstance(resp, ErrorResponse)
        assert "confirmation_ids" in resp.message
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_empty_id_list_reads_as_no_id_at_all(self):
        with _env() as db:
            resp = await _confirm(_approve(make_session(_USER)), confirmation_ids=[])
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_oversized_batch_is_refused_before_anything_is_consumed(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            first, _ = await _two_previews(session)
            resp = await _confirm(
                _approve(session),
                confirmation_ids=[first]
                + [f"filler-{i}" for i in range(MAX_BATCH_CONFIRMATIONS)],
            )
        assert isinstance(resp, ErrorResponse)
        assert str(MAX_BATCH_CONFIRMATIONS) in resp.message
        assert proposal_key(first) in redis.store
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_blank_id_in_the_batch_is_refused(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            first, _ = await _two_previews(session)
            resp = await _confirm(_approve(session), confirmation_ids=[first, "   "])
        assert isinstance(resp, ErrorResponse)
        assert proposal_key(first) in redis.store
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_still_refuses_inline_field_values(self):
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            resp = await _confirm(
                _approve(session),
                confirmation_ids=[first, second],
                name="Someone else",
            )
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_is_refused_in_a_block_origin_session(self):
        with _env() as db:
            session = make_session(_USER)
            first, second = await _two_previews(session)
            block_session = _automation_session()
            block_session.session_id = session.session_id
            resp = await _confirm(block_session, confirmation_ids=[first, second])
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_batch_confirmed_in_the_same_turn_as_its_previews_is_refused(self):
        """The watermark gate is per id, so batching must not become the way
        the model previews and confirms inside one uninterrupted turn."""
        with _env() as db:
            session = _approve(make_session(_USER))
            first, second = await _two_previews(session)
            resp = await _confirm(session, confirmation_ids=[first, second])
        assert isinstance(resp, ExpertChangeBatchAppliedResponse)
        assert resp.applied is False
        assert all(result.outcome == "failed" for result in resp.results)
        assert all(
            result.error is not None and "not answered" in result.error
            for result in resp.results
        )
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()
