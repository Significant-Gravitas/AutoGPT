"""Tests for the confirm-gated hire/raise flow.

The contract under test is the gate itself: a preview must never create an
expert, the confirmation_id must be single-use and bound to the Autopilot
session that produced it, and confirm must apply exactly what was previewed.
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
from backend.util.exceptions import ExpertNotFoundError

from ._test_data import make_session
from .confirm_expert_change import ConfirmExpertChangeTool
from .expert_proposal import ExpertChangeProposal, apply_proposal
from .hire_expert import HireExpertTool
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
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
    db.update_soul = AsyncMock(return_value=_created("Otto", "exp-2"))
    shared_redis = AsyncMock(return_value=redis or _FakeRedis())
    with (
        patch(f"{_HIRE_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_UPDATE_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_PROPOSAL_MODULE}.experts_db", MagicMock(return_value=db)),
        patch(f"{_HIRE_MODULE}.get_redis_async", shared_redis),
        patch(f"{_RAISE_MODULE}.get_redis_async", shared_redis),
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
    @pytest.mark.asyncio(loop_scope="session")
    async def test_rename_applies_immediately_merged_over_the_current_soul(self):
        with _env() as db:
            db.update_soul.return_value = _created("Nick", "exp-2")
            resp = await _update(make_session(_USER), expert_id="exp-2", name="Nick")
        assert isinstance(resp, ExpertChangeAppliedResponse)
        assert resp.kind == "update"
        assert resp.expert.name == "Nick"
        otto = _hired_otto()
        db.update_soul.assert_awaited_once_with(
            _USER,
            "exp-2",
            ExpertSoulUpdate(
                name="Nick",
                identity=otto.identity,
                boundaries=otto.boundaries,
                voice_preferences=otto.voice_preferences,
            ),
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_blank_boundaries_keep_the_stored_value(self):
        """Unlike ``voice_preferences``, boundaries has no documented
        empty-string clearing and a raise requires them — a whitespace-only
        edit must not silently wipe the existing boundaries."""
        with _env() as db:
            resp = await _update(
                make_session(_USER), expert_id="exp-2", boundaries="   "
            )
        assert isinstance(resp, ExpertChangeAppliedResponse)
        otto = _hired_otto()
        db.update_soul.assert_awaited_once_with(
            _USER,
            "exp-2",
            ExpertSoulUpdate(
                name=otto.name,
                identity=otto.identity,
                boundaries=otto.boundaries,
                voice_preferences=otto.voice_preferences,
            ),
        )

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
        db.update_soul.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expert_vanishing_mid_edit_is_reported(self):
        with _env() as db:
            db.update_soul.side_effect = ExpertNotFoundError("exp-2")
            resp = await _update(make_session(_USER), expert_id="exp-2", name="Nick")
        assert isinstance(resp, ErrorResponse)


class TestConfirm:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_confirm_hires_exactly_what_was_previewed(self):
        redis = _FakeRedis()
        with _env(redis=redis) as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout", name="Recon")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
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
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
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
    async def test_second_confirm_fails(self):
        with _env() as db:
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            await _confirm(session, confirmation_id=preview.confirmation_id)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        assert db.hire_expert.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expired_or_unknown_id_fails(self):
        with _env() as db:
            resp = await _confirm(make_session(_USER), confirmation_id="nope")
        assert isinstance(resp, ErrorResponse)
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
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        assert str(ACTIVE_EXPERT_LIMIT) in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_apply_time_template_loss_is_reported(self):
        with _env(hire_error=ExpertTemplateNotFoundError("tpl-scout")):
            session = make_session(_USER)
            preview = await _hire(session, template_id="tpl-scout")
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        assert "template" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_apply_time_lifetime_limit_is_reported(self):
        error = RaisedExpertLifetimeLimitExceededError(LIFETIME_RAISED_EXPERT_LIMIT)
        with _env(raise_error=error):
            session = make_session(_USER)
            preview = await _raise(session, **_CHARTER)
            assert isinstance(preview, ExpertChangeProposedResponse)
            resp = await _confirm(session, confirmation_id=preview.confirmation_id)
        assert isinstance(resp, ErrorResponse)
        assert str(LIFETIME_RAISED_EXPERT_LIMIT) in resp.message


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
        db.update_soul.assert_not_called()

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


class TestApplyProposalDispatch:
    """apply_proposal must be exhaustive over ExpertChangeKind — a preview
    whose kind is neither 'hire' nor 'raise' must be refused rather than
    silently falling through to _apply_raise, which would CREATE an expert
    instead of applying whatever that other kind was meant to mean."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unsupported_kind_is_refused_not_applied(self):
        with _env() as db:
            proposal = ExpertChangeProposal(
                user_id=_USER,
                session_id="s1",
                preview=ExpertChangePreview(kind="update", name="Nope"),
            )
            resp = await apply_proposal(_USER, "s1", proposal)
        assert isinstance(resp, ErrorResponse)
        db.hire_expert.assert_not_called()
        db.create_raised_expert.assert_not_called()
