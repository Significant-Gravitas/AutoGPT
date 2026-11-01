"""Expert routing and model-tier checks for bot chat turns."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import ExpertIdentity
from backend.copilot.offers import EntitlementUnavailable
from backend.util.exceptions import NotFoundError

from .chat import ensure_chat_session, list_user_experts, start_chat_turn
from .chat_experts import (
    ADVANCED_FALLBACK_NOTICE,
    SERVER_EXPERT_REFUSAL,
    require_turn_expert,
    resolve_turn_model,
    session_reusable,
)
from .models import BotChatRequest, BotExpert, Platform

CHAT = "backend.platform_linking.chat"
EXPERTS = "backend.platform_linking.chat_experts"


def _session(session_id: str = "s1", expert_id: str | None = None, **kw) -> MagicMock:
    session = MagicMock(session_id=session_id, expert_id=expert_id, **kw)
    session.metadata.llm_auth_provider = "platform"
    return session


def _experts_db(owns: bool = True, identities: list | None = None) -> MagicMock:
    return MagicMock(
        owns_active_expert=AsyncMock(return_value=owns),
        list_expert_identities=AsyncMock(return_value=identities or []),
    )


def _identity(expert_id: str, name: str, archived: bool = False) -> ExpertIdentity:
    return ExpertIdentity(
        id=expert_id, name=name, avatar_url=None, role="Builder", is_archived=archived
    )


class TestSessionReusable:
    @pytest.mark.asyncio
    async def test_plain_session_serves_plain_turn(self):
        assert await session_reusable(_session(), "o", None, True)

    @pytest.mark.asyncio
    async def test_plain_session_does_not_serve_an_expert_turn(self):
        assert not await session_reusable(_session(), "o", "e1", True)

    @pytest.mark.asyncio
    async def test_expert_session_never_reused_in_a_server(self):
        db = _experts_db()
        with patch(f"{EXPERTS}.experts_db", return_value=db):
            assert not await session_reusable(
                _session(expert_id="e1"), "o", None, False
            )
        db.owns_active_expert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_expert_starts_a_new_session(self):
        with patch(f"{EXPERTS}.experts_db", return_value=_experts_db()):
            assert not await session_reusable(_session(expert_id="e1"), "o", "e2", True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asked", [None, "e1"])
    async def test_same_or_unspecified_expert_reuses_while_hired(self, asked):
        db = _experts_db(owns=True)
        with patch(f"{EXPERTS}.experts_db", return_value=db):
            assert await session_reusable(_session(expert_id="e1"), "o", asked, True)
        db.owns_active_expert.assert_awaited_once_with("o", "e1")

    @pytest.mark.asyncio
    async def test_fired_expert_session_is_dropped(self):
        with patch(f"{EXPERTS}.experts_db", return_value=_experts_db(owns=False)):
            assert not await session_reusable(_session(expert_id="e1"), "o", None, True)


class TestRequireTurnExpert:
    @pytest.mark.asyncio
    async def test_no_expert_is_always_fine(self):
        await require_turn_expert("o", None, "guild-1")

    @pytest.mark.asyncio
    async def test_server_refuses_experts_without_a_lookup(self):
        db = _experts_db()
        with patch(f"{EXPERTS}.experts_db", return_value=db):
            with pytest.raises(NotFoundError, match=SERVER_EXPERT_REFUSAL):
                await require_turn_expert("o", "e1", "guild-1")
        db.owns_active_expert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unowned_expert_refused(self):
        with patch(f"{EXPERTS}.experts_db", return_value=_experts_db(owns=False)):
            with pytest.raises(NotFoundError, match="Expert not found"):
                await require_turn_expert("o", "e1", None)


class TestResolveTurnModel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [None, "standard"])
    async def test_non_advanced_passes_through(self, model):
        assert await resolve_turn_model("o", model, _session()) == (model, None)

    @pytest.mark.asyncio
    async def test_entitled_advanced_kept(self):
        with patch(
            f"{EXPERTS}.advanced_tier_entitled", new=AsyncMock(return_value=True)
        ):
            assert await resolve_turn_model("o", "advanced", _session()) == (
                "advanced",
                None,
            )

    @pytest.mark.asyncio
    async def test_unentitled_advanced_falls_back_with_notice(self):
        with patch(
            f"{EXPERTS}.advanced_tier_entitled", new=AsyncMock(return_value=False)
        ):
            assert await resolve_turn_model("o", "advanced", _session()) == (
                "standard",
                ADVANCED_FALLBACK_NOTICE,
            )

    @pytest.mark.asyncio
    async def test_lookup_failure_falls_back_rather_than_spending(self):
        with patch(
            f"{EXPERTS}.advanced_tier_entitled",
            new=AsyncMock(side_effect=EntitlementUnavailable("down")),
        ):
            model, notice = await resolve_turn_model("o", "advanced", _session())
        assert model == "standard" and notice == ADVANCED_FALLBACK_NOTICE

    @pytest.mark.asyncio
    async def test_microsoft_copilot_owns_its_model(self):
        session = _session()
        session.metadata.llm_auth_provider = "microsoft_365_copilot"
        check = AsyncMock()
        with patch(f"{EXPERTS}.advanced_tier_entitled", new=check):
            assert await resolve_turn_model("o", "advanced", session) == (
                "advanced",
                None,
            )
        check.assert_not_awaited()


class TestListUserExperts:
    @pytest.mark.asyncio
    async def test_dm_lists_live_hires_of_the_dm_owner(self):
        link_db = MagicMock(find_user_link_owner=AsyncMock(return_value="owner-1"))
        db = _experts_db(
            identities=[_identity("e1", "Bolt"), _identity("e2", "Gone", True)]
        )
        with (
            patch(f"{CHAT}.platform_linking_db", return_value=link_db),
            patch(f"{EXPERTS}.experts_db", return_value=db),
        ):
            experts = await list_user_experts(Platform.DISCORD, "pu1", None)
        assert experts == [BotExpert(id="e1", name="Bolt", role="Builder")]
        db.list_expert_identities.assert_awaited_once_with("owner-1")

    @pytest.mark.asyncio
    async def test_server_lists_none_but_still_needs_a_link(self):
        link_db = MagicMock(find_server_link_owner=AsyncMock(return_value="owner-1"))
        db = _experts_db(identities=[_identity("e1", "Bolt")])
        with (
            patch(f"{CHAT}.platform_linking_db", return_value=link_db),
            patch(f"{EXPERTS}.experts_db", return_value=db),
        ):
            assert await list_user_experts(Platform.DISCORD, "pu1", "guild-1") == []
        db.list_expert_identities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlinked_dm_raises(self):
        link_db = MagicMock(find_user_link_owner=AsyncMock(return_value=None))
        with patch(f"{CHAT}.platform_linking_db", return_value=link_db):
            with pytest.raises(NotFoundError):
                await list_user_experts(Platform.TELEGRAM, "pu1", None)


def _turn_patches(stack: ExitStack, *, cached, created, experts_db, entitled=True):
    link_db = MagicMock(
        find_user_link_owner=AsyncMock(return_value="owner-1"),
        find_server_link_owner=AsyncMock(return_value="owner-1"),
    )
    orgs = MagicMock(get_user_default_team=AsyncMock(return_value=("org", "team")))
    stack.enter_context(
        patch(f"{CHAT}.evaluate_turn_gate", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch(f"{CHAT}.platform_linking_db", return_value=link_db))
    stack.enter_context(patch(f"{CHAT}.orgs_db", return_value=orgs))
    stack.enter_context(
        patch(f"{CHAT}.get_chat_session", new=AsyncMock(return_value=cached))
    )
    create = stack.enter_context(
        patch(f"{CHAT}.create_chat_session", new=AsyncMock(return_value=created))
    )
    stack.enter_context(
        patch(
            f"{CHAT}.resolve_default_chat_route",
            new=AsyncMock(return_value=("platform", None)),
        )
    )
    stack.enter_context(
        patch(
            f"{CHAT}.append_and_save_message", new=AsyncMock(return_value=MagicMock())
        )
    )
    registry = stack.enter_context(patch(f"{CHAT}.stream_registry"))
    registry.create_session = AsyncMock()
    enqueue = stack.enter_context(
        patch(f"{CHAT}.enqueue_copilot_turn", new=AsyncMock())
    )
    stack.enter_context(patch(f"{EXPERTS}.experts_db", return_value=experts_db))
    stack.enter_context(
        patch(f"{EXPERTS}.advanced_tier_entitled", new=AsyncMock(return_value=entitled))
    )
    return create, enqueue


def _req(**kw) -> BotChatRequest:
    return BotChatRequest(
        platform=Platform.DISCORD, platform_user_id="pu1", message="hi", **kw
    )


class TestStartChatTurnExperts:
    @pytest.mark.asyncio
    async def test_expert_turn_creates_session_bound_to_expert(self):
        created = _session(
            "new", expert_id="e1", organization_id="p-org", team_id="p-team"
        )
        with ExitStack() as stack:
            create, enqueue = _turn_patches(
                stack, cached=None, created=created, experts_db=_experts_db()
            )
            handle = await start_chat_turn(_req(expert_id="e1"))
        assert handle.session_id == "new"
        assert create.await_args.kwargs["expert_id"] == "e1"
        assert enqueue.await_args.kwargs["organization_id"] == "p-org"
        assert enqueue.await_args.kwargs["team_id"] == "p-team"

    @pytest.mark.asyncio
    async def test_switching_expert_replaces_cached_session(self):
        cached = _session("old", expert_id="e1")
        created = _session("new", expert_id="e2")
        with ExitStack() as stack:
            create, _ = _turn_patches(
                stack, cached=cached, created=created, experts_db=_experts_db()
            )
            handle = await start_chat_turn(_req(expert_id="e2", session_id="old"))
        assert handle.session_id == "new"
        assert create.await_args.kwargs["expert_id"] == "e2"

    @pytest.mark.asyncio
    async def test_server_expert_turn_refused_before_any_session(self):
        with ExitStack() as stack:
            create, enqueue = _turn_patches(
                stack, cached=None, created=_session(), experts_db=_experts_db()
            )
            with pytest.raises(NotFoundError, match=SERVER_EXPERT_REFUSAL):
                await start_chat_turn(_req(expert_id="e1", platform_server_id="g1"))
        create.assert_not_awaited()
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unowned_expert_refused(self):
        with ExitStack() as stack:
            create, enqueue = _turn_patches(
                stack,
                cached=None,
                created=_session(),
                experts_db=_experts_db(owns=False),
            )
            with pytest.raises(NotFoundError):
                await start_chat_turn(_req(expert_id="e1"))
        create.assert_not_awaited()
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attachment_session_of_another_expert_refused(self):
        with ExitStack() as stack:
            _, enqueue = _turn_patches(
                stack,
                cached=_session("s", expert_id="e1"),
                created=_session(),
                experts_db=_experts_db(),
            )
            with pytest.raises(NotFoundError):
                await start_chat_turn(
                    _req(expert_id="e2", session_id="s", file_ids=["f1"])
                )
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entitled,model,notice",
        [(True, "advanced", None), (False, "standard", ADVANCED_FALLBACK_NOTICE)],
    )
    async def test_advanced_tier_checked_before_enqueue(self, entitled, model, notice):
        with ExitStack() as stack:
            _, enqueue = _turn_patches(
                stack,
                cached=None,
                created=_session("new"),
                experts_db=_experts_db(),
                entitled=entitled,
            )
            handle = await start_chat_turn(_req(model="advanced"))
        assert enqueue.await_args.kwargs["model"] == model
        assert handle.notice == notice

    @pytest.mark.asyncio
    async def test_no_model_leaves_server_default(self):
        with ExitStack() as stack:
            _, enqueue = _turn_patches(
                stack, cached=None, created=_session("new"), experts_db=_experts_db()
            )
            handle = await start_chat_turn(_req())
        assert enqueue.await_args.kwargs["model"] is None
        assert handle.notice is None


class TestEnsureChatSessionExperts:
    @pytest.mark.asyncio
    async def test_attachments_land_in_the_expert_session(self):
        created = _session("expert-new", expert_id="e1")
        with ExitStack() as stack:
            create, _ = _turn_patches(
                stack, cached=None, created=created, experts_db=_experts_db()
            )
            result = await ensure_chat_session(
                Platform.DISCORD, "pu1", None, None, expert_id="e1"
            )
        assert result.session_id == "expert-new"
        assert create.await_args.kwargs["expert_id"] == "e1"

    @pytest.mark.asyncio
    async def test_server_refuses_expert(self):
        with ExitStack() as stack:
            create, _ = _turn_patches(
                stack, cached=None, created=_session(), experts_db=_experts_db()
            )
            with pytest.raises(NotFoundError):
                await ensure_chat_session(
                    Platform.DISCORD, "pu1", "g1", None, expert_id="e1"
                )
        create.assert_not_awaited()
