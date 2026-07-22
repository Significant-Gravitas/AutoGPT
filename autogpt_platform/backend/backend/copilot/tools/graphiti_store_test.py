"""Tests for MemoryStoreTool."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.ingest import MAX_EPISODE_BODY_BYTES
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_store import MemoryStoreTool
from backend.copilot.tools.models import ErrorResponse, MemoryStoreResponse


def _make_session(session_id: str = "test-session") -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="test-user",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _org_session(
    session_id: str = "test-session",
    org_id: str | None = "org-1",
    team_id: str | None = None,
) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="test-user",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        organization_id=org_id,
        team_id=team_id,
    )


class TestMemoryStoreTool:
    """Tests for MemoryStoreTool._execute."""

    @pytest.mark.asyncio
    async def test_store_no_user_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        result = await tool._execute(
            user_id=None,
            session=session,
            name="pref",
            content="likes python",
        )

        assert isinstance(result, ErrorResponse)
        assert "Authentication required" in result.message
        assert result.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_store_feature_disabled_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "not enabled" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_name_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_content_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_both_name_and_content_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="",
                content="",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_success_enqueues_episode(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="user_prefers_python",
                content="The user prefers Python over JavaScript.",
                source_description="Direct statement",
            )

        assert isinstance(result, MemoryStoreResponse)
        assert result.memory_name == "user_prefers_python"
        assert "queued for storage" in result.message
        assert result.session_id == "test-session"

        mock_enqueue.assert_awaited_once()
        call_kwargs = mock_enqueue.await_args.kwargs
        assert call_kwargs["name"] == "user_prefers_python"
        assert call_kwargs["source_description"] == "Direct statement"
        assert call_kwargs["is_json"] is True
        envelope = json.loads(call_kwargs["episode_body"])
        assert envelope["content"] == "The user prefers Python over JavaScript."
        assert envelope["memory_kind"] == "fact"

    @pytest.mark.asyncio
    async def test_store_success_uses_default_source_description(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="some_fact",
                content="A fact worth remembering.",
            )

        assert isinstance(result, MemoryStoreResponse)
        mock_enqueue.assert_awaited_once()
        call_kwargs = mock_enqueue.await_args.kwargs
        assert call_kwargs["name"] == "some_fact"
        assert call_kwargs["source_description"] == "Conversation memory"
        assert call_kwargs["is_json"] is True
        envelope = json.loads(call_kwargs["episode_body"])
        assert envelope["content"] == "A fact worth remembering."

    @pytest.mark.asyncio
    async def test_store_invalid_source_kind_falls_back(self):
        """Invalid enum values should fall back to defaults, not crash."""
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="some_fact",
                content="A fact.",
                source_kind="INVALID_SOURCE",
                memory_kind="INVALID_KIND",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["source_kind"] == "user_asserted"
        assert envelope["memory_kind"] == "fact"

    @pytest.mark.asyncio
    async def test_store_valid_enum_values_preserved(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="rule_1",
                content="Always CC Sarah.",
                source_kind="user_asserted",
                memory_kind="rule",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["source_kind"] == "user_asserted"
        assert envelope["memory_kind"] == "rule"

    @pytest.mark.asyncio
    async def test_store_queue_full_returns_retryable_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "queue" in result.message.lower()
        assert "try again" in result.message.lower()

    @pytest.mark.asyncio
    async def test_store_oversized_content_returns_split_guidance_without_enqueueing(
        self,
    ):
        """An envelope over the 64KB ingest cap is a permanent rejection.
        The tool must tell the LLM to split the content — not surface the
        retryable queue-full message that invites an identical retry —
        and must never hand the oversized body to the ingest queue."""
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="giant_memory",
                content="x" * (MAX_EPISODE_BODY_BYTES + 1),
            )

        assert isinstance(result, ErrorResponse)
        assert "too large" in result.message.lower()
        assert "split" in result.message.lower()
        assert "queue" not in result.message.lower()
        assert "try again" not in result.message.lower()
        mock_enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_size_check_measures_full_envelope_not_just_content(self):
        """The cap applies to the serialized MemoryEnvelope (what is
        actually enqueued), so content just under the cap is still
        rejected once the envelope's metadata pushes it over."""
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="barely_oversized_memory",
                content="x" * (MAX_EPISODE_BODY_BYTES - 10),
            )

        assert isinstance(result, ErrorResponse)
        assert "too large" in result.message.lower()
        mock_enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_with_scope(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="project_note",
                content="CRM uses PostgreSQL.",
                scope="project:crm",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["scope"] == "project:crm"


class TestMemoryStoreTierGovernance:
    """Shared-tier writes route to the tier's group and land active or
    tentative per the writer's role + the org hold-buffer setting."""

    def _patches(
        self,
        *,
        is_org_admin=None,
        is_org_member=None,
        hold_buffer=None,
        resolve_store_team=None,
    ):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            )
        )
        enqueue = stack.enter_context(
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                new_callable=AsyncMock,
                return_value=True,
            )
        )
        if is_org_member is not None:
            stack.enter_context(
                patch(
                    "backend.copilot.tools.graphiti_store.is_org_member",
                    new_callable=AsyncMock,
                    return_value=is_org_member,
                )
            )
        if is_org_admin is not None:
            stack.enter_context(
                patch(
                    "backend.copilot.tools.graphiti_store.is_org_admin",
                    new_callable=AsyncMock,
                    return_value=is_org_admin,
                )
            )
        if hold_buffer is not None:
            stack.enter_context(
                patch(
                    "backend.copilot.tools.graphiti_store.hold_buffer_enabled",
                    new_callable=AsyncMock,
                    return_value=hold_buffer,
                )
            )
        if resolve_store_team is not None:
            stack.enter_context(
                patch(
                    "backend.copilot.tools.graphiti_store.resolve_store_team",
                    resolve_store_team,
                )
            )
        return stack, enqueue

    @pytest.mark.asyncio
    async def test_org_store_as_admin_lands_active(self) -> None:
        tool = MemoryStoreTool()
        stack, enqueue = self._patches(
            is_org_admin=True, is_org_member=True, hold_buffer=True
        )
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(),
                name="policy",
                content="Refunds within 30 days.",
                tier="org",
            )

        assert isinstance(result, MemoryStoreResponse)
        kwargs = enqueue.await_args.kwargs
        assert kwargs["group_id"] == "org_org-1"
        assert kwargs["edge_metadata"]["status"] == "active"
        envelope = json.loads(kwargs["episode_body"])
        assert envelope["status"] == "active"
        assert "queued for storage in org memory" in result.message

    @pytest.mark.asyncio
    async def test_org_store_non_member_rejected(self) -> None:
        # A revoked/stale org membership must be blocked at the write path.
        tool = MemoryStoreTool()
        stack, enqueue = self._patches(is_org_member=False, hold_buffer=True)
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(),
                name="policy",
                content="Refunds within 30 days.",
                tier="org",
            )

        assert isinstance(result, ErrorResponse)
        assert "not an active member" in result.message
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_org_store_as_member_lands_tentative(self) -> None:
        tool = MemoryStoreTool()
        stack, enqueue = self._patches(
            is_org_admin=False, is_org_member=True, hold_buffer=True
        )
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(),
                name="policy",
                content="Refunds within 30 days.",
                tier="org",
            )

        assert isinstance(result, MemoryStoreResponse)
        kwargs = enqueue.await_args.kwargs
        assert kwargs["group_id"] == "org_org-1"
        assert kwargs["edge_metadata"]["status"] == "tentative"
        envelope = json.loads(kwargs["episode_body"])
        assert envelope["status"] == "tentative"
        assert "pending admin review" in result.message

    @pytest.mark.asyncio
    async def test_org_store_member_active_when_hold_buffer_disabled(self) -> None:
        tool = MemoryStoreTool()
        stack, enqueue = self._patches(
            is_org_admin=False, is_org_member=True, hold_buffer=False
        )
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(),
                name="policy",
                content="Refunds within 30 days.",
                tier="org",
            )

        assert isinstance(result, MemoryStoreResponse)
        assert enqueue.await_args.kwargs["edge_metadata"]["status"] == "active"
        assert "pending admin review" not in result.message

    @pytest.mark.asyncio
    async def test_org_store_without_org_context_errors(self) -> None:
        tool = MemoryStoreTool()
        stack, enqueue = self._patches(is_org_admin=True, hold_buffer=True)
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(org_id=None),
                name="policy",
                content="Refunds within 30 days.",
                tier="org",
            )

        assert isinstance(result, ErrorResponse)
        assert "organization" in result.message.lower()
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_team_store_as_team_admin_lands_active(self) -> None:
        tool = MemoryStoreTool()
        from types import SimpleNamespace

        resolve = AsyncMock(return_value=SimpleNamespace(teamId="team-1", isAdmin=True))
        stack, enqueue = self._patches(hold_buffer=True, resolve_store_team=resolve)
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(team_id="team-1"),
                name="convention",
                content="Deploys go out Tuesdays.",
                tier="team",
            )

        assert isinstance(result, MemoryStoreResponse)
        kwargs = enqueue.await_args.kwargs
        assert kwargs["group_id"] == "team_team-1"
        assert kwargs["edge_metadata"]["status"] == "active"
        assert "queued for storage in team memory" in result.message

    @pytest.mark.asyncio
    async def test_team_store_as_non_admin_member_lands_tentative(self) -> None:
        tool = MemoryStoreTool()
        from types import SimpleNamespace

        resolve = AsyncMock(
            return_value=SimpleNamespace(teamId="team-1", isAdmin=False)
        )
        stack, enqueue = self._patches(hold_buffer=True, resolve_store_team=resolve)
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(team_id="team-1"),
                name="convention",
                content="Deploys go out Tuesdays.",
                tier="team",
            )

        assert isinstance(result, MemoryStoreResponse)
        assert enqueue.await_args.kwargs["edge_metadata"]["status"] == "tentative"
        assert "pending admin review" in result.message

    @pytest.mark.asyncio
    async def test_team_store_when_not_a_member_errors_clearly(self) -> None:
        from backend.copilot.graphiti.tiers import TierError

        tool = MemoryStoreTool()
        resolve = AsyncMock(
            side_effect=TierError(
                "You are not an active member of the specified team, so you "
                "cannot store to its team memory."
            )
        )
        stack, enqueue = self._patches(hold_buffer=True, resolve_store_team=resolve)
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(team_id="team-1"),
                name="convention",
                content="Deploys go out Tuesdays.",
                tier="team",
            )

        assert isinstance(result, ErrorResponse)
        assert "not an active member" in result.message
        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_personal_tier_default_unchanged(self) -> None:
        """Default personal store must not route to a shared group or stamp
        edge metadata — the pre-existing path is untouched."""
        tool = MemoryStoreTool()
        stack, enqueue = self._patches()
        with stack:
            result = await tool._execute(
                user_id="user-1",
                session=_org_session(org_id="org-1", team_id="team-1"),
                name="pref",
                content="likes python",
            )

        assert isinstance(result, MemoryStoreResponse)
        kwargs = enqueue.await_args.kwargs
        assert kwargs["group_id"] is None  # personal path
        assert kwargs["edge_metadata"] is None
        assert "queued for storage" in result.message
