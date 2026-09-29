"""Tests for the memory_forget tools.

The retraction itself (Cypher, per-uuid failures) is pinned in
``graphiti/recall_forget_test.py``; here the tools are exercised with that
layer either mocked or driven through a mock driver.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge

from backend.copilot.graphiti import scope_lock
from backend.copilot.graphiti.memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
)
from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_forget import (
    _MAX_FAILURE_DETAIL,
    MemoryForgetConfirmTool,
    MemoryForgetSearchTool,
    _build_confirm_message,
)
from backend.copilot.tools.models import (
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
)

_MODULE = "backend.copilot.tools.graphiti_forget"


async def _enabled(_user_id: str) -> bool:
    return True


@pytest.fixture(autouse=True)
def lock_redis(mocker) -> FakeRedis:
    """The real ``retract`` takes the graph's write lock; keep it in memory."""
    redis = FakeRedis()
    mocker.patch.object(
        scope_lock, "get_redis_async", mocker.AsyncMock(return_value=redis)
    )
    return redis


def _mock_driver(*results) -> AsyncMock:
    """A FalkorDB driver whose queries return ``results`` in order, for
    driving the real ``retract`` from the confirm tool, after the read of
    pending dream records every forget makes first (none here)."""
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, [], None) for r in ([], *results)
    ]
    return driver


class TestExpertMemoryScope:
    @pytest.mark.asyncio
    async def test_forget_search_uses_expert_memory_group(self) -> None:
        search_facts = AsyncMock(return_value=[])
        session = ChatSession.new("user-abc", dry_run=False, expert_id="expert-1")
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.search_facts", search_facts),
        ):
            await MemoryForgetSearchTool()._execute(
                "user-abc", session, query="private fact"
            )

        search_facts.assert_awaited_once_with(
            MemoryScope.for_expert("user-abc", "expert-1"), "private fact", limit=10
        )

    @pytest.mark.asyncio
    async def test_forget_confirm_uses_expert_memory_group(self) -> None:
        retract = AsyncMock(return_value=ForgetResult())
        session = ChatSession.new("user-abc", dry_run=False, expert_id="expert-1")
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
        ):
            await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["private-edge"]
            )

        retract.assert_awaited_once_with(
            MemoryScope.for_expert("user-abc", "expert-1"),
            ["private-edge"],
            hard=False,
        )


class TestForgetSearchCandidates:
    @pytest.mark.asyncio
    async def test_candidates_carry_uuid_fact_and_validity(self) -> None:
        edge = EntityEdge(
            uuid="e1",
            group_id="user_user-abc",
            source_node_uuid="a",
            target_node_uuid="b",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            name="works_on",
            fact="Alice works on Atlas",
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.search_facts", AsyncMock(return_value=[edge])),
        ):
            response = await MemoryForgetSearchTool()._execute(
                "user-abc", session, query="atlas"
            )

        assert isinstance(response, MemoryForgetCandidatesResponse)
        assert response.candidates == [
            {
                "uuid": "e1",
                "fact": "Alice works on Atlas",
                "valid_from": "2025-01-01 00:00:00+00:00",
                "valid_to": "present",
            }
        ]


class TestForgetConfirmModes:
    @pytest.mark.asyncio
    async def test_hard_delete_asks_for_a_hard_retract(self) -> None:
        retract = AsyncMock(return_value=ForgetResult(deleted=["e1"]))
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"], hard_delete=True
            )

        assert retract.await_args is not None
        assert retract.await_args.kwargs == {"hard": True}
        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.message == "1 memory edge(s) permanently deleted."

    @pytest.mark.asyncio
    async def test_the_facts_derived_from_it_are_reported(self) -> None:
        """The user hears of the dream's facts the forget retracted too."""
        result = ForgetResult(deleted=["e1"], derived=["d1", "d2", "d3"])
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", AsyncMock(return_value=result)),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.derived_uuids == ["d1", "d2", "d3"]
        assert response.message == (
            "1 memory edge(s) retracted from memory; "
            "3 fact(s) derived from them retracted too."
        )

    @pytest.mark.asyncio
    async def test_unavailable_graph_is_an_error_response(self) -> None:
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", AsyncMock(side_effect=OSError("refused"))),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert not isinstance(response, MemoryForgetConfirmResponse)
        assert "temporarily unavailable" in response.message


class TestForgetFailuresAreActionable:
    """SECRT-2371: soft delete must not fail silently. Every failure — whether
    the query errored or matched nothing — has to carry a per-UUID reason so
    the model can act (retry, hard-delete, or tell the user) instead of seeing
    a bare "0 invalidated, N failed". Driven through the real ``retract``
    with a mock driver."""

    @pytest.mark.asyncio
    async def test_confirm_tool_reports_reasons_in_response(self) -> None:
        """End to end: a soft delete that matches nothing must return the
        per-UUID reason in both the structured `failures` field and the
        human-readable message — not a bare "0 invalidated, 1 failed"."""
        driver = _mock_driver([], [])  # the lookup finds nothing, nothing names it
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["missing-uuid"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == []
        assert [f.uuid for f in response.failures] == ["missing-uuid"]
        assert response.failed_uuids == ["missing-uuid"]
        assert response.failures[0].code == MemoryForgetFailureCode.NO_MATCH
        # The reason must reach the model-visible message, not just the count.
        assert response.failures[0].reason in response.message
        assert "missing-uuid" in response.message

    @pytest.mark.asyncio
    async def test_confirm_tool_mixed_batch_reports_both(self) -> None:
        """A batch where some UUIDs delete and some fail must co-populate
        `deleted_uuids` and `failures`, and the message must carry BOTH the
        success count and the per-UUID failure detail."""
        driver = _mock_driver(
            [{"uuid": "kept"}],  # lookup: only "kept" exists
            [],  # nothing names "gone" as a purged root
            [{"uuid": "kept"}],  # retract "kept"
            [],  # scrub its sentence
            [],  # find the entities to scrub (none)
            [],  # redact its episodes
            [],  # the cascade: no earlier try,
            [],  # no episode citing it,
            [],  # nothing derived from it
            [],
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["kept", "gone"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == ["kept"]
        assert [f.uuid for f in response.failures] == ["gone"]
        # Two-part message: success half + failure detail half.
        assert "1 memory edge(s) retracted from memory." in response.message
        assert "1 failed" in response.message
        assert "gone" in response.message

    @pytest.mark.asyncio
    async def test_confirm_tool_reports_a_failed_clean_up(self) -> None:
        """Retracted, but the episode redaction failed: the model is told the
        fact is forgotten and that the clean-up did not finish."""
        driver = _mock_driver(
            [{"uuid": "u1"}],  # lookup
            [{"uuid": "u1"}],  # retract
            [],  # scrub its sentence
            [],  # find the entities to scrub (none)
            RuntimeError("down"),  # redact its episodes
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["u1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == ["u1"]
        assert [f.code for f in response.failures] == [
            MemoryForgetFailureCode.CLEANUP_ERROR
        ]
        assert "no longer recalled" in response.message
        assert "RuntimeError: down" in response.message


class TestBuildConfirmMessage:
    """`_build_confirm_message` formatting: the no-failure early return, the
    two-part success+failure string, and the bounded detail (thread: a driver
    outage failing every UUID must not blow the tool output past its size
    threshold and lose all detail)."""

    def test_no_failures_returns_summary_only(self) -> None:
        message = _build_confirm_message(3, "retracted from memory", [])
        assert message == "3 memory edge(s) retracted from memory."

    def test_the_facts_derived_from_them_are_counted(self) -> None:
        message = _build_confirm_message(1, "permanently deleted", [], 3)
        assert message == (
            "1 memory edge(s) permanently deleted; "
            "3 fact(s) derived from them retracted too."
        )

    def test_edges_already_erased_are_told_apart(self) -> None:
        message = _build_confirm_message(0, "retracted from memory", [], 2, 1)
        assert message == (
            "0 memory edge(s) retracted from memory; "
            "2 fact(s) derived from them retracted too. "
            "1 memory edge(s) had already been erased; what was derived from "
            "them was retracted and erased."
        )

    def test_caps_inlined_detail_and_notes_remainder(self) -> None:
        overflow = _MAX_FAILURE_DETAIL + 4
        failures = [
            MemoryForgetFailure(
                uuid=f"uuid-{i}",
                code=MemoryForgetFailureCode.NO_MATCH,
                reason="x" * 120,
            )
            for i in range(overflow)
        ]

        message = _build_confirm_message(0, "retracted from memory", failures)

        # Full count is reported, but only the first N reasons are inlined.
        assert f"{overflow} failed" in message
        assert f"…and {overflow - _MAX_FAILURE_DETAIL} more" in message
        assert message.count("uuid-") == _MAX_FAILURE_DETAIL
        # Bounded regardless of batch size — cannot grow with the input.
        assert len(message) < _MAX_FAILURE_DETAIL * 300
