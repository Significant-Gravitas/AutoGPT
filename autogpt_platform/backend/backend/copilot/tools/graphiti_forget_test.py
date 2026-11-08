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
from backend.copilot.graphiti.recall import FORGOTTEN_FACT
from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_forget import (
    _MAX_FAILURE_DETAIL,
    SOFT_FORGET_NOTE,
    MemoryForgetConfirmTool,
    MemoryForgetSearchTool,
    _build_confirm_message,
    _timestamp,
)
from backend.copilot.tools.models import (
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
)

_MODULE = "backend.copilot.tools.graphiti_forget"


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
        with (patch(f"{_MODULE}.search_facts", search_facts),):
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
        with (patch(f"{_MODULE}.retract", retract),):
            await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["private-edge"]
            )

        retract.assert_awaited_once_with(
            MemoryScope.for_expert("user-abc", "expert-1"),
            ["private-edge"],
            hard=False,
        )


class TestForgetSearchCandidates:
    @pytest.mark.parametrize("status", [None, "active", "tentative"])
    @pytest.mark.asyncio
    async def test_candidates_carry_uuid_fact_and_validity(
        self, status: str | None
    ) -> None:
        edge = EntityEdge(
            uuid="e1",
            group_id="user_user-abc",
            source_node_uuid="a",
            target_node_uuid="b",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            name="works_on",
            fact="Alice works on Atlas",
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            attributes={"status": status} if status else {},
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (patch(f"{_MODULE}.search_facts", AsyncMock(return_value=[edge])),):
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
                **({"status": status} if status else {}),
            }
        ]


class TestForgetSearchForgottenCandidates:
    @staticmethod
    def _edge(
        uuid: str,
        fact: str,
        expired_at: datetime | None = None,
        valid_at: datetime | None = None,
        **attributes: str,
    ) -> EntityEdge:
        return EntityEdge(
            uuid=uuid,
            group_id="user_user-abc",
            source_node_uuid="a",
            target_node_uuid="b",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            name="works_on",
            fact=fact,
            expired_at=expired_at,
            valid_at=valid_at,
            attributes=attributes,
        )

    async def _search(self, include_forgotten: bool, live, forgotten):
        session = ChatSession.new("user-abc", dry_run=False)
        search_forgotten = AsyncMock(return_value=forgotten)
        with (
            patch(f"{_MODULE}.search_facts", AsyncMock(return_value=live)),
            patch(f"{_MODULE}.search_forgotten_facts", search_forgotten),
            patch(f"{_MODULE}.recheck", AsyncMock(return_value=(live, []))),
        ):
            response = await MemoryForgetSearchTool()._execute(
                "user-abc",
                session,
                query="atlas",
                include_forgotten=include_forgotten,
            )
        assert isinstance(response, MemoryForgetCandidatesResponse)
        return response, search_forgotten

    @pytest.mark.asyncio
    async def test_are_left_out_unless_asked_for(self) -> None:
        live = [self._edge("e1", "Bob leads Atlas")]

        response, search_forgotten = await self._search(False, live, [])

        search_forgotten.assert_not_awaited()
        assert [c["uuid"] for c in response.candidates] == ["e1"]
        assert "forgotten" not in response.candidates[0]

    @pytest.mark.asyncio
    async def test_when_asked_come_after_the_live_ones_saying_nothing_they_said(
        self,
    ) -> None:
        """A soft forget can be made permanent, but nothing a forgotten fact
        said goes back to the model: not the audit copy its forget kept
        (``fact_redacted``), not the text a fact forgotten before the policy
        still has, not its scope nor the dates it held true. Only that it
        exists, when it was stored and when forgotten, searched for in the
        whole response the model reads."""
        live = [self._edge("e1", "Bob leads Atlas")]
        gone = [
            self._edge(
                "e2",
                FORGOTTEN_FACT,
                valid_at=datetime(2024, 3, 9, tzinfo=timezone.utc),
                fact_redacted="Alice works on the secret Atlas plan",
                name_redacted="works_on_secret_plan",
                scope="project:Orion",
                forgotten_at="2026-01-01T00:00:00+00:00",
            ),
            # Forgotten before the policy: its own text was never scrubbed.
            self._edge(
                "e3",
                "Carol owns the Atlas budget",
                expired_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            ),
        ]

        response, search_forgotten = await self._search(True, live, gone)

        search_forgotten.assert_awaited_once_with(
            MemoryScope.for_user("user-abc"), "atlas", limit=10
        )
        assert response.candidates[1:] == [
            {
                "uuid": "e2",
                "fact": FORGOTTEN_FACT,
                "stored_at": "2025-01-01T00:00:00+00:00",
                "forgotten_at": "2026-01-01T00:00:00+00:00",
                "forgotten": "true",
            },
            {
                "uuid": "e3",
                "fact": FORGOTTEN_FACT,
                "stored_at": "2025-01-01T00:00:00+00:00",
                "forgotten_at": "2025-06-01T00:00:00+00:00",
                "forgotten": "true",
            },
        ]
        assert response.candidates[0]["fact"] == "Bob leads Atlas"
        seen = response.model_dump_json()
        for said in ("secret", "Carol", "budget", "Orion", "2024-03-09"):
            assert said not in seen, said
        assert "hard_delete=true" in response.message

    @pytest.mark.asyncio
    async def test_their_dates_are_only_ever_timestamps(self) -> None:
        """Codex's probe: a ``forgotten_at`` that is not a timestamp (one an
        extraction invented before attribute answers were filtered) reads
        "unknown", and what it held reaches no part of the response. A real
        one is written out anew, in UTC to the second."""
        secret = "Alice works on the secret Atlas plan"
        gone = [
            self._edge("e2", FORGOTTEN_FACT, forgotten_at=secret),
            self._edge(
                "e3", FORGOTTEN_FACT, forgotten_at="2026-01-01T07:30:15.250+02:00"
            ),
            self._edge("e4", FORGOTTEN_FACT, forgotten_at="0001-01-01T00:00:00+05:00"),
        ]

        response, _ = await self._search(True, [], gone)

        assert [c["forgotten_at"] for c in response.candidates] == [
            "unknown",
            "2026-01-01T05:30:15+00:00",
            "unknown",
        ]
        assert secret not in response.model_dump_json()

    def test_a_date_that_is_not_one_reads_unknown(self) -> None:
        naive = datetime(2025, 3, 4, 5, 6, 7)

        assert [
            _timestamp(value)
            for value in (None, "", "yesterday", 12, naive, "2025-03-04")
        ] == [
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "2025-03-04T05:06:07+00:00",
            "2025-03-04T00:00:00+00:00",
        ]

    @pytest.mark.asyncio
    async def test_are_offered_when_nothing_live_matches(self) -> None:
        gone = [self._edge("e2", FORGOTTEN_FACT, fact_redacted="Alice works on Atlas")]

        response, _ = await self._search(True, [], gone)

        assert [c["uuid"] for c in response.candidates] == ["e2"]


class TestForgetConfirmModes:
    @pytest.mark.asyncio
    async def test_hard_delete_asks_for_a_hard_retract(self) -> None:
        retract = AsyncMock(return_value=ForgetResult(deleted=["e1"]))
        session = ChatSession.new("user-abc", dry_run=False)
        with (patch(f"{_MODULE}.retract", retract),):
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
        with (patch(f"{_MODULE}.retract", AsyncMock(return_value=result)),):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.derived_uuids == ["d1", "d2", "d3"]
        assert response.message == (
            "1 memory edge(s) retracted from memory; "
            "3 fact(s) derived from them retracted too. " + SOFT_FORGET_NOTE
        )

    @pytest.mark.asyncio
    async def test_unavailable_graph_is_an_error_response(self) -> None:
        session = ChatSession.new("user-abc", dry_run=False)
        with (patch(f"{_MODULE}.retract", AsyncMock(side_effect=OSError("refused"))),):
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
            [{"count": 0}],  # no dream write still in flight cites it
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
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
        message = _build_confirm_message(3, True, [])
        assert message == "3 memory edge(s) permanently deleted."

    def test_the_facts_derived_from_them_are_counted(self) -> None:
        message = _build_confirm_message(1, True, [], 3)
        assert message == (
            "1 memory edge(s) permanently deleted; "
            "3 fact(s) derived from them retracted too."
        )

    def test_edges_already_erased_are_told_apart(self) -> None:
        message = _build_confirm_message(0, False, [], 2, 1)
        assert message.startswith("1 memory edge(s) had already been erased.")
        assert "0 memory edge(s) retracted from memory" not in message

    def test_a_hard_forget_whose_deletion_failed_says_so(self) -> None:
        """Its cascade ran, then deleting the edge failed: the edge and its
        text are still there, so the message must not call it erased."""
        failures = [
            MemoryForgetFailure(
                uuid="uuid-1",
                code=MemoryForgetFailureCode.CLEANUP_ERROR,
                reason="graph unavailable",
            )
        ]
        message = _build_confirm_message(0, True, failures, 2)
        assert message.startswith(
            "The memory edge(s) could not be permanently deleted, but 2 fact(s) "
            "derived from them were retracted; erasing their text may not have "
            "finished (see below)."
        )
        assert "already" not in message
        assert "uuid-1" in message

    def test_resumed_edges_claim_derived_facts_only_when_there_were_any(
        self,
    ) -> None:
        none_derived = _build_confirm_message(1, True, [], 0, 1)
        assert none_derived == (
            "1 memory edge(s) permanently deleted. "
            "1 memory edge(s) had already been erased."
        )
        assert "derived" not in none_derived

        derived = _build_confirm_message(1, True, [], 2, 1)
        assert derived == (
            "1 memory edge(s) permanently deleted. "
            "1 memory edge(s) had already been erased. "
            "2 fact(s) derived from these edges were retracted and erased."
        )

    def test_one_derived_count_is_credited_to_every_kind_of_edge(self) -> None:
        """The count covers what the deleted and the already-erased edges
        derived, so it is never credited to the deleted ones alone. A soft
        forget erases only the already-erased edges' derived facts."""
        hard = _build_confirm_message(2, True, [], 5, 1)
        assert "derived from them" not in hard
        assert hard.endswith(
            "5 fact(s) derived from these edges were retracted and erased."
        )

        soft = _build_confirm_message(2, False, [], 5, 1)
        assert soft.startswith(
            "2 memory edge(s) retracted from memory. "
            "1 memory edge(s) had already been erased. "
            "5 fact(s) derived from these edges were retracted, and those "
            "derived from the already-erased edges were erased too."
        )
        assert "retracted and erased" not in soft

        failures = [
            MemoryForgetFailure(
                uuid="uuid-3",
                code=MemoryForgetFailureCode.CLEANUP_ERROR,
                reason="erase failed",
            )
        ]
        failed = _build_confirm_message(2, False, failures, 5, 1)
        assert (
            "erasing those derived from the already-erased edges may not have "
            "finished (see below)."
        ) in failed
        assert "erased too" not in failed

    def test_erasure_is_claimed_only_when_nothing_failed(self) -> None:
        """A derived fact is counted once retracted, before its text is erased,
        so a failure means its text may still be there."""
        clean = _build_confirm_message(0, True, [], 2, 1)
        assert clean == (
            "1 memory edge(s) had already been erased. "
            "2 derived fact(s) retracted and erased."
        )

        failures = [
            MemoryForgetFailure(
                uuid="uuid-2",
                code=MemoryForgetFailureCode.CLEANUP_ERROR,
                reason="erase failed",
            )
        ]
        resumed = _build_confirm_message(0, True, failures, 2, 1)
        assert resumed.startswith(
            "1 memory edge(s) had already been erased. 2 derived fact(s) "
            "retracted; erasing their text may not have finished (see below)."
        )
        assert "retracted and erased" not in resumed

        mixed = _build_confirm_message(1, True, failures, 2, 1)
        assert "derived from these edges were retracted; erasing" in mixed
        assert "retracted and erased" not in mixed

    def test_a_soft_derived_only_result_never_claims_erasure(self) -> None:
        soft = _build_confirm_message(0, False, [], 2)
        assert soft == (
            "The memory edge(s) could not be retracted from memory, but 2 fact(s) "
            "derived from them were retracted."
        )
        assert "erased" not in soft

    def test_a_soft_forget_says_it_is_not_a_permanent_deletion(self) -> None:
        """A model told a user who had asked for a permanent delete that a soft
        forget was permanent. A soft forget's message says it was not, what it
        did instead and how to delete permanently; a hard forget's says none
        of that, and neither does a soft forget that retracted nothing."""
        soft = _build_confirm_message(1, False, [], 2)
        assert soft == (
            "1 memory edge(s) retracted from memory; 2 fact(s) derived from "
            "them retracted too. This is not a permanent deletion: they are "
            "hidden from recall, from the memory context chats start with and "
            "from dreams, and their text is kept for audit. To delete them "
            "permanently, call tool:memory_forget_confirm with the same UUIDs "
            "and hard_delete=true."
        )

        hard = _build_confirm_message(1, True, [], 2)
        assert hard == (
            "1 memory edge(s) permanently deleted; "
            "2 fact(s) derived from them retracted too."
        )

        nothing = _build_confirm_message(0, False, [])
        assert nothing == "0 memory edge(s) retracted from memory."

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

        message = _build_confirm_message(0, False, failures)

        # Full count is reported, but only the first N reasons are inlined.
        assert f"{overflow} failed" in message
        assert f"…and {overflow - _MAX_FAILURE_DETAIL} more" in message
        assert message.count("uuid-") == _MAX_FAILURE_DETAIL
        # Bounded regardless of batch size — cannot grow with the input.
        assert len(message) < _MAX_FAILURE_DETAIL * 300
