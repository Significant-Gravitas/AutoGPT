"""Tests for the chat sessions a dream leaves out after a forget.

The live counterpart is
``graphiti/recall_integration_test.py::test_the_dream_leaves_out_the_sessions_a_forget_hid``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti import ingest
from backend.copilot.graphiti.memory_model import MemoryEnvelope
from backend.copilot.graphiti.recall import (
    forgotten_facts_clause,
    recallable_episode_predicate,
)

from . import hidden_sessions


def _envelope(provenance: str) -> str:
    return MemoryEnvelope(content="a fact", provenance=provenance).model_dump_json()


class TestEpisodeSessionIds:
    @pytest.mark.parametrize(
        ("name", "description", "content", "sessions"),
        [
            ("conversation_s1", "User message in session s1", "Alice: hi", {"s1"}),
            (
                "finding_s2",
                "Assistant-derived finding in session s2",
                _envelope("session:s2"),
                {"s2"},
            ),
            ("my note", "Conversation memory", _envelope("session:s3#msg:4"), {"s3"}),
            ("dream_p1_consolidate_0", "dream-pass", _envelope("dream:p1"), set()),
            ("episode-1", "a test episode", "not an envelope", set()),
            (None, None, None, set()),
        ],
    )
    def test_reads_the_session_from_name_description_or_provenance(
        self,
        name: str | None,
        description: str | None,
        content: str | None,
        sessions: set[str],
    ) -> None:
        assert hidden_sessions.episode_session_ids(name, description, content) == (
            sessions
        )

    @pytest.mark.asyncio
    async def test_reads_the_session_of_everything_ingestion_writes(self) -> None:
        """Drift guard: whatever a chat turn and its derived finding are
        named, the parser must still find their session in them."""
        payloads: list[dict] = []

        async def enqueue(_user_id: str, _group_id: str, payload: dict) -> bool:
            payloads.append(payload)
            return True

        with (
            patch.object(ingest, "_enqueue_payload", enqueue),
            patch.object(ingest, "resolve_user_name", AsyncMock(return_value="Al")),
        ):
            await ingest.enqueue_conversation_turn(
                "user-1", "sess-42", "hello", "The analysis shows " + "x" * 200
            )

        assert len(payloads) == 2, "a turn and its derived finding"
        for payload in payloads:
            found = hidden_sessions.episode_session_ids(
                payload["name"], payload["source_description"], payload["episode_body"]
            )
            assert found == {"sess-42"}


class TestHiddenSessionIds:
    @pytest.mark.asyncio
    async def test_asks_for_every_episode_the_policy_hides(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = (
            [
                {"name": "conversation_s1", "source_description": "", "content": ""},
                {"name": "a memory", "source_description": "", "content": "text"},
            ],
            [],
            None,
        )

        found = await hidden_sessions.hidden_session_ids(driver, "user_g")

        assert found == {"s1"}
        query = driver.execute_query.await_args.args[0]
        assert query.startswith(forgotten_facts_clause())
        assert f"NOT ({recallable_episode_predicate('n')})" in query
        assert driver.execute_query.await_args.kwargs == {"g": "user_g"}

    @pytest.mark.asyncio
    async def test_an_unreadable_graph_is_none_not_an_empty_set(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("falkordb down")

        assert await hidden_sessions.hidden_session_ids(driver, "user_g") is None
