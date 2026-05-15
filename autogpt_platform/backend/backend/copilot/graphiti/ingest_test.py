"""Tests for Graphiti ingestion queue and worker logic."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import ingest

# Per-loop state in ingest.py auto-isolates between tests: pytest-asyncio
# creates a fresh event loop per test function, and the WeakKeyDictionary
# forgets the previous loop's state when it is GC'd. No manual reset needed.


class TestIngestionWorkerExceptionHandling:
    @pytest.mark.asyncio
    async def test_worker_continues_after_client_error(self) -> None:
        """If get_graphiti_client raises, the worker logs and continues."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        queue.put_nowait(
            {
                "name": "ep1",
                "episode_body": "hello",
                "source": "message",
                "source_description": "test",
                "reference_time": None,
                "group_id": "user_test",
            }
        )

        with (
            patch.object(
                ingest,
                "derive_group_id",
                return_value="user_test",
            ),
            patch.object(
                ingest,
                "get_graphiti_client",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection failed"),
            ),
        ):
            # Use a short idle timeout so the worker exits quickly.
            original_timeout = ingest._WORKER_IDLE_TIMEOUT
            ingest._WORKER_IDLE_TIMEOUT = 0.1
            try:
                await ingest._ingestion_worker("test-user", queue)
            finally:
                ingest._WORKER_IDLE_TIMEOUT = original_timeout

        # Worker processed the item (task_done called) and exited.
        assert queue.empty()


class TestEnqueueConversationTurn:
    @pytest.mark.asyncio
    async def test_empty_user_id_returns_without_error(self) -> None:
        await ingest.enqueue_conversation_turn(
            user_id="",
            session_id="sess1",
            user_msg="hi",
        )
        # No queue should have been created.
        assert len(ingest._get_loop_state().user_queues) == 0


class TestQueueFullScenario:
    @pytest.mark.asyncio
    async def test_queue_full_logs_warning_no_crash(self) -> None:
        user_id = "abc-valid-id"

        mock_understanding = SimpleNamespace(user_name="Alice")
        mock_understanding_db = MagicMock()
        mock_understanding_db.return_value.get_business_understanding = AsyncMock(
            return_value=mock_understanding
        )

        with (
            patch.object(
                ingest,
                "derive_group_id",
                return_value="user_abc-valid-id",
            ),
            patch(
                "backend.copilot.graphiti.ingest._resolve_user_name",
                new_callable=AsyncMock,
                return_value="Alice",
            ),
        ):
            # Create a tiny queue so it fills instantly.
            await ingest._ensure_worker(user_id)
            # Replace the queue with one that is already full.
            tiny_q: asyncio.Queue = asyncio.Queue(maxsize=1)
            tiny_q.put_nowait({"dummy": True})
            ingest._get_loop_state().user_queues[user_id] = tiny_q

            # Should not raise even though the queue is full.
            await ingest.enqueue_conversation_turn(
                user_id=user_id,
                session_id="sess1",
                user_msg="hi",
            )


class TestResolveUserName:
    @pytest.mark.asyncio
    async def test_fallback_when_db_raises(self) -> None:
        mock_db = MagicMock()
        mock_db.return_value.get_business_understanding = AsyncMock(
            side_effect=RuntimeError("DB not available")
        )

        with patch(
            "backend.data.db_accessors.understanding_db",
            mock_db,
        ):
            name = await ingest._resolve_user_name("some-user-id")

        assert name == "User"

    @pytest.mark.asyncio
    async def test_returns_user_name_when_available(self) -> None:
        mock_understanding = SimpleNamespace(user_name="Alice")
        mock_db = MagicMock()
        mock_db.return_value.get_business_understanding = AsyncMock(
            return_value=mock_understanding
        )

        with patch(
            "backend.data.db_accessors.understanding_db",
            mock_db,
        ):
            name = await ingest._resolve_user_name("some-user-id")

        assert name == "Alice"

    @pytest.mark.asyncio
    async def test_returns_user_when_understanding_is_none(self) -> None:
        mock_db = MagicMock()
        mock_db.return_value.get_business_understanding = AsyncMock(return_value=None)

        with patch(
            "backend.data.db_accessors.understanding_db",
            mock_db,
        ):
            name = await ingest._resolve_user_name("some-user-id")

        assert name == "User"


class TestEnqueueEpisode:
    @pytest.mark.asyncio
    async def test_enqueue_episode_returns_true_on_success(self) -> None:
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(
                ingest, "_ensure_worker", new_callable=AsyncMock
            ) as mock_worker,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            mock_worker.return_value = q

            result = await ingest.enqueue_episode(
                user_id="abc",
                session_id="sess1",
                name="test_ep",
                episode_body="hello",
                is_json=False,
            )
            assert result is True
            assert not q.empty()

    @pytest.mark.asyncio
    async def test_enqueue_episode_returns_false_for_empty_user(self) -> None:
        result = await ingest.enqueue_episode(
            user_id="",
            session_id="sess1",
            name="test_ep",
            episode_body="hello",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_enqueue_episode_returns_false_on_invalid_user(self) -> None:
        with patch.object(ingest, "derive_group_id", side_effect=ValueError("bad id")):
            result = await ingest.enqueue_episode(
                user_id="bad",
                session_id="sess1",
                name="test_ep",
                episode_body="hello",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_enqueue_episode_json_mode(self) -> None:
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(
                ingest, "_ensure_worker", new_callable=AsyncMock
            ) as mock_worker,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            mock_worker.return_value = q

            result = await ingest.enqueue_episode(
                user_id="abc",
                session_id="sess1",
                name="test_ep",
                episode_body='{"content": "hello"}',
                is_json=True,
            )
            assert result is True
            item = q.get_nowait()
            from graphiti_core.nodes import EpisodeType

            assert item["source"] == EpisodeType.json


class TestDerivedFindingLane:
    @pytest.mark.asyncio
    async def test_finding_worthy_message_enqueues_two_episodes(self) -> None:
        """A substantive assistant message should enqueue both the user
        episode and a derived-finding episode."""
        long_msg = "The analysis reveals significant growth patterns " + "x" * 200

        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(
                ingest, "_ensure_worker", new_callable=AsyncMock
            ) as mock_worker,
            patch(
                "backend.copilot.graphiti.ingest._resolve_user_name",
                new_callable=AsyncMock,
                return_value="Alice",
            ),
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            mock_worker.return_value = q

            await ingest.enqueue_conversation_turn(
                user_id="abc",
                session_id="sess1",
                user_msg="tell me about growth",
                assistant_msg=long_msg,
            )
            # Should have 2 items: user episode + derived finding
            assert q.qsize() == 2

    @pytest.mark.asyncio
    async def test_short_assistant_msg_skips_finding(self) -> None:
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(
                ingest, "_ensure_worker", new_callable=AsyncMock
            ) as mock_worker,
            patch(
                "backend.copilot.graphiti.ingest._resolve_user_name",
                new_callable=AsyncMock,
                return_value="Alice",
            ),
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            mock_worker.return_value = q

            await ingest.enqueue_conversation_turn(
                user_id="abc",
                session_id="sess1",
                user_msg="hi",
                assistant_msg="ok",
            )
            # Only 1 item: the user episode (no finding for short msg)
            assert q.qsize() == 1


class TestDerivedFindingDistillation:
    """_is_finding_worthy and _distill_finding gate derived-finding creation."""

    def test_short_message_not_finding_worthy(self) -> None:
        assert ingest._is_finding_worthy("ok") is False

    def test_chatter_prefix_not_finding_worthy(self) -> None:
        assert ingest._is_finding_worthy("done " + "x" * 200) is False

    def test_long_substantive_message_is_finding_worthy(self) -> None:
        msg = "The quarterly revenue analysis shows a 15% increase " + "x" * 200
        assert ingest._is_finding_worthy(msg) is True

    def test_distill_finding_truncates_to_500(self) -> None:
        result = ingest._distill_finding("x" * 600)
        assert result is not None
        assert len(result) == 503  # 500 + "..."


class TestWorkerIdleTimeout:
    @pytest.mark.asyncio
    async def test_worker_cleans_up_on_idle(self) -> None:
        user_id = "idle-user"
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        # Pre-populate state so cleanup can remove entries.
        state = ingest._get_loop_state()
        state.user_queues[user_id] = queue
        task_sentinel = MagicMock()
        state.user_workers[user_id] = task_sentinel

        original_timeout = ingest._WORKER_IDLE_TIMEOUT
        ingest._WORKER_IDLE_TIMEOUT = 0.05
        try:
            await ingest._ingestion_worker(user_id, queue)
        finally:
            ingest._WORKER_IDLE_TIMEOUT = original_timeout

        # After idle timeout the worker should have cleaned up.
        assert user_id not in state.user_queues
        assert user_id not in state.user_workers
