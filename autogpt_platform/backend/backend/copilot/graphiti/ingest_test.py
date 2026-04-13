"""Tests for Graphiti ingestion queue and worker logic."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import ingest


def _clean_module_state() -> None:
    """Reset module-level state to avoid cross-test contamination."""
    ingest._user_queues.clear()
    ingest._user_workers.clear()


@pytest.fixture(autouse=True)
def _reset_state():
    _clean_module_state()
    yield
    # Cancel any lingering worker tasks.
    for task in ingest._user_workers.values():
        task.cancel()
    _clean_module_state()


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
        assert len(ingest._user_queues) == 0


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
            ingest._user_queues[user_id] = tiny_q

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


class TestWorkerIdleTimeout:
    @pytest.mark.asyncio
    async def test_worker_cleans_up_on_idle(self) -> None:
        user_id = "idle-user"
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        # Pre-populate state so cleanup can remove entries.
        ingest._user_queues[user_id] = queue
        task_sentinel = MagicMock()
        ingest._user_workers[user_id] = task_sentinel

        original_timeout = ingest._WORKER_IDLE_TIMEOUT
        ingest._WORKER_IDLE_TIMEOUT = 0.05
        try:
            await ingest._ingestion_worker(user_id, queue)
        finally:
            ingest._WORKER_IDLE_TIMEOUT = original_timeout

        # After idle timeout the worker should have cleaned up.
        assert user_id not in ingest._user_queues
        assert user_id not in ingest._user_workers
