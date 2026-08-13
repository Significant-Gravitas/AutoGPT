"""Tests for Graphiti ingestion queue and worker logic."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import ingest


@pytest.fixture(autouse=True)
def _stub_dream_registration(mocker):
    """_ensure_worker fires ensure_dream_system_scheduled fire-and-forget for
    every first-seen user. Unmocked it runs a REAL Prisma timezone lookup on
    this test's function-scoped event loop whenever an earlier test already
    connected Prisma, leaving a pool connection bound to a dead loop that
    later kills a session-loop test with "Event loop is closed"."""
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )


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

    @pytest.mark.asyncio
    async def test_worker_marks_task_done_even_when_ingestion_fails(self) -> None:
        """``queue.join()`` (the basis of ``wait_for_ingestion``) only
        completes if the worker calls ``task_done()`` for every item —
        including items whose graph write raised."""
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
            patch.object(ingest, "derive_group_id", return_value="user_test"),
            patch.object(
                ingest,
                "get_graphiti_client",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection failed"),
            ),
        ):
            original_timeout = ingest._WORKER_IDLE_TIMEOUT
            ingest._WORKER_IDLE_TIMEOUT = 0.05
            try:
                await ingest._ingestion_worker("test-user", queue)
            finally:
                ingest._WORKER_IDLE_TIMEOUT = original_timeout

        # join() resolves immediately only if task_done() was called for
        # the failed item; a hang here means the worker leaked the count.
        await asyncio.wait_for(queue.join(), timeout=0.1)


class TestIngestionCompletion:
    """``IngestionCompletion`` — the per-pass barrier that lets a caller wait
    on ONLY its own enqueued episodes, decoupled from the shared queue.

    These are pure in-memory and touch no process-global ``user_queues``,
    so they leak no shared state across tests."""

    @pytest.mark.asyncio
    async def test_no_registered_items_is_vacuously_drained(self) -> None:
        assert await ingest.IngestionCompletion().wait(0.01) is True

    @pytest.mark.asyncio
    async def test_wait_true_once_all_registered_complete(self) -> None:
        c = ingest.IngestionCompletion()
        c.register()
        c.register()
        assert c.registered == 2
        c.complete_one()
        c.complete_one()
        assert await c.wait(0.01) is True

    @pytest.mark.asyncio
    async def test_wait_false_while_items_still_outstanding(self) -> None:
        c = ingest.IngestionCompletion()
        c.register()
        c.register()
        c.complete_one()  # only 1 of 2 landed
        assert await c.wait(0.02) is False

    @pytest.mark.asyncio
    async def test_over_completion_never_hangs_the_wait(self) -> None:
        """Counters are monotonic rather than a decrementing balance, so more
        completions than registrations cannot drive the outstanding count
        negative and strand the waiter."""
        c = ingest.IngestionCompletion()
        c.register()
        c.complete_one()
        c.complete_one()  # spurious extra completion
        assert await c.wait(0.01) is True

    @pytest.mark.asyncio
    async def test_registration_after_a_completion_still_resolves(self) -> None:
        """The worker can complete an early episode while the caller is still
        enqueueing later ones; the barrier must resolve once the counts meet
        again, not stay latched on the interleave."""
        c = ingest.IngestionCompletion()
        c.register()
        c.complete_one()
        c.register()  # enqueued after the first landed
        assert await c.wait(0.02) is False
        c.complete_one()
        assert await c.wait(0.01) is True


class TestWaitForIngestion:
    """``wait_for_ingestion`` — the drain barrier dream-pass apply uses so
    'enqueued' can be upgraded to 'written' before counts are reported.
    Scoped to the caller's own episodes via ``IngestionCompletion``."""

    @pytest.mark.asyncio
    async def test_no_registered_episodes_returns_true(self) -> None:
        """Nothing enqueued — vacuously drained."""
        assert (
            await ingest.wait_for_ingestion(ingest.IngestionCompletion(), 0.1) is True
        )

    @pytest.mark.asyncio
    async def test_drains_own_episodes_while_unrelated_chat_still_in_flight(
        self,
    ) -> None:
        """The drain resolves as soon as the pass's OWN episodes land, even
        while an unrelated live-chat episode is still being processed on the
        same shared per-user queue — a whole-queue ``join()`` would still be
        blocked at that instant."""
        user_id = "scoped-user"
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        completion = ingest.IngestionCompletion()
        chat_blocked = asyncio.Event()

        async def fake_add_episode(**kwargs):
            # Block only on the unrelated chat episode; the dream episodes
            # return immediately so the completion tracker can resolve.
            if kwargs.get("name") == "chat_1":
                await chat_blocked.wait()
            return None

        mock_client = MagicMock()
        mock_client.add_episode = AsyncMock(side_effect=fake_add_episode)

        for name in ("dream_1", "dream_2"):
            queue.put_nowait(
                {
                    "name": name,
                    "episode_body": "x",
                    "source": "message",
                    "source_description": "d",
                    "reference_time": None,
                    "group_id": "user_scoped",
                    "_completion": completion,
                }
            )
            completion.register()
        # Unrelated chat episode enqueued behind the dream writes, NOT tracked.
        queue.put_nowait(
            {
                "name": "chat_1",
                "episode_body": "y",
                "source": "message",
                "source_description": "c",
                "reference_time": None,
                "group_id": "user_scoped",
            }
        )

        with (
            patch.object(ingest, "derive_group_id", return_value="user_scoped"),
            patch.object(
                ingest,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
        ):
            worker = asyncio.create_task(ingest._ingestion_worker(user_id, queue))
            try:
                result = await ingest.wait_for_ingestion(completion, 5)
                assert result is True
                # The scoped drain returned while chat_1 is still in flight:
                # a whole-queue barrier would time out here.
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(queue.join(), 0.05)
            finally:
                chat_blocked.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_returns_false_when_own_episodes_never_land(self) -> None:
        """A stalled/dead worker (crashed mid add_episode, never signalling
        completion) must not hang the caller — give up after the timeout."""
        completion = ingest.IngestionCompletion()
        completion.register()  # one episode enqueued, never completed
        assert await ingest.wait_for_ingestion(completion, 0.05) is False


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
    async def test_enqueue_episode_rejects_oversized_body_without_queueing(
        self,
    ) -> None:
        """A body over MAX_EPISODE_BODY_BYTES is rejected (False) before any
        worker or queue is touched — degraded dream writes must not reach
        FalkorDB or the extraction LLM."""
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
                name="runaway_consolidated_fact",
                episode_body="x" * (ingest.MAX_EPISODE_BODY_BYTES + 1),
                is_json=True,
            )
            assert result is False
            mock_worker.assert_not_awaited()
            assert q.empty()

    @pytest.mark.asyncio
    async def test_enqueue_episode_accepts_body_at_exact_size_cap(self) -> None:
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
                name="cap_sized_ep",
                episode_body="x" * ingest.MAX_EPISODE_BODY_BYTES,
            )
            assert result is True
            assert not q.empty()

    @pytest.mark.asyncio
    async def test_enqueue_episode_size_cap_counts_bytes_not_chars(self) -> None:
        """Multi-byte UTF-8 content is measured in encoded bytes, so a
        char-count under the cap can still be rejected."""
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(
                ingest, "_ensure_worker", new_callable=AsyncMock
            ) as mock_worker,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            mock_worker.return_value = q

            # "é" encodes to 2 bytes — half the cap in chars, just over in bytes.
            body = "é" * (ingest.MAX_EPISODE_BODY_BYTES // 2 + 1)
            result = await ingest.enqueue_episode(
                user_id="abc",
                session_id="sess1",
                name="multibyte_ep",
                episode_body=body,
            )
            assert result is False
            assert q.empty()

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


class TestStampEdgeMetadata:
    """#13389: dream-envelope metadata is stamped onto the edges a dream
    episode newly created, gated on the ``episodes == [episode_uuid]``
    dedup-safety invariant so user-authored edges are never clobbered."""

    # The real producer (``dream/apply._edge_metadata``) always emits all
    # five keys, so tests pass a complete payload unless exercising the
    # incomplete-payload guard explicitly.
    FULL_META = {
        "status": "active",
        "source_kind": "user_asserted",
        "scope": "real:global",
        "confidence": None,
        "provenance": None,
    }

    def _edge(self, uuid: str, episodes: list[str], expired_at=None, invalid_at=None):
        return SimpleNamespace(
            uuid=uuid,
            episodes=episodes,
            expired_at=expired_at,
            invalid_at=invalid_at,
        )

    def _result(self, episode_uuid: str, edges):
        return SimpleNamespace(episode=SimpleNamespace(uuid=episode_uuid), edges=edges)

    def _client(self):
        client = SimpleNamespace()
        client.driver = SimpleNamespace(execute_query=AsyncMock())
        return client

    @pytest.mark.asyncio
    async def test_stamps_only_sole_sourced_edges(self) -> None:
        """The core safety test: only edges whose ``episodes`` is exactly
        [this episode] get stamped. A dedup-merge (extra uuid) and an
        invalidated pre-existing edge are both skipped — they may be
        user-authored, and stamping would overwrite their provenance."""
        client = self._client()
        result = self._result(
            "ep-1",
            [
                self._edge("new", ["ep-1"]),  # freshly created → stamp
                self._edge("merged", ["ep-1", "old-ep"]),  # dedup merge → skip
                self._edge("invalidated", ["other-ep"]),  # predates → skip
            ],
        )
        await ingest._stamp_edge_metadata(
            client, "user_abc", result, self.FULL_META, "abc"
        )
        client.driver.execute_query.assert_awaited_once()
        kwargs = client.driver.execute_query.await_args.kwargs
        assert kwargs["uuids"] == ["new"]
        assert kwargs["gid"] == "user_abc"

    @pytest.mark.asyncio
    async def test_skips_retired_but_stamps_future_invalid_at(self) -> None:
        """A brand-new edge graphiti retired in the same add_episode
        (episodes==[uuid] but expired_at, or a PAST invalid_at) must NOT be
        stamped with a live status — that would contradict its temporal
        fields. A FUTURE invalid_at is still live (true now, ends later) and
        MUST be stamped, else its dream metadata never lands."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        future = datetime.now(timezone.utc) + timedelta(days=365)
        client = self._client()
        result = self._result(
            "ep-1",
            [
                self._edge("live", ["ep-1"]),  # new + temporally live → stamp
                self._edge("expired", ["ep-1"], expired_at=past),  # retired → skip
                self._edge("invalid_past", ["ep-1"], invalid_at=past),  # → skip
                self._edge(
                    "invalid_future", ["ep-1"], invalid_at=future
                ),  # still live → stamp
            ],
        )
        await ingest._stamp_edge_metadata(
            client, "user_abc", result, self.FULL_META, "abc"
        )
        kwargs = client.driver.execute_query.await_args.kwargs
        assert kwargs["uuids"] == ["live", "invalid_future"]

    @pytest.mark.asyncio
    async def test_no_new_edges_skips_query_entirely(self) -> None:
        """A dream fact that only merged into existing edges produces no
        sole-sourced target → no Cypher runs at all."""
        client = self._client()
        result = self._result("ep-1", [self._edge("merged", ["ep-1", "old"])])
        await ingest._stamp_edge_metadata(
            client, "user_abc", result, self.FULL_META, "abc"
        )
        client.driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_incomplete_metadata_skips_stamp(self) -> None:
        """A partial payload missing a required field (status/source_kind/
        scope) must skip the stamp entirely rather than NULL-clobber the
        edge's required props."""
        client = self._client()
        result = self._result("ep-1", [self._edge("new", ["ep-1"])])
        await ingest._stamp_edge_metadata(
            client, "user_abc", result, {"status": "active"}, "abc"
        )
        client.driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sets_all_five_envelope_fields(self) -> None:
        client = self._client()
        result = self._result("ep-1", [self._edge("new", ["ep-1"])])
        meta = {
            "status": "tentative",
            "source_kind": "assistant_derived",
            "scope": "real:global",
            "confidence": 0.8,
            "provenance": "dream:p1:recombine:2026",
        }
        await ingest._stamp_edge_metadata(client, "user_abc", result, meta, "abc")
        kwargs = client.driver.execute_query.await_args.kwargs
        for k, v in meta.items():
            assert kwargs[k] == v

    @pytest.mark.asyncio
    async def test_stamp_failure_is_swallowed(self) -> None:
        """A stamp failure must not propagate — the edge still exists with
        graphiti defaults; ingestion must not be failed by a metadata miss."""
        client = self._client()
        client.driver.execute_query = AsyncMock(side_effect=RuntimeError("boom"))
        result = self._result("ep-1", [self._edge("new", ["ep-1"])])
        # Should not raise.
        await ingest._stamp_edge_metadata(
            client, "user_abc", result, self.FULL_META, "abc"
        )


class TestEnqueueEpisodeEdgeMetadata:
    @pytest.mark.asyncio
    async def test_edge_metadata_rides_payload_sidecar(self) -> None:
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(ingest, "_ensure_worker", new_callable=AsyncMock) as w,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            w.return_value = q
            meta = {"status": "active", "provenance": "dream:p1"}
            await ingest.enqueue_episode(
                user_id="abc",
                session_id="s",
                name="dream_ep",
                episode_body="{}",
                is_json=True,
                edge_metadata=meta,
            )
            payload = q.get_nowait()
            assert payload["_edge_metadata"] == meta

    @pytest.mark.asyncio
    async def test_default_sidecar_is_none_for_non_dream_writes(self) -> None:
        """Conversation turns / memory-store calls pass no edge_metadata →
        sidecar is None → worker skips stamping → no behavior change."""
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(ingest, "_ensure_worker", new_callable=AsyncMock) as w,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            w.return_value = q
            await ingest.enqueue_episode(
                user_id="abc", session_id="s", name="ep", episode_body="hi"
            )
            payload = q.get_nowait()
            assert payload["_edge_metadata"] is None
            # Non-scoped-drain writes carry no completion tracker.
            assert payload["_completion"] is None

    @pytest.mark.asyncio
    async def test_completion_rides_payload_sidecar(self) -> None:
        """A scoped-drain caller's completion tracker is threaded onto the
        payload so the worker can signal it after processing."""
        with (
            patch.object(ingest, "derive_group_id", return_value="user_abc"),
            patch.object(ingest, "_ensure_worker", new_callable=AsyncMock) as w,
        ):
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            w.return_value = q
            completion = ingest.IngestionCompletion()
            await ingest.enqueue_episode(
                user_id="abc",
                session_id="s",
                name="dream_ep",
                episode_body="{}",
                is_json=True,
                completion=completion,
            )
            payload = q.get_nowait()
            assert payload["_completion"] is completion
