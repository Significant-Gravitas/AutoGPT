"""The ingestion worker hands graphiti's extraction only the earlier episodes
the recall policy allows, writes each episode holding the graph's write lock,
puts back to ``active`` any edge the extraction gave ``retracted``, and
promotes the tentative facts a user's chat turn states, mocked at the graphiti
boundary and on the in-memory Redis.

The live runs are ``ingest_recall_integration_test.py`` and
``recall_inflight_integration_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT

from . import ingest, scope_lock
from .recall import is_recallable_episode, recallable_episode_predicate
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_LOCK = write_lock_key("user_test")
# What the episode store holds, newest first as the query orders them.
_STORED = [
    {"uuid": "ep-newest", "redacted": False, "entity_edges": ["kept"]},
    {"uuid": "ep-redacted", "redacted": True, "entity_edges": []},
    {
        "uuid": "ep-pending",
        "redacted": False,
        "entity_edges": [],
        "write_pending": True,
    },
    {"uuid": "ep-cites-a-forgotten-fact", "redacted": False, "entity_edges": ["gone"]},
    {"uuid": "ep-oldest", "redacted": False, "entity_edges": []},
]


async def _policy_filtered_store(query: str, **params: object):
    """The graph's answer to a recallable-episodes read, filtered with the
    predicate's Python twin; any other read would be unfiltered."""
    assert recallable_episode_predicate("e") in query, "an unfiltered episode read"
    rows = [
        {"uuid": episode["uuid"]}
        for episode in _STORED
        if is_recallable_episode(
            episode["entity_edges"],
            {"gone"},
            redacted=episode["redacted"],
            write_pending=episode.get("write_pending", False),
        )
    ]
    return rows, [], None


def _payload(**extra: object) -> dict:
    return {
        "name": "conversation_s1",
        "episode_body": "User: Bob leads Atlas",
        "source": EpisodeType.message,
        "source_description": "User message in session s1",
        "reference_time": _NOW,
        "group_id": "user_test",
        **extra,
    }


async def _run_worker(
    client: MagicMock, monkeypatch: pytest.MonkeyPatch, *payloads: dict
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    for payload in payloads or (_payload(),):
        queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
    ):
        worker = ingest._ingestion_worker("test-user", "user_test", queue)
        await asyncio.wait_for(worker, 10)


def _graphiti_client() -> MagicMock:
    client = MagicMock()
    client.add_episode = AsyncMock()
    client.driver.execute_query = AsyncMock(side_effect=_policy_filtered_store)
    return client


def _undo_calls(client: MagicMock) -> list:
    return [
        call
        for call in client.driver.execute_query.await_args_list
        if call.args[0] == ingest._UNDO_EXTRACTED_RETRACTIONS_QUERY
    ]


class TestExtractionContext:
    @pytest.mark.asyncio
    async def test_add_episode_gets_only_recallable_earlier_episodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _graphiti_client()

        await _run_worker(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert client.add_episode.await_args is not None
        previous = client.add_episode.await_args.kwargs["previous_episode_uuids"]
        assert previous == ["ep-oldest", "ep-newest"], "oldest first, as graphiti"
        assert "ep-redacted" not in previous
        [read] = client.driver.execute_query.await_args_list
        assert read.kwargs == {
            "group_id": "user_test",
            "reference_time": _NOW,
            "source": "message",
            "limit": RELEVANT_SCHEMA_LIMIT,
        }

    @pytest.mark.asyncio
    async def test_a_failed_read_still_writes_without_earlier_episodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` would let graphiti pick for itself, forgotten episodes
        included; the episode is still written."""
        client = _graphiti_client()
        client.driver.execute_query = AsyncMock(side_effect=RuntimeError("down"))

        await _run_worker(client, monkeypatch)

        assert client.add_episode.await_args is not None
        assert client.add_episode.await_args.kwargs["previous_episode_uuids"] == []


class TestAnExtractedRetraction:
    @pytest.mark.asyncio
    async def test_is_put_back_to_active_while_the_lock_is_held(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """graphiti does not hold its extraction model to ``MemoryFact``'s
        statuses, so the worker puts any edge the episode wrote back from
        ``retracted`` to ``active``, under the lock the write held; the query
        never touches an edge a forget marked."""
        client = _graphiti_client()
        written = [SimpleNamespace(uuid="e1"), SimpleNamespace(uuid="e2")]
        client.add_episode.return_value = SimpleNamespace(edges=written)
        held: list[bool] = []

        async def store(query: str, **params: object):
            if query == ingest._UNDO_EXTRACTED_RETRACTIONS_QUERY:
                held.append(_LOCK in lock_redis.values)
                return [], [], None
            return await _policy_filtered_store(query, **params)

        client.driver.execute_query = AsyncMock(side_effect=store)

        await _run_worker(client, monkeypatch)

        [undo] = _undo_calls(client)
        assert undo.kwargs == {
            "uuids": ["e1", "e2"],
            "retracted": "retracted",
            "active": "active",
        }
        assert held == [True]
        assert "e.forgotten_at IS NULL" in ingest._UNDO_EXTRACTED_RETRACTIONS_QUERY

    @pytest.mark.asyncio
    async def test_that_cannot_be_checked_leaves_the_episode_written(
        self,
        lock_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = _graphiti_client()
        client.add_episode.return_value = SimpleNamespace(
            edges=[SimpleNamespace(uuid="e1")]
        )

        async def store(query: str, **params: object):
            if query == ingest._UNDO_EXTRACTED_RETRACTIONS_QUERY:
                raise ConnectionError("falkordb down")
            return await _policy_filtered_store(query, **params)

        client.driver.execute_query = AsyncMock(side_effect=store)

        await _run_worker(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert len(_undo_calls(client)) == 1
        assert "Could not check the statuses" in caplog.text
        assert "Graphiti ingestion failed" not in caplog.text


class TestWriteLock:
    @pytest.mark.asyncio
    async def test_the_episode_is_written_holding_the_graphs_lock(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _graphiti_client()
        held: list[bool] = []
        client.add_episode.side_effect = lambda **_: held.append(
            _LOCK in lock_redis.values
        )

        await _run_worker(client, monkeypatch)

        assert held == [True]
        assert _LOCK not in lock_redis.values, "released afterwards"

    @pytest.mark.asyncio
    async def test_a_graph_a_forget_keeps_locked_is_tried_again(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first attempt writes nothing; the forget finishes while the
        episode pauses, and the episode, back at the end of the queue, is
        written, and only then does the enqueuer's completion count it."""
        client = _graphiti_client()
        lock_redis.values[_LOCK] = "a forget's token"
        completion = ingest.IngestionCompletion()
        completion.register()

        async def forget_finishes(waits: int) -> None:
            assert completion.pending == 1 and not client.add_episode.called
            del lock_redis.values[_LOCK]

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        monkeypatch.setattr(ingest, "_pause_before_retry", forget_finishes)
        await _run_worker(client, monkeypatch, _payload(_completion=completion))

        client.add_episode.assert_awaited_once()
        assert "_lock_waits" not in client.add_episode.await_args.kwargs
        assert await completion.wait(0)
        assert (completion.written, completion.pending) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_graph_locked_for_many_waits_keeps_the_episode(
        self,
        lock_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Codex's probe: an episode the graph's lock kept out wait after wait
        is never dropped. It goes back on the queue each time, after a pause
        that doubles up to its cap; every third wait a warning names it; it
        is written once the lock frees."""
        client = _graphiti_client()
        lock_redis.values[_LOCK] = "a stuck forget's token"
        completion = ingest.IngestionCompletion()
        completion.register()
        pauses: list[int] = []

        async def pause(waits: int) -> None:
            pauses.append(waits)
            if waits == 7:
                del lock_redis.values[_LOCK]

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        monkeypatch.setattr(ingest, "_pause_before_retry", pause)
        payload = _payload(name="remember-me", _completion=completion)
        await _run_worker(client, monkeypatch, payload)

        assert pauses == [1, 2, 3, 4, 5, 6, 7]
        client.add_episode.assert_awaited_once()
        assert (completion.written, completion.pending) == (1, 0)
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert [m for m in warned if "'remember-me' is still queued" in m] == [
            f"Graph user_test locked by another writer for {waits} waits; the "
            "episode 'remember-me' is still queued, unwritten"
            for waits in (3, 6)
        ]
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    @pytest.mark.asyncio
    async def test_with_no_room_in_the_queue_the_episode_is_tried_again_in_place(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The queue filled up while the episode waited: it keeps its turn,
        is tried again where it stands, written first once the lock frees,
        and the episode behind it after."""
        client = _graphiti_client()
        lock_redis.values[_LOCK] = "a forget's token"
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        queue.put_nowait(_payload(name="first"))
        behind: list[str] = []

        async def pause(waits: int) -> None:
            if not behind:
                behind.append("second")
                queue.put_nowait(_payload(name="second"))
            else:
                lock_redis.values.pop(_LOCK, None)

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        monkeypatch.setattr(ingest, "_pause_before_retry", pause)
        monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
        with (
            patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
            patch.object(ingest, "ensure_indices_once", AsyncMock()),
        ):
            worker = ingest._ingestion_worker("test-user", "user_test", queue)
            await asyncio.wait_for(worker, 10)

        written = [call.kwargs["name"] for call in client.add_episode.await_args_list]
        assert written == ["first", "second"]

    def test_the_pause_doubles_up_to_its_cap(self) -> None:
        pauses = [ingest._retry_pause(waits) for waits in (1, 2, 3, 5, 6, 10_000)]

        assert pauses == [1.0, 2.0, 4.0, 16.0, 30.0, 30.0]

    @pytest.mark.asyncio
    async def test_a_write_that_raises_is_counted_failed(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _graphiti_client()
        client.add_episode.side_effect = RuntimeError("extraction failed")
        completion = ingest.IngestionCompletion()
        completion.register()

        await _run_worker(client, monkeypatch, _payload(_completion=completion))

        assert await completion.wait(0)
        assert (completion.written, completion.failed, completion.pending) == (0, 1, 0)

    @pytest.mark.asyncio
    async def test_without_redis_the_episode_is_written_anyway(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _graphiti_client()
        down = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch.object(scope_lock, "get_redis_async", down):
            await _run_worker(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert "write lock unavailable" in caplog.text


def _promotions(client: MagicMock) -> list[str]:
    """The edges the worker promoted (``ratification._promote_if_tentative``)."""
    return [
        str(call.kwargs["uuid"])
        for call in client.driver.execute_query.await_args_list
        if "e.ratified_at = $now" in call.args[0]
    ]


def _turn_result() -> SimpleNamespace:
    """A chat turn graphiti wrote: a fact it merged the turn into, one it
    created, and one it invalidated (the turn's episode not on it)."""
    return SimpleNamespace(
        episode=SimpleNamespace(uuid="ep-turn"),
        edges=[
            SimpleNamespace(uuid="merged", episodes=["ep-old", "ep-turn"]),
            SimpleNamespace(uuid="created", episodes=["ep-turn"]),
            SimpleNamespace(uuid="invalidated", episodes=["ep-old"]),
        ],
    )


class TestAUsersRestatement:
    @pytest.mark.asyncio
    async def test_promotes_the_facts_the_turn_states_holding_the_lock(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every fact whose episodes include the turn's, merged into or new;
        the one graphiti invalidated is left alone, and so are the recall
        stamps and hit counts (the promotion writes status and
        ``ratified_at`` only)."""
        client = _graphiti_client()
        client.add_episode.return_value = _turn_result()
        held: list[bool] = []

        async def store(query: str, **params: object):
            if "e.ratified_at = $now" in query:
                held.append(_LOCK in lock_redis.values)
                assert "recall" not in query
                return [{"uuid": params["uuid"]}], [], None
            return await _policy_filtered_store(query, **params)

        client.driver.execute_query = AsyncMock(side_effect=store)

        await _run_worker(client, monkeypatch, _payload(_restatement=True))

        assert _promotions(client) == ["merged", "created"]
        assert held == [True, True]
        assert client.add_episode.await_args is not None
        assert "_restatement" not in client.add_episode.await_args.kwargs

    @pytest.mark.asyncio
    async def test_an_episode_that_is_no_users_turn_promotes_nothing(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dream write, an assistant's finding or a memory-store call: no
        sidecar, so what they state stays as graphiti left it."""
        client = _graphiti_client()
        client.add_episode.return_value = _turn_result()

        await _run_worker(client, monkeypatch, _payload())

        assert _promotions(client) == []

    @pytest.mark.asyncio
    async def test_a_promotion_that_raises_leaves_the_episode_written(
        self,
        lock_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = _graphiti_client()
        client.add_episode.return_value = _turn_result()
        completion = ingest.IngestionCompletion()
        completion.register()

        async def store(query: str, **params: object):
            if "e.ratified_at = $now" in query:
                raise ConnectionError("falkordb down")
            return await _policy_filtered_store(query, **params)

        client.driver.execute_query = AsyncMock(side_effect=store)

        await _run_worker(
            client, monkeypatch, _payload(_restatement=True, _completion=completion)
        )

        assert _promotions(client) == ["merged", "created"], "each one tried"
        assert (completion.written, completion.failed) == (1, 0)
        assert "Graphiti ingestion failed" not in caplog.text
