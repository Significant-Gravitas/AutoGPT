import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import falkordb_driver as fdb
from .falkordb_driver import AutoGPTFalkorDriver
from .migrations import backfill_legacy_forgets
from .scope import MemoryScope


class _FakeResult:
    """Minimal stand-in for a falkordb query result (header + rows)."""

    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set


def _set_query(driver: AutoGPTFalkorDriver, side_effect) -> AsyncMock:
    """Wire ``graph.query`` (reached via ``_get_graph``) to ``side_effect``."""
    query = AsyncMock(side_effect=side_effect)
    driver.client.select_graph.return_value.query = query
    return query


def _set_ro_query(driver: AutoGPTFalkorDriver, side_effect) -> AsyncMock:
    """Wire ``graph.ro_query`` (reached via ``_get_graph``) to ``side_effect``."""
    ro_query = AsyncMock(side_effect=side_effect)
    driver.client.select_graph.return_value.ro_query = ro_query
    return ro_query


def _overflow() -> Exception:
    return Exception("Max pending queries exceeded")


@pytest.fixture
def driver() -> AutoGPTFalkorDriver:
    # ``build_fulltext_query`` is a pure string-builder that never touches
    # the FalkorDB client; the query tests wire the mock's ``select_graph``
    # directly.
    return AutoGPTFalkorDriver(falkor_db=MagicMock())


def test_build_fulltext_query_uses_unquoted_group_ids_for_falkordb(
    driver: AutoGPTFalkorDriver,
) -> None:
    query = driver.build_fulltext_query(
        "Sarah",
        group_ids=["user_883cc9da-fe37-4863-839b-acba022bf3ef"],
    )

    assert query == "(@group_id:user_883cc9da-fe37-4863-839b-acba022bf3ef) (Sarah)"
    assert '"user_883cc9da-fe37-4863-839b-acba022bf3ef"' not in query


def test_build_fulltext_query_joins_multiple_group_ids_with_or(
    driver: AutoGPTFalkorDriver,
) -> None:
    query = driver.build_fulltext_query("Sarah", group_ids=["user_a", "user_b"])

    assert query == "(@group_id:user_a|user_b) (Sarah)"


def test_stopwords_only_query_returns_group_filter_only(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Line 25: sanitized_query is empty (all stopwords) but group_ids present."""
    # "the" is a common stopword — the query should reduce to just the group filter.
    query = driver.build_fulltext_query(
        "the",
        group_ids=["user_abc"],
    )

    assert query == "(@group_id:user_abc)"


def test_query_without_group_ids_returns_parenthesized_query(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Line 27: sanitized_query has content but no group_ids provided."""
    query = driver.build_fulltext_query("Sarah", group_ids=None)

    assert query == "(Sarah)"


# ---------------------------------------------------------------------------
# build_indices — pins the contract that suppresses graphiti-core's per-driver
# background indexing task. Default is OFF because CREATE INDEX materializes
# the graph; only explicit write paths (``ensure_indices``) may create one.
# Regression coverage for the 13.7k-empty-graph FalkorDB maxmemory wedge, and
# for the "Buffer is closed" log spam on admin viz loads.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_indices_false_skips_super_call() -> None:
    """``build_indices=False`` → our override returns early and never
    delegates to ``FalkorDriver.build_indices_and_constraints``."""
    with (
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
            return_value=None,
        ),
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
            new=AsyncMock(),
        ) as super_build,
    ):
        driver = AutoGPTFalkorDriver(build_indices=False)
        await driver.build_indices_and_constraints()
    super_build.assert_not_called()


@pytest.mark.asyncio
async def test_build_indices_true_delegates_to_super() -> None:
    """Default ``build_indices=True`` preserves upstream behaviour —
    the long-lived chat-write client still gets its indices built."""
    with (
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
            return_value=None,
        ),
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
            new=AsyncMock(),
        ) as super_build,
    ):
        driver = AutoGPTFalkorDriver(build_indices=True)
        await driver.build_indices_and_constraints()
    super_build.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_build_indices_does_not_create_graph() -> None:
    """Omitting the kwarg must NOT build indices.

    ``CREATE INDEX`` is a write, and a write materializes the graph in
    FalkorDB. An index-building default meant every driver construction —
    including read-only searches and the AGENTS.md debugging cookbook —
    minted a permanent empty graph for that user. Prod accumulated 13,732
    graphs, ~99.7% of them empty, which pinned FalkorDB at maxmemory.
    """
    with (
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
            return_value=None,
        ),
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
            new=AsyncMock(),
        ) as super_build,
    ):
        driver = AutoGPTFalkorDriver()
        await driver.build_indices_and_constraints()
    super_build.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_indices_builds_despite_default_optout() -> None:
    """``ensure_indices()`` is the write path's explicit opt-in — it must
    delegate to upstream even though the init-time task is suppressed."""
    with (
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
            return_value=None,
        ),
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
            new=AsyncMock(),
        ) as super_build,
    ):
        driver = AutoGPTFalkorDriver()
        await driver.build_indices_and_constraints()
        super_build.assert_not_called()
        await driver.ensure_indices()
    super_build.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_indices_false_persists_across_repeated_calls() -> None:
    """The override doesn't flip after the first call — every invocation
    against a ``build_indices=False`` driver stays a no-op."""
    with (
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
            return_value=None,
        ),
        patch(
            "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
            new=AsyncMock(),
        ) as super_build,
    ):
        driver = AutoGPTFalkorDriver(build_indices=False)
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()
    super_build.assert_not_called()


# ---------------------------------------------------------------------------
# execute_query retry — pins the bounded backoff on FalkorDB's transient
# "Max pending queries exceeded" backpressure. (SENTRY-1384.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_query_success_returns_upstream_shaped_records(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Happy path: no retry, and results are shaped like upstream
    (list-of-dicts, header, None)."""
    _set_ro_query(driver, [_FakeResult([("n.count", "count")], [[5]])])

    records, header, meta = await driver.execute_query("MATCH (n) RETURN count(n)")

    assert records == [{"count": 5}]
    assert header == ["count"]
    assert meta is None


@pytest.mark.asyncio
async def test_execute_query_retries_pending_queue_overflow_then_succeeds(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Two transient overflows are retried; the third attempt succeeds and its
    result is returned (memory op recovers instead of being dropped)."""
    query = _set_ro_query(
        driver,
        [_overflow(), _overflow(), _FakeResult([("x", "count")], [[1]])],
    )

    # max_attempts=5 (non-default) so the third attempt succeeds well within
    # budget — proves retries continue past the first failure.
    with (
        patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep,
        patch(
            "backend.copilot.graphiti.config.graphiti_config.falkordb_query_max_attempts",
            5,
        ),
    ):
        records, _, _ = await driver.execute_query("MATCH (n) RETURN 1")

    assert records == [{"count": 1}]
    assert query.await_count == 3
    assert sleep.await_count == 2  # backoff between the three attempts


@pytest.mark.asyncio
async def test_execute_query_raises_after_exhausting_retries(
    driver: AutoGPTFalkorDriver,
) -> None:
    """A sustained overflow exhausts the budget, then raises AND logs exactly
    one terminal error under the upstream logger (so Sentry sees one event)."""
    query = _set_ro_query(driver, _overflow())

    # max_attempts=4 (non-default) proves the knob controls the attempt count.
    with (
        patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep,
        patch.object(fdb, "_UPSTREAM_QUERY_LOGGER") as upstream_logger,
        patch(
            "backend.copilot.graphiti.config.graphiti_config.falkordb_query_max_attempts",
            4,
        ),
    ):
        with pytest.raises(Exception, match="Max pending queries exceeded"):
            await driver.execute_query("MATCH (n) RETURN 1")

    assert query.await_count == 4
    assert sleep.await_count == 3
    upstream_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_pending_queue_retry_delay_is_bounded_and_jittered() -> None:
    """Backoff grows exponentially within [base*2**n, base*2**n + base) and is
    capped so a high max_attempts can't balloon a single wait."""
    with patch(
        "backend.copilot.graphiti.config.graphiti_config.falkordb_query_backoff_base",
        0.1,
    ):
        assert 0.1 <= AutoGPTFalkorDriver._pending_queue_retry_delay(0) < 0.2
        assert 0.2 <= AutoGPTFalkorDriver._pending_queue_retry_delay(1) < 0.3
        # attempt 20 would be ~100k*base uncapped; the cap holds it near the max.
        capped = AutoGPTFalkorDriver._pending_queue_retry_delay(20)
        assert (
            fdb._MAX_RETRY_DELAY_SECONDS <= capped < fdb._MAX_RETRY_DELAY_SECONDS + 0.1
        )


@pytest.mark.asyncio
async def test_execute_query_non_overflow_error_fails_fast(
    driver: AutoGPTFalkorDriver,
) -> None:
    """A genuine query error (Cypher typo, missing graph, teardown) is not
    retried — it raises on the first attempt and logs once."""
    query = _set_ro_query(driver, ValueError("syntax error near RETRN"))

    with (
        patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep,
        patch.object(fdb, "_UPSTREAM_QUERY_LOGGER") as upstream_logger,
    ):
        with pytest.raises(ValueError):
            await driver.execute_query("MATCH (n) RETRN 1")

    assert query.await_count == 1
    sleep.assert_not_awaited()
    upstream_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_execute_query_already_indexed_returns_none_without_retry(
    driver: AutoGPTFalkorDriver,
) -> None:
    """The upstream 'already indexed' short-circuit is preserved: returns None,
    no retry, no terminal error."""
    query = _set_query(driver, Exception("Index already indexed"))

    with (
        patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep,
        patch.object(fdb, "_UPSTREAM_QUERY_LOGGER") as upstream_logger,
    ):
        result = await driver.execute_query("CREATE INDEX ...")

    assert result is None
    assert query.await_count == 1
    sleep.assert_not_awaited()
    upstream_logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# Read-only query routing.
#
# GRAPH.QUERY materializes the graph even for a pure MATCH, so routing reads
# through it silently creates an empty, permanently resident graph for every
# user merely looked at. That filled FalkorDB to maxmemory twice — most
# recently a single weekly community-rebuild sweep took prod from 91 to 19,033
# graphs, 100% of sampled ones empty. Reads must use GRAPH.RO_QUERY.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher,expected_read_only",
    [
        ("MATCH (n) RETURN count(n)", True),
        ("MATCH (n:Entity) WHERE n.group_id = $g RETURN n.name", True),
        ("CALL db.idx.fulltext.queryNodes('Entity', $q) YIELD node RETURN node", True),
        ("CREATE INDEX FOR (n:Entity) ON (n.uuid)", False),
        ("MERGE (n:Entity {uuid: $u})", False),
        ("MATCH (n) SET n.x = 1", False),
        ("MATCH (n) DETACH DELETE n", False),
        ("MATCH (n) REMOVE n.x", False),
    ],
)
def test_read_only_cypher_classification(cypher, expected_read_only) -> None:
    assert fdb._is_read_only_cypher(cypher) is expected_read_only


@pytest.mark.asyncio
async def test_read_only_query_uses_ro_query(driver: AutoGPTFalkorDriver) -> None:
    """A MATCH must go to ro_query and must never touch the creating path."""
    ro = _set_ro_query(driver, [_FakeResult([("x", "n")], [[1]])])
    write = _set_query(driver, [_FakeResult([("x", "n")], [[1]])])
    await driver.execute_query("MATCH (n) RETURN count(n)")
    ro.assert_awaited_once()
    write.assert_not_called()


@pytest.mark.asyncio
async def test_write_query_uses_query(driver: AutoGPTFalkorDriver) -> None:
    ro = _set_ro_query(driver, [_FakeResult([], [])])
    write = _set_query(driver, [_FakeResult([], [])])
    await driver.execute_query("MERGE (n:Entity {uuid: $u})", u="x")
    write.assert_awaited_once()
    ro.assert_not_called()


@pytest.mark.asyncio
async def test_missing_graph_returns_empty_without_creating(
    driver: AutoGPTFalkorDriver,
) -> None:
    """The regression guard.

    A read against a group with no graph yet must yield an empty result, NOT
    an error and NOT a fallback to ``graph.query`` — that fallback would
    materialize the graph, which is the entire bug.
    """
    ro = _set_ro_query(driver, Exception("ERR Invalid graph operation on empty key"))
    write = _set_query(driver, [_FakeResult([], [])])
    records, header, _ = await driver.execute_query("MATCH (n) RETURN count(n)")
    assert records == []
    assert header == []
    ro.assert_awaited_once()
    write.assert_not_called()


@pytest.mark.asyncio
async def test_ro_violation_falls_back_to_write(driver: AutoGPTFalkorDriver) -> None:
    """If the classifier is wrong, degrade to the write path rather than fail."""
    ro = _set_ro_query(
        driver, Exception("graph.RO_QUERY is to be executed only on read-only queries")
    )
    write = _set_query(driver, [_FakeResult([("x", "n")], [[1]])])
    await driver.execute_query("MATCH (n) RETURN count(n)")
    ro.assert_awaited_once()
    write.assert_awaited_once()


# --- regression guards for the review findings on the read-only routing ----


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n {kind: 'CREATE'}) RETURN n",
        'MATCH (n) WHERE n.name = "DELETE" RETURN n',
        "MATCH (n) RETURN n // SET is only mentioned in this comment",
        "/* MERGE */ MATCH (n) RETURN n",
    ],
)
def test_write_keywords_inside_literals_stay_read_only(cypher) -> None:
    """A write keyword inside a string literal or comment must not flip the
    query onto the graph-materializing path."""
    assert fdb._is_read_only_cypher(cypher) is True


@pytest.mark.asyncio
async def test_ro_violation_on_final_attempt_still_writes(
    driver: AutoGPTFalkorDriver,
) -> None:
    """The RO fallback must not consume a retry.

    With max_attempts=1 (or on the last attempt) a fallback that burned an
    attempt would exit the loop having never issued the write, hitting the
    "unreachable" AssertionError.
    """
    ro = _set_ro_query(
        driver, Exception("graph.RO_QUERY is to be executed only on read-only queries")
    )
    write = _set_query(driver, [_FakeResult([("x", "n")], [[1]])])
    with patch(
        "backend.copilot.graphiti.config.graphiti_config.falkordb_query_max_attempts",
        1,
    ):
        records, _, _ = await driver.execute_query("MATCH (n) RETURN count(n)")
    assert records == [{"n": 1}]
    ro.assert_awaited_once()
    write.assert_awaited_once()


# --- guards for the fulltext index-creation writes -------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        # graphiti-core builds all three fulltext indices this way. `\bCREATE\b`
        # does NOT match `createNodeIndex`, so these were classified read-only
        # and bounced off RO_QUERY on every new group's first write.
        (
            "CALL db.idx.fulltext.createNodeIndex({label: 'Episodic', stopwords: []},"
            " 'content', 'source', 'source_description', 'group_id')"
        ),
        "CALL db.idx.fulltext.createNodeIndex({label: 'Entity'}, 'name')",
        "CALL db.idx.fulltext.createNodeIndex({label: 'Community'}, 'name')",
        "CALL db.idx.fulltext.drop('Entity')",
    ],
)
def test_write_procedures_are_not_read_only(cypher) -> None:
    assert fdb._is_read_only_cypher(cypher) is False


def test_fulltext_query_procedure_stays_read_only() -> None:
    """The search path must keep using RO_QUERY."""
    assert (
        fdb._is_read_only_cypher(
            "CALL db.idx.fulltext.queryNodes('Entity', $query) YIELD node RETURN node"
        )
        is True
    )


def test_clone_returns_subclass_with_indices_disabled() -> None:
    """Upstream clone() builds a plain FalkorDriver, which would re-enable the
    init-time index task and route reads back through the creating path."""
    driver = AutoGPTFalkorDriver(falkor_db=MagicMock())
    driver._database = "user_a"
    cloned = driver.clone("user_b")
    assert isinstance(cloned, AutoGPTFalkorDriver)
    assert cloned._build_indices_at_init is False
    assert driver.clone("user_a") is driver


def test_open_driver_targets_the_scope_graph_without_building_indices() -> None:
    """The factory opens the scope's own graph, keeps the default that a
    bare construction never creates one, and builds no client to do it."""
    scope = MemoryScope.for_expert("user-1", "expert-1")
    with patch.object(fdb, "new_falkordb_client") as new_client:
        driver = fdb.open_driver(scope)

    assert driver.graph_name == scope.group_id
    assert driver._build_indices_at_init is False
    assert isinstance(driver.client, fdb.DeferredFalkorDB)
    new_client.assert_not_called()


def test_every_client_carries_the_transport_deadlines(monkeypatch) -> None:
    """falkordb defaults both to None, and its constructor probes the server
    with a synchronous INFO: without deadlines an unresponsive server holds
    an abandoned worker thread indefinitely."""
    monkeypatch.setattr(fdb.graphiti_config, "falkordb_socket_connect_timeout", 0.7)
    monkeypatch.setattr(fdb.graphiti_config, "falkordb_socket_timeout", 1.3)
    with patch.object(fdb, "FalkorDB") as falkordb:
        assert fdb.new_falkordb_client() is falkordb.return_value
        fdb.new_falkordb_client("falkordb.internal", 7000)

    configured, explicit = (call.kwargs for call in falkordb.call_args_list)
    for kwargs in (configured, explicit):
        assert kwargs["socket_connect_timeout"] == 0.7
        assert kwargs["socket_timeout"] == 1.3
    assert configured["host"] == fdb.graphiti_config.falkordb_host
    assert configured["port"] == fdb.graphiti_config.falkordb_port
    assert (explicit["host"], explicit["port"]) == ("falkordb.internal", 7000)


def _open_through(route: str) -> AutoGPTFalkorDriver:
    if route == "open_driver":
        return fdb.open_driver(MemoryScope.for_user("user-1"))
    if route == "host_and_port":
        return AutoGPTFalkorDriver(host="127.0.0.1", port=6380, database="user_a")
    return backfill_legacy_forgets._graph_driver("default_db")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["open_driver", "host_and_port", "legacy_forget_backfill"]
)
async def test_no_driver_builds_its_client_on_the_event_loop(route: str) -> None:
    """Building a client sends falkordb's synchronous cluster probe. However
    a driver is made (``open_driver``; host and port, as the test fixtures
    do; the legacy-forget backfill's ``_graph_driver``), making it builds
    nothing, and its first command builds the client on the connect pool
    while the loop keeps ticking."""
    built_on: list[str] = []
    real = MagicMock()
    real.execute_command = AsyncMock(side_effect=RuntimeError("no server here"))
    real.aclose = AsyncMock()

    def slow_build(**_kwargs) -> MagicMock:
        built_on.append(threading.current_thread().name)
        time.sleep(0.3)  # the probe, waiting on the server
        return real

    gaps: list[float] = []

    async def beat() -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    with patch.object(fdb, "FalkorDB", side_effect=slow_build):
        driver = _open_through(route)
        assert built_on == [], "making the driver built a client"
        ticker = asyncio.create_task(beat())
        try:
            with pytest.raises(RuntimeError, match="no server here"):
                await driver.execute_query("MATCH (n) RETURN count(n)")
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            await driver.close()

    assert len(built_on) == 1 and built_on[0].startswith("falkordb-connect")
    assert gaps and max(gaps) < 0.15, f"the loop stalled {max(gaps):.3f}s"
    real.aclose.assert_awaited_once()


class TestDeferredFalkorDB:
    @pytest.mark.asyncio
    async def test_concurrent_first_commands_share_one_build(self) -> None:
        builds: list[int] = []
        real = MagicMock()
        real.execute_command = AsyncMock(return_value="PONG")

        def build() -> MagicMock:
            builds.append(1)
            time.sleep(0.1)
            return real

        client = fdb.DeferredFalkorDB(build)
        results = await asyncio.gather(
            *(client.execute_command("PING") for _ in range(5))
        )

        assert results == ["PONG"] * 5
        assert len(builds) == 1

    @pytest.mark.asyncio
    async def test_a_failed_build_is_retried_by_the_next_command(self) -> None:
        attempts: list[int] = []
        real = MagicMock()
        real.list_graphs = AsyncMock(return_value=["user_a"])

        def build() -> MagicMock:
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("connection refused")
            return real

        client = fdb.DeferredFalkorDB(build)
        with pytest.raises(ConnectionError):
            await client.list_graphs()

        assert await client.list_graphs() == ["user_a"]
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_a_caller_that_gives_up_leaves_the_build_to_the_next(
        self,
    ) -> None:
        release = threading.Event()
        builds: list[int] = []
        real = MagicMock()
        real.execute_command = AsyncMock(return_value=1)

        def build() -> MagicMock:
            builds.append(1)
            release.wait(5)
            return real

        client = fdb.DeferredFalkorDB(build)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.execute_command("PING"), timeout=0.05)
        release.set()

        assert await client.execute_command("PING") == 1
        assert len(builds) == 1

    @pytest.mark.asyncio
    async def test_close_releases_the_built_client_and_builds_none(self) -> None:
        unused = fdb.DeferredFalkorDB(MagicMock(side_effect=AssertionError("built")))
        await unused.aclose()

        real = MagicMock()
        real.execute_command = AsyncMock()
        real.aclose = AsyncMock()
        used = fdb.DeferredFalkorDB(lambda: real)
        await used.execute_command("PING")
        await used.aclose()

        real.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_driver_builds_the_client_off_the_event_loop() -> None:
    """The blocking construction runs on the connect pool; only the driver,
    whose constructor schedules a task, is made on the loop."""
    built_on: list[int] = []

    def build() -> MagicMock:
        built_on.append(threading.get_ident())
        return MagicMock()

    with patch.object(fdb, "new_falkordb_client", side_effect=build):
        driver = await fdb.connect_driver("user_a", build_indices=False)

    assert built_on and built_on[0] != threading.get_ident()
    assert driver.graph_name == "user_a"
    assert driver._build_indices_at_init is False


@pytest.mark.asyncio
async def test_unrelated_readonly_wording_does_not_trigger_write_fallback(
    driver: AutoGPTFalkorDriver,
) -> None:
    """A spurious RO-violation match would send a genuine READ down the write
    path and materialize the graph. Redis's replica error ("read only", no
    hyphen) must not be mistaken for FalkorDB's RO_QUERY rejection."""
    ro = _set_ro_query(
        driver, Exception("READONLY You can't write against a read only replica.")
    )
    write = _set_query(driver, [_FakeResult([("x", "n")], [[1]])])
    with pytest.raises(Exception, match="read only replica"):
        await driver.execute_query("MATCH (n) RETURN count(n)")
    ro.assert_awaited()
    write.assert_not_called()
