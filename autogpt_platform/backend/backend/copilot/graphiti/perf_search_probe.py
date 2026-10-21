"""Do-not-merge probe (#15739 integration preview), v3: memory recall's
FalkorDB reads with #15842's lean projection in place.

- Each graphiti edge-search method's duration (``perf-search``), as in v1.
- Every labelled FalkorDB read split into server execution time, the rest of
  the command and falkordb-py's parse, and for a BFS read how many origins it
  was given and how many distinct (``perf-q``).
- Once the executor has been idle for a minute (after a browser run), the
  captured BFS reads are replayed as they were and with their origin list
  de-duplicated, checking both return the same rows in the same order with
  the same attributes, then timing each (``perf-bench``).
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_METHODS = ("edge_fulltext_search", "edge_similarity_search", "edge_bfs_search")
_ORIGINS = "bfs_origin_node_uuids"
_IDLE_SECONDS = 60.0
_ROUNDS = 5

_installed = False
_counted: set[str] = set()
_captured: dict[str, tuple[object, str, dict, int]] = {}
_last_read = 0.0
_bench: asyncio.Task | None = None
_original_query = None


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    from graphiti_core.search import search as graphiti_search

    for name in _METHODS:
        _wrap(graphiti_search, name)
    _wrap_falkordb()


def _wrap(module, name: str) -> None:
    original = getattr(module, name)

    async def timed(driver, *args, **kwargs):
        started = time.perf_counter()
        results = None
        try:
            results = await original(driver, *args, **kwargs)
            return results
        finally:
            graph = str(getattr(driver, "_database", "?"))[:20]
            logger.info(
                "perf-search %s %.0f ms results=%s graph=%s",
                name,
                (time.perf_counter() - started) * 1000,
                len(results) if results is not None else "error",
                graph,
            )
            if name == "edge_similarity_search" and graph not in _counted:
                _counted.add(graph)
                await _log_fact_count(driver, graph)

    setattr(module, name, timed)


async def _log_fact_count(driver, graph: str) -> None:
    try:
        records, _, _ = await driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS c"
        )
        logger.info("perf-search graph=%s facts=%s", graph, records[0]["c"])
    except Exception as e:  # probe only
        logger.info("perf-search graph=%s facts=? (%s)", graph, type(e).__name__)


def _wrap_falkordb() -> None:
    global _original_query
    from falkordb.asyncio.graph import AsyncGraph
    from falkordb.asyncio.query_result import QueryResult

    original_parse = QueryResult.parse

    async def timed_parse(self, response):
        started = time.perf_counter()
        try:
            return await original_parse(self, response)
        finally:
            self._perf_parse_ms = (time.perf_counter() - started) * 1000

    QueryResult.parse = timed_parse
    _original_query = AsyncGraph._query

    async def timed_query(self, q, params=None, timeout=None, read_only=False):
        global _last_read
        started = time.perf_counter()
        result = await _original_query(
            self, q, params=params, timeout=timeout, read_only=read_only
        )
        if not read_only:
            return result
        _last_read = time.monotonic()
        label = _label(q)
        total, parse, rows = _measure(result, started)
        origins = list((params or {}).get(_ORIGINS) or [])
        if label != "other" or total >= 30 or rows >= 10:
            logger.info(
                "perf-q %s total=%.0f exec=%.0f parse=%.0f server=%.1f rows=%d"
                " graph=%s%s",
                label,
                total,
                total - parse,
                parse,
                _server_ms(result),
                rows,
                self.name[:20],
                f" origins={len(origins)} unique={len(set(origins))}"
                if label == "bfs"
                else "",
            )
        if label == "bfs" and len(origins) > len(set(origins)):
            _capture(self, q, dict(params or {}), len(origins) - len(set(origins)))
        return result

    AsyncGraph._query = timed_query


def _label(q: str) -> str:
    if "vec.cosineDistance" in q:
        return "sim" if "fact_embedding" in q else "nsim"
    if "RELATES_TO|MENTIONS" in q:
        return "bfs"
    if "fulltext.queryRelationships" in q:
        return "ft"
    if "fulltext.queryNodes" in q:
        return "ftn"
    return "other"


def _measure(result, started: float) -> tuple[float, float, int]:
    total = (time.perf_counter() - started) * 1000
    return total, getattr(result, "_perf_parse_ms", 0.0), len(result.result_set)


def _server_ms(result) -> float:
    try:
        return float(result.run_time_ms)
    except Exception:  # probe only
        return float("nan")


def _capture(graph, q: str, params: dict, repeats: int) -> None:
    global _bench
    current = _captured.get(graph.name)
    if current is None or repeats >= current[3]:
        _captured[graph.name] = (graph, q, params, repeats)
    if _bench is None or _bench.done():
        _bench = asyncio.get_running_loop().create_task(_bench_loop())


async def _bench_loop() -> None:
    try:
        while True:
            await asyncio.sleep(5.0)
            if not _captured or time.monotonic() - _last_read < _IDLE_SECONDS:
                continue
            name, (graph, q, params, repeats) = _captured.popitem()
            origins = params[_ORIGINS]
            logger.info(
                "perf-bench start bfs graph=%s origins=%d unique=%d",
                name[:20],
                len(origins),
                len(set(origins)),
            )
            deduped = dict(params, **{_ORIGINS: list(dict.fromkeys(origins))})
            await _check(graph, q, params, deduped)
            for round_ in range(_ROUNDS):
                for variant, variant_params in (("as_is", params), ("dedup", deduped)):
                    await _time(graph, q, variant, variant_params, round_)
    except Exception:  # probe only
        logger.exception("perf-bench loop stopped")


async def _time(graph, q: str, variant: str, params: dict, round_: int) -> None:
    started = time.perf_counter()
    try:
        result = await _original_query(graph, q, params=params, read_only=True)
    except Exception as e:  # probe only
        logger.info(
            "perf-bench bfs %s graph=%s error=%s",
            variant,
            graph.name[:20],
            " ".join(str(e).split())[:160],
        )
        return
    total, parse, rows = _measure(result, started)
    logger.info(
        "perf-bench bfs %s total=%.0f exec=%.0f parse=%.0f server=%.1f rows=%d"
        " graph=%s round=%d",
        variant,
        total,
        total - parse,
        parse,
        _server_ms(result),
        rows,
        graph.name[:20],
        round_,
    )


async def _check(graph, q: str, params: dict, deduped: dict) -> None:
    try:
        as_is = await _rows(graph, q, params)
        dedup = await _rows(graph, q, deduped)
    except Exception as e:  # probe only
        logger.info("perf-bench check error=%s", " ".join(str(e).split())[:160])
        return
    logger.info(
        "perf-bench check bfs graph=%s rows=%d/%d same_order=%s same_set=%s"
        " same_attributes=%s",
        graph.name[:20],
        len(as_is),
        len(dedup),
        [uuid for uuid, _ in as_is] == [uuid for uuid, _ in dedup],
        {uuid for uuid, _ in as_is} == {uuid for uuid, _ in dedup},
        dict(as_is) == dict(dedup),
    )


async def _rows(graph, q: str, params: dict) -> list[tuple[str, dict]]:
    result = await _original_query(graph, q, params=params, read_only=True)
    names = [column[1] for column in result.header]
    uuid_at, attributes_at = names.index("uuid"), names.index("attributes")
    return [
        (row[uuid_at], dict(row[attributes_at] or {})) for row in result.result_set
    ]
