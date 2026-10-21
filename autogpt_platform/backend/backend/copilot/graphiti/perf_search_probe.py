"""Do-not-merge probe (#15739 integration preview), v2: where memory recall's
FalkorDB time goes.

- Each graphiti edge-search method's duration (``perf-search``), as in v1.
- Every labelled FalkorDB read split into server execution time (FalkorDB's
  own stat), the rest of the command (transfer and redis-py's protocol parse)
  and falkordb-py's result parse (``perf-q``).
- When the executor has been idle for a few seconds, the heaviest captured
  similarity and BFS reads are replayed with cheaper edge projections in place
  of ``properties(e)``, which carries every fact's embedding (``perf-bench``),
  and each variant's attributes are compared with the original's.
"""

import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)

_METHODS = ("edge_fulltext_search", "edge_similarity_search", "edge_bfs_search")
_ATTRIBUTES = "properties(e) AS attributes"
_VARIANTS = {
    "orig": _ATTRIBUTES,
    "null": "NULL AS attributes",
    "proj": "e {.*, fact_embedding: NULL} AS attributes",
    "pairs": "[k IN keys(e) WHERE k <> 'fact_embedding' | [k, e[k]]] AS attributes",
    "pairs2": (
        "[k IN keys(e) WHERE k <> 'fact_embedding' | [k, properties(e)[k]]]"
        " AS attributes"
    ),
}
_IDLE_SECONDS = 6.0
_ROUNDS = 5

_installed = False
_counted: set[str] = set()
_captured: dict[tuple[str, str], tuple[object, str, object, int]] = {}
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
    from redis.utils import HIREDIS_AVAILABLE

    logger.info("perf-q installed hiredis=%s", HIREDIS_AVAILABLE)
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
        label = _label(q)
        total, parse, rows = _measure(result, started)
        _last_read = time.monotonic()
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
                "" if label != "other" else " q=" + " ".join(q.split())[:90],
            )
        if label in ("sim", "bfs") and _ATTRIBUTES in q:
            _capture(self, label, q, params, rows)
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


def _capture(graph, label: str, q: str, params, rows: int) -> None:
    global _bench
    key = (graph.name, label)
    current = _captured.get(key)
    if current is None or rows >= current[3]:
        _captured[key] = (graph, q, params, rows)
    if _bench is None or _bench.done():
        _bench = asyncio.get_running_loop().create_task(_bench_loop(graph))


async def _bench_loop(first_graph) -> None:
    try:
        await _log_modules(first_graph)
        while True:
            await asyncio.sleep(1.0)
            if not _captured or time.monotonic() - _last_read < _IDLE_SECONDS:
                continue
            (name, label), (graph, q, params, rows) = _captured.popitem()
            logger.info(
                "perf-bench start %s graph=%s captured_rows=%d", label, name[:20], rows
            )
            await _check_variants(graph, label, q, params)
            await _time_variants(graph, label, q, params)
    except Exception:  # probe only
        logger.exception("perf-bench loop stopped")


async def _log_modules(graph) -> None:
    try:
        modules = await graph.execute_command("MODULE", "LIST")
        logger.info("perf-bench modules=%s", modules)
    except Exception as e:  # probe only
        logger.info("perf-bench modules=? (%s)", type(e).__name__)


async def _wait_idle() -> None:
    while time.monotonic() - _last_read < _IDLE_SECONDS:
        await asyncio.sleep(1.0)


async def _time_variants(graph, label: str, q: str, params) -> None:
    order = list(_VARIANTS)
    for round_ in range(_ROUNDS):
        await _wait_idle()
        random.shuffle(order)
        for variant in order:
            started = time.perf_counter()
            try:
                result = await _original_query(
                    graph,
                    q.replace(_ATTRIBUTES, _VARIANTS[variant]),
                    params=params,
                    read_only=True,
                )
            except Exception as e:  # probe only
                logger.info(
                    "perf-bench %s %s graph=%s error=%s",
                    label,
                    variant,
                    graph.name[:20],
                    " ".join(str(e).split())[:160],
                )
                continue
            total, parse, rows = _measure(result, started)
            logger.info(
                "perf-bench %s %s total=%.0f exec=%.0f parse=%.0f server=%.1f"
                " rows=%d graph=%s round=%d",
                label,
                variant,
                total,
                total - parse,
                parse,
                _server_ms(result),
                rows,
                graph.name[:20],
                round_,
            )


async def _check_variants(graph, label: str, q: str, params) -> None:
    """Does each cheaper projection carry the same attributes, embedding aside?"""
    baseline = await _attributes_by_uuid(graph, q, params)
    if baseline is None:
        return
    for variant in ("proj", "pairs", "pairs2"):
        candidate = await _attributes_by_uuid(
            graph, q.replace(_ATTRIBUTES, _VARIANTS[variant]), params
        )
        if candidate is None:
            continue
        same_rows = set(candidate) == set(baseline)
        diffs = sorted(
            {
                key
                for uuid in set(candidate) & set(baseline)
                for key in set(candidate[uuid]) ^ set(baseline[uuid])
                | {
                    k
                    for k in set(candidate[uuid]) & set(baseline[uuid])
                    if candidate[uuid][k] != baseline[uuid][k]
                }
            }
        )
        logger.info(
            "perf-bench check %s %s graph=%s rows=%d same_rows=%s differing_keys=%s",
            label,
            variant,
            graph.name[:20],
            len(candidate),
            same_rows,
            diffs,
        )


async def _attributes_by_uuid(graph, q: str, params) -> dict | None:
    try:
        result = await _original_query(graph, q, params=params, read_only=True)
    except Exception as e:  # probe only
        logger.info("perf-bench check error=%s", " ".join(str(e).split())[:160])
        return None
    names = [column[1] for column in result.header]
    uuid_at, attributes_at = names.index("uuid"), names.index("attributes")
    by_uuid = {}
    for row in result.result_set:
        attributes = row[attributes_at]
        if isinstance(attributes, list):
            attributes = {pair[0]: pair[1] for pair in attributes}
        attributes = dict(attributes or {})
        attributes.pop("fact_embedding", None)
        by_uuid[row[uuid_at]] = attributes
    return by_uuid
