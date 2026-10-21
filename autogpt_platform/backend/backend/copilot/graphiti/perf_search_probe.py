"""Do-not-merge probe (#15739 integration preview): log each graphiti
edge-search method's duration and, once per graph, its fact count, to see
where memory recall's FalkorDB time goes (new chats and follow-up refreshes)."""

import logging
import time

logger = logging.getLogger(__name__)

_METHODS = ("edge_fulltext_search", "edge_similarity_search", "edge_bfs_search")
_installed = False
_counted: set[str] = set()


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    from graphiti_core.search import search as graphiti_search

    for name in _METHODS:
        _wrap(graphiti_search, name)


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
