"""Erase a memory scope's whole graph: the Settings > Memory erase and an
expert's reset both go through here."""

from redis.exceptions import ResponseError

from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

_MISSING_GRAPH_MARKERS = ("no such graph", "does not exist", "invalid graph")


def is_missing_graph_error(exc: BaseException) -> bool:
    """Whether FalkorDB refused a query because the scope has no graph yet."""
    if not isinstance(exc, ResponseError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_GRAPH_MARKERS)


async def erase_scope(scope: MemoryScope) -> int:
    """Delete every node in *scope*'s graph; returns how many there were.

    A scope that never had a graph has nothing to erase. A missing-graph error
    on the delete means the graph vanished after the count: it is gone either
    way, so the counted value stands rather than misreporting 0.
    """
    driver = open_driver(scope)
    deleted = 0
    try:
        result = await driver.execute_query("MATCH (n) RETURN count(n) AS c")
        rows = result[0] if result else []
        deleted = int(rows[0]["c"]) if rows else 0
        if deleted:
            await driver.execute_query("MATCH (n) DETACH DELETE n")
    except ResponseError as exc:
        if not is_missing_graph_error(exc):
            raise
    finally:
        await driver.close()
    return deleted
