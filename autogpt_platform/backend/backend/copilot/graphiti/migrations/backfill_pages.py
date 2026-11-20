"""Paged reads and batched writes for the derivation backfill
(``backfill_derivations.py``, ``backfill_cascade.py``).

A read pages by uuid (``$after``, ``$limit``), so a dry run never writes, nor
creates, a graph; a write sends ``BATCH_SIZE`` rows at a time.
"""

from typing import Any

from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

BATCH_SIZE = 500


async def pages(driver: AutoGPTFalkorDriver, query: str) -> list[dict[str, Any]]:
    """Every row of ``query``, read ``BATCH_SIZE`` at a time by uuid."""
    found: list[dict[str, Any]] = []
    after = ""
    while True:
        result = await driver.execute_query(query, after=after, limit=BATCH_SIZE)
        page = rows(result)
        found.extend(page)
        if len(page) < BATCH_SIZE:
            return found
        after = page[-1]["uuid"]


async def write_rows(
    driver: AutoGPTFalkorDriver, query: str, batch: list[dict[str, Any]]
) -> None:
    """``query`` over ``batch`` as ``$rows``, ``BATCH_SIZE`` rows at a time."""
    for start in range(0, len(batch), BATCH_SIZE):
        await driver.execute_query(query, rows=batch[start : start + BATCH_SIZE])


def rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []
