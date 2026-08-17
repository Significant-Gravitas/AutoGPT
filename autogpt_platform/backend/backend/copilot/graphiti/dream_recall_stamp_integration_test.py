"""Recall-stamp integration test: dedupe + prev-shift against live FalkorDB.

``_stamp_recall``'s 24h dedupe and its ``prev_recalled_at`` shift are what
make "recalled on two separate occasions within the window" mean anything —
the unit tests can only assert the query STRING, so a wrong comparison
operator, a missing ``IS NULL`` branch, or a mis-shifted property would keep
them green. These run the real Cypher.

Lives in ``graphiti/`` for the conftest that supplies ``falkordb_available``
(skip guard) and the no-op ``server``/``graph_cleanup`` overrides keeping the
suite off SpinTestServer.

Run with the stack up::

    cd autogpt_platform && docker compose up -d falkordb && cd backend
    poetry run pytest backend/copilot/graphiti/dream_recall_stamp_integration_test.py -xvs
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from backend.copilot.dream.ratification import _stamp_recall
from backend.copilot.dream.usage import RECALL_DEDUPE_INTERVAL

from .falkordb_driver import AutoGPTFalkorDriver

USER_ID = "u-stamp-integration"


async def _create_edge(
    driver: AutoGPTFalkorDriver,
    uuid: str,
    *,
    last_recalled_at: str | None = None,
    prev_recalled_at: str | None = None,
    expired_at: str | None = None,
) -> None:
    await driver.execute_query(
        """
        CREATE (:Entity {name: 'src'})-[:RELATES_TO {
            uuid: $uuid,
            fact: 'src relates to tgt',
            last_recalled_at: $last_recalled_at,
            prev_recalled_at: $prev_recalled_at,
            expired_at: $expired_at
        }]->(:Entity {name: 'tgt'})
        """,
        uuid=uuid,
        last_recalled_at=last_recalled_at,
        prev_recalled_at=prev_recalled_at,
        expired_at=expired_at,
    )


async def _read_edge(driver: AutoGPTFalkorDriver, uuid: str) -> dict[str, Any]:
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
        RETURN e.recall_count AS recall_count,
               e.last_recalled_at AS last_recalled_at,
               e.prev_recalled_at AS prev_recalled_at
        """,
        uuid=uuid,
    )
    assert len(records) == 1
    return records[0]


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_first_stamp_starts_the_counter_without_a_prior(clean_graph) -> None:
    """COALESCE is what makes pre-hook edges start at 0, not NULL."""
    driver, _ = clean_graph
    await _create_edge(driver, "fresh")

    assert await _stamp_recall(driver, ["fresh"], USER_ID) == 1

    edge = await _read_edge(driver, "fresh")
    assert edge["recall_count"] == 1
    assert edge["last_recalled_at"] is not None
    assert edge["prev_recalled_at"] is None, (
        "a first recall has no prior — a non-null prev here would make one "
        "recall look like two and protect the fact for a full window"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_stamp_inside_the_dedupe_interval_is_skipped(
    clean_graph,
) -> None:
    """Warm context re-pulls the same edge every turn: within one
    conversation those must collapse into a single use, or two turns
    would clear the two-occasion protection bar."""
    driver, _ = clean_graph
    recent = _ago(hours=1)
    await _create_edge(driver, "hot", last_recalled_at=recent)

    assert await _stamp_recall(driver, ["hot"], USER_ID) == 0

    edge = await _read_edge(driver, "hot")
    assert edge["recall_count"] is None, "skipped stamp must not increment"
    assert edge["last_recalled_at"] == recent, "skipped stamp must not re-date"
    assert edge["prev_recalled_at"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stamp_past_the_dedupe_interval_shifts_the_previous_stamp(
    clean_graph,
) -> None:
    """The (last, prev) pair the protection predicate reads."""
    driver, _ = clean_graph
    older = _ago(seconds=int(RECALL_DEDUPE_INTERVAL.total_seconds()) + 3600)
    await _create_edge(driver, "returning", last_recalled_at=older)

    assert await _stamp_recall(driver, ["returning"], USER_ID) == 1

    edge = await _read_edge(driver, "returning")
    assert edge["recall_count"] == 1
    assert edge["prev_recalled_at"] == older, (
        "the previous stamp must shift into prev_recalled_at — without it "
        "protected_fact_uuids can never see two recalls"
    )
    assert edge["last_recalled_at"] != older


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retracted_edges_are_never_restamped(clean_graph) -> None:
    """Re-stamping a superseded edge would misrepresent it as live."""
    driver, _ = clean_graph
    await _create_edge(driver, "retired", expired_at=_ago(days=2))

    assert await _stamp_recall(driver, ["retired"], USER_ID) == 0

    assert (await _read_edge(driver, "retired"))["recall_count"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_one_round_trip_stamps_each_eligible_edge_of_a_mixed_batch(
    clean_graph,
) -> None:
    """The batched UNWIND must apply the per-edge dedupe individually,
    not skip (or stamp) the whole set on one edge's state."""
    driver, _ = clean_graph
    await _create_edge(driver, "eligible")
    await _create_edge(driver, "deduped", last_recalled_at=_ago(hours=2))
    await _create_edge(driver, "expired-edge", expired_at=_ago(days=1))

    assert (
        await _stamp_recall(driver, ["eligible", "deduped", "expired-edge"], USER_ID)
        == 1
    )

    assert (await _read_edge(driver, "eligible"))["recall_count"] == 1
    assert (await _read_edge(driver, "deduped"))["recall_count"] is None
    assert (await _read_edge(driver, "expired-edge"))["recall_count"] is None
