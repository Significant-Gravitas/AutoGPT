"""Recall stamps against a live FalkorDB: the dedupe, the shift of the
previous stamp, the live-only rule (a forget then a recall never stamps the
forgotten fact again), and the reads the dream makes of them.

The unit tests (``recall_stamp_test.py``) can only pin the Cypher's text; a
wrong comparison, a missing ``IS NULL`` branch or a mis-shifted property
would keep them green. These run it. Adapted from #13776's
``dream_recall_stamp_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_stamp_integration_test.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.copilot.dream import ratification

from . import context
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import ALICE, ingest_facts, patch_recall_boundaries
from .recall_stamp import (
    RECALL_DEDUPE_INTERVAL,
    read_recall_stamps,
    stamp_recalls,
    stamp_time,
)

_OWNER = "u-stamp-integration"


async def _edge(
    driver: AutoGPTFalkorDriver, group_id: str, uuid: str, **props: Any
) -> None:
    """A live fact between two entities of its own, with *props* set on it."""
    await driver.execute_query(
        """
        CREATE (:Entity {uuid: $uuid + '-src', name: 'src', group_id: $g})
               -[e:RELATES_TO {uuid: $uuid, group_id: $g, fact: 'src knows tgt',
                               name: 'knows', status: 'active',
                               created_at: $created}]->
               (:Entity {uuid: $uuid + '-tgt', name: 'tgt', group_id: $g})
        SET e += $props
        """,
        uuid=uuid,
        g=group_id,
        created=stamp_time(datetime.now(timezone.utc)),
        props=props,
    )


async def _stamps(driver: AutoGPTFalkorDriver, uuid: str) -> dict[str, Any]:
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
        RETURN e.recall_count AS recall_count,
               e.last_recalled_at AS last_recalled_at,
               e.prev_recalled_at AS prev_recalled_at
        """,
        uuid=uuid,
    )
    assert result is not None
    [row] = result[0]
    return row


def _ago(**kwargs: float) -> str:
    return stamp_time(datetime.now(timezone.utc) - timedelta(**kwargs))


_PAST_THE_DEDUPE = {"seconds": RECALL_DEDUPE_INTERVAL.total_seconds() + 3600}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_first_recall_starts_the_count_with_no_prior(clean_graph) -> None:
    driver, group_id = clean_graph
    await _edge(driver, group_id, "fresh")

    assert await stamp_recalls(driver, ["fresh"], owner=_OWNER) == 1

    edge = await _stamps(driver, "fresh")
    assert edge["recall_count"] == 1
    assert edge["last_recalled_at"] is not None
    assert edge["prev_recalled_at"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_recall_inside_the_dedupe_interval_is_not_counted(
    clean_graph,
) -> None:
    driver, group_id = clean_graph
    recent = _ago(hours=1)
    await _edge(driver, group_id, "hot", recall_count=1, last_recalled_at=recent)

    assert await stamp_recalls(driver, ["hot"], owner=_OWNER) == 0

    assert await _stamps(driver, "hot") == {
        "recall_count": 1,
        "last_recalled_at": recent,
        "prev_recalled_at": None,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_recall_past_the_interval_counts_and_shifts_the_last_one(
    clean_graph,
) -> None:
    driver, group_id = clean_graph
    older, oldest = _ago(**_PAST_THE_DEDUPE), _ago(days=6)
    await _edge(
        driver,
        group_id,
        "veteran",
        recall_count=7,
        last_recalled_at=older,
        prev_recalled_at=oldest,
    )

    assert await stamp_recalls(driver, ["veteran"], owner=_OWNER) == 1

    edge = await _stamps(driver, "veteran")
    assert edge["recall_count"] == 8, "an existing count grows, never resets"
    assert edge["prev_recalled_at"] == older
    assert edge["last_recalled_at"] > older


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "props",
    [
        pytest.param({"forgotten_at": _ago(days=1)}, id="forgotten"),
        pytest.param({"status": "retracted"}, id="retracted"),
        pytest.param({"expired_at": _ago(days=1)}, id="legacy-forget"),
        pytest.param(
            {
                "expired_at": _ago(days=1),
                "status": "superseded",
                "expiration_reason": "stale_fact",
            },
            id="expired",
        ),
        pytest.param({"status": "contradicted"}, id="contradicted"),
    ],
)
async def test_a_fact_that_is_not_live_is_never_stamped(
    clean_graph, props: dict[str, Any]
) -> None:
    driver, group_id = clean_graph
    await _edge(driver, group_id, "retired", **props)

    assert await stamp_recalls(driver, ["retired"], owner=_OWNER) == 0

    assert (await _stamps(driver, "retired"))["recall_count"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_one_write_applies_the_rules_edge_by_edge(clean_graph) -> None:
    driver, group_id = clean_graph
    await _edge(driver, group_id, "eligible")
    await _edge(
        driver, group_id, "deduped", recall_count=1, last_recalled_at=_ago(hours=2)
    )
    await _edge(driver, group_id, "forgotten", forgotten_at=_ago(days=1))

    stamped = await stamp_recalls(
        driver, ["eligible", "deduped", "forgotten", "missing"], owner=_OWNER
    )

    assert stamped == 1
    assert (await _stamps(driver, "eligible"))["recall_count"] == 1
    assert (await _stamps(driver, "deduped"))["recall_count"] == 1
    assert (await _stamps(driver, "forgotten"))["recall_count"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_then_a_recall_never_stamps_the_forgotten_fact(
    mocker, scope_graph, stub_graphiti_client
) -> None:
    """A fact warm context stamped, then forgotten: a recall hook that still
    names it (a search that began before the forget) writes nothing on it,
    and the dream's re-read of the stamps no longer sees it."""
    driver, scope = scope_graph
    patch_recall_boundaries(mocker, driver, stub_graphiti_client)
    mocker.patch.object(ratification, "record_memory_hit", AsyncMock())
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    fact = edges[ALICE[2]]

    assert await context._fetch(scope, "Atlas")
    await asyncio.gather(*context._pending_hit_tasks)
    stamped = await _stamps(driver, fact)
    assert stamped["recall_count"] == 1

    await retract(scope, [fact])
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.last_recalled_at = $old",
        uuid=fact,
        old=_ago(**_PAST_THE_DEDUPE),
    )
    await ratification.try_ratify_on_hit(scope, [fact])

    after = await _stamps(driver, fact)
    assert after["recall_count"] == 1
    assert after["prev_recalled_at"] is None
    assert await read_recall_stamps(driver, scope.group_id, [fact]) == []
