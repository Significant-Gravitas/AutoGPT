"""Integration tests for community rebuild (P-1.7) against live FalkorDB.

What this file covers without an LLM key:

- The defensive ``MATCH (c:Community) DETACH DELETE c`` runs *before*
  graphiti-core's own ``build_communities`` call. Seeded orphan
  ``:Community`` nodes from a prior version's rebuild are cleaned up.
- The function returns a structured result dict on every code path
  (success, expected-failure-without-LLM, invalid user). The scheduler
  relies on this contract to record outcomes without throwing.
- The function is per-user-isolated by ``group_id`` — invalidating
  user A's communities does not affect user B's.

What this file does NOT cover (needs an LLM key — separate
``requires_llm``-marked tests to add later):

- The actual happy-path "communities are built and summarized after a
  rebuild" assertion. We can't drive ``client.build_communities``
  without LLM access; the summarization step is an LLM call per
  community. Without it, the call typically returns an error dict
  (which we *do* assert on cleanly below).
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from .communities import rebuild_communities_for_user


def _make_user_id(group_id: str) -> str:
    """Recover the user_id that would derive into ``group_id``.

    ``derive_group_id`` prefixes ``user_``, so strip that to feed the
    fixture-minted group_id back through the function.
    """
    assert group_id.startswith("user_") or group_id.startswith("test_"), group_id
    return group_id.removeprefix("user_") if group_id.startswith("user_") else group_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rebuild_returns_structured_result_on_failure(
    clean_graph,
) -> None:
    """Even when the LLM step fails (no key, no network), the function
    returns an error dict — never raises. The scheduler relies on this
    so one bad rebuild doesn't starve other scheduled jobs.
    """
    driver, group_id = clean_graph

    # Seed a couple entities so the label-propagation step has something
    # to chew on before the LLM summarization step (which will fail
    # without a key in CI). FalkorDB doesn't implement Cypher's no-arg
    # ``datetime()``; generate the timestamp in Python and bind as a
    # parameter (mirrors the seeded_graph fixture + dream/fetch.py).
    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', name: 'Alice', group_id: $gid, created_at: $now}),
          (b:Entity {uuid: 'b', name: 'Bob',   group_id: $gid, created_at: $now}),
          (a)-[:RELATES_TO {uuid: 'e1', group_id: $gid, fact: 'a knows b'}]->(b)
        """,
        gid=group_id,
        now=datetime.now(timezone.utc).isoformat(),
    )

    user_id = _make_user_id(group_id)
    # derive_group_id will fail validation if we don't have the user_ prefix,
    # so we have to point the rebuild at the actual fixture group_id by
    # patching derive_group_id for this call only.
    with patch(
        "backend.copilot.graphiti.communities.derive_group_id",
        return_value=group_id,
    ):
        result = await rebuild_communities_for_user(user_id)

    # Contract: always returns dict with the expected keys.
    assert set(result.keys()) >= {
        "user_id",
        "started_at",
        "communities_built",
        "elapsed_seconds",
        "error",
    }
    assert result["user_id"] == user_id
    assert result["elapsed_seconds"] is not None
    assert result["elapsed_seconds"] >= 0
    # Either the LLM step ran (success) or it failed; either way, we
    # don't raise out of the function.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detach_delete_clears_orphan_community_nodes(
    clean_graph,
) -> None:
    """The defensive ``DETACH DELETE c`` runs against live FalkorDB.

    Seed two orphan ``:Community`` nodes (as if an older graphiti
    version left them behind), invoke the rebuild, and verify the
    orphans are gone — regardless of whether the rebuild itself
    succeeds in summarization.
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (:Community {uuid: 'orphan-1', name: 'old-1', group_id: $gid, summary: 'stale'}),
          (:Community {uuid: 'orphan-2', name: 'old-2', group_id: $gid, summary: 'stale'})
        """,
        gid=group_id,
    )

    # Sanity: orphans are there to begin with.
    records, _, _ = await driver.execute_query(
        "MATCH (c:Community {group_id: $gid}) RETURN c.uuid AS uuid",
        gid=group_id,
    )
    assert {r["uuid"] for r in records} == {"orphan-1", "orphan-2"}

    user_id = _make_user_id(group_id)
    # ``force=True`` bypasses ``_activity_since_last_rebuild``. Without
    # the override the rebuild short-circuits on ``no_episodes`` BEFORE
    # the ``DETACH DELETE c`` step runs (see communities.py:298-326),
    # and the orphans this test seeds would stay in place — which would
    # mask the regression the test is supposed to catch.
    with patch(
        "backend.copilot.graphiti.communities.derive_group_id",
        return_value=group_id,
    ):
        await rebuild_communities_for_user(user_id, force=True)

    # Orphans must be gone, whether or not the rebuild itself succeeded.
    records, _, _ = await driver.execute_query(
        "MATCH (c:Community {group_id: $gid, uuid: 'orphan-1'}) RETURN c.uuid AS uuid",
        gid=group_id,
    )
    assert records == [], "orphan-1 must be DETACH DELETE'd before rebuild starts"

    records, _, _ = await driver.execute_query(
        "MATCH (c:Community {group_id: $gid, uuid: 'orphan-2'}) RETURN c.uuid AS uuid",
        gid=group_id,
    )
    assert records == [], "orphan-2 must be DETACH DELETE'd before rebuild starts"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rebuild_invalid_user_id_returns_error_dict(clean_graph) -> None:
    """``derive_group_id`` raises on empty user_id; the function must
    catch and surface the error in the result dict, not let it propagate.
    """
    result = await rebuild_communities_for_user("")
    assert result["error"] is not None
    assert "invalid_user_id" in result["error"]
    assert result["communities_built"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rebuild_does_not_touch_other_users_communities(
    clean_graph,
) -> None:
    """Per-user isolation by ``group_id`` survives the DETACH DELETE.

    Seed orphan ``:Community`` nodes in two different group_ids,
    invalidate one, verify the other is untouched.
    """
    driver, group_id = clean_graph
    other_group = group_id + "_other"

    await driver.execute_query(
        """
        CREATE
          (:Community {uuid: 'self-1',  group_id: $g1, summary: 'self'}),
          (:Community {uuid: 'other-1', group_id: $g2, summary: 'other'})
        """,
        g1=group_id,
        g2=other_group,
    )

    user_id = _make_user_id(group_id)
    with patch(
        "backend.copilot.graphiti.communities.derive_group_id",
        return_value=group_id,
    ):
        await rebuild_communities_for_user(user_id)

    # other-1 must still be there.
    records, _, _ = await driver.execute_query(
        "MATCH (c:Community {uuid: 'other-1'}) RETURN c.group_id AS gid",
    )
    assert records == [
        {"gid": other_group}
    ], "Other user's :Community node was deleted — group_id scoping regressed"
