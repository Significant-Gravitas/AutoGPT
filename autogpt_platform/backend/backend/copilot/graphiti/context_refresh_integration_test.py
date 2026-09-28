"""The follow-up warm-context refresh (SECRT-2378) against a live FalkorDB.

A fact the user forgets between two turns of a chat is in the first turn's
warm context and not in the next turn's refresh: the refresh reads through
the same recall policy as the first turn (``recall.search_facts``,
``recall.recent_episodes``, ``recall_recheck.recheck``), so neither the fact
nor the text of the chat turn it came from comes back, while a fact the
forget did not touch still does. Facts go in through graphiti's real
``add_episode`` with only the LLM boundary scripted
(``recall_integration_fixtures.py``). The unit sibling is
``context_test.py::TestRefreshReadsThroughTheRecallPolicy``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/context_refresh_integration_test.py
"""

import asyncio

import pytest

from . import context
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    ingest_facts,
    patch_recall_boundaries,
)

# The chat turn Alice's fact was extracted from. It says more than the fact
# sentence (``ALICE[2]``), so its absence shows the episode itself is hidden.
_ALICE_TURN = "Quick note for later: Alice works on Atlas, she joined in March"


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)
    # Budgets for a local container, not what these tests are about: a
    # timeout would read as "nothing recalled" and pass the negative checks.
    mocker.patch.object(context.graphiti_config, "context_timeout", 30.0)
    mocker.patch.object(context.graphiti_config, "context_refresh_timeout", 30.0)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
async def test_a_fact_forgotten_between_turns_is_not_refreshed(
    scope_graph, stub_graphiti_client, hard: bool
) -> None:
    driver, scope = scope_graph
    _, alice = await ingest_facts(
        driver,
        scope,
        stub_graphiti_client,
        [ALICE],
        session_id="s-1",
        body=_ALICE_TURN,
    )
    await ingest_facts(driver, scope, stub_graphiti_client, [BOB], session_id="s-2")

    first_turn = await context.fetch_warm_context(
        scope.owner_user_id, "Alice Bob Atlas"
    )
    await asyncio.gather(*context._pending_hit_tasks)
    assert first_turn is not None
    assert ALICE[2] in first_turn.split("<RECENT_EPISODES>")[0], "the fact"
    assert _ALICE_TURN in first_turn, "and the turn it came from"

    forgot = await retract(scope, [alice[ALICE[2]]], hard=hard)
    assert (forgot.deleted, forgot.failures) == ([alice[ALICE[2]]], [])

    refreshed = await context.refresh_warm_context(
        scope.owner_user_id, "who works on Atlas now"
    )

    assert refreshed is not None
    assert ALICE[2] not in refreshed, "the forgotten fact came back"
    assert _ALICE_TURN not in refreshed, "the text it came from came back"
    facts, episodes = refreshed.split("<RECENT_EPISODES>")
    assert BOB[2] in facts, "a fact the forget did not touch"
    assert BOB[2] in episodes, "and its episode"
