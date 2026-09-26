"""Unit tests for ``recall_hide``: what a forget scrubs and redacts beyond
the edge, against a mock driver.

The live runs are ``recall_forget_integration_test.py`` (the episodes) and
``recall_scrub_integration_test.py`` / ``ingest_recall_integration_test.py``
(graphiti's own prompts).
"""

from unittest.mock import AsyncMock

import pytest

from . import recall_hide
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT, forgotten_facts_clause, recallable_episode_predicate
from .recall_hide import Hiding

_GROUP = "user_abc"
_NOW = "2026-09-26T12:00:00+00:00"
_ALICE_KEYS = [
    "uuid",
    "name",
    "group_id",
    "summary",
    "created_at",
    "role",
    "attributes",
]


def _driver(*results) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, [], None) for r in results
    ]
    return driver


class TestHide:
    @pytest.mark.asyncio
    async def test_scrubs_facts_then_entities_then_redacts(self) -> None:
        driver = _driver(
            [{"ends": ["alice", "atlas"]}],
            [{"uuid": "alice", "keys": _ALICE_KEYS}],
            [],
            [{"uuid": "ep1"}],
        )
        hiding = Hiding(uuids=["u1"], recovered=[["u1", "sentence", "MemoryFact"]])
        result = ForgetResult()

        assert await recall_hide.hide(driver, _GROUP, hiding, _NOW, result)

        assert result.redacted_episodes == ["ep1"] and result.failures == []
        facts, keys, entities, redact = driver.execute_query.await_args_list
        assert facts.args == (recall_hide.SCRUB_FACTS_QUERY,)
        assert facts.kwargs == {
            "uuids": ["u1"],
            "placeholder": FORGOTTEN_FACT,
            "recovered": [["u1", "sentence", "MemoryFact"]],
        }
        assert keys.kwargs == {"uuids": ["u1"], "entities": ["alice", "atlas"]}
        assert entities.kwargs == {
            "entities": [
                {"uuid": "alice", "cleared": {"role": None, "attributes": None}}
            ]
        }
        assert redact.args == (recall_hide.REDACT_EPISODES_QUERY,)
        assert redact.kwargs == {"uuids": ["u1"], "now": _NOW}

    @pytest.mark.asyncio
    async def test_entities_beyond_the_endpoints_are_scrubbed_too(self) -> None:
        """A hard forget's endpoints, when its edge is gone."""
        driver = _driver([{"ends": []}], [], [])

        await recall_hide.hide(
            driver, _GROUP, Hiding(uuids=["u1"], entities=["bob"]), _NOW, ForgetResult()
        )

        assert driver.execute_query.await_args_list[1].kwargs["entities"] == ["bob"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failing", [0, 1, 2], ids=["facts", "entities", "redaction"]
    )
    async def test_a_failed_write_is_each_edges_cleanup_error(
        self, failing: int
    ) -> None:
        results: list = [[{"ends": []}], [], []]
        results[failing] = RuntimeError("down")
        driver = _driver(*results)
        result = ForgetResult()

        hidden = await recall_hide.hide(
            driver, _GROUP, Hiding(uuids=["u1", "u2"]), _NOW, result
        )

        assert hidden is False
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("u1", MemoryForgetFailureCode.CLEANUP_ERROR),
            ("u2", MemoryForgetFailureCode.CLEANUP_ERROR),
        ]
        assert "Forgetting it again is safe" in result.failures[0].reason

    @pytest.mark.asyncio
    async def test_nothing_retracted_means_nothing_to_hide(self) -> None:
        driver = _driver()

        assert await recall_hide.hide(
            driver, _GROUP, Hiding(uuids=[]), _NOW, ForgetResult()
        )
        driver.execute_query.assert_not_awaited()


class TestScrubQuery:
    def test_an_audit_copy_is_never_replaced_by_the_placeholder(self) -> None:
        """The audit copy already written wins, then the edge's own text
        unless it reads the placeholder, then the text the stash kept."""
        query = recall_hide.SCRUB_FACTS_QUERY
        assert (
            "coalesce(e.fact_redacted,\n"
            "              CASE WHEN e.fact <> $placeholder THEN e.fact END,\n"
            "              [r IN $recovered WHERE r[0] = e.uuid | r[1]][0]) AS sentence"
            in query
        )
        assert "CASE WHEN e.name <> $placeholder THEN e.name END" in query
        assert "SET e.fact_redacted = sentence," in query
        assert "e.fact = $placeholder," in query
        assert "e.name = $placeholder" in query

    def test_writes_no_status_or_marker(self) -> None:
        """The retraction owns those; the scrub only moves text."""
        for query in (recall_hide.SCRUB_FACTS_QUERY, recall_hide._SCRUB_ENTITIES_QUERY):
            for field in ("status", "forgotten_at", "expired_at", "invalid_at"):
                assert field not in query


class TestEntityScrub:
    def test_reaches_every_entity_a_hidden_episode_mentions(self) -> None:
        query = recall_hide._ENTITY_KEYS_QUERY
        assert "(ep:Episodic)-[:MENTIONS]->(mentioned:Entity)" in query
        assert "any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)" in query
        assert "collect(DISTINCT mentioned.uuid) + $entities AS targets" in query
        assert "keys(n) AS keys" in query

    def test_clears_attributes_and_summaries_and_their_communities(self) -> None:
        query = recall_hide._SCRUB_ENTITIES_QUERY
        assert "SET n += entity.cleared, n.summary = ''" in query
        assert "OPTIONAL MATCH (c:Community)-[:HAS_MEMBER]->(member)" in query
        assert "SET c.summary = ''" in query

    @pytest.mark.asyncio
    async def test_an_entity_keeps_only_its_identity(self) -> None:
        keys = [*_ALICE_KEYS, "labels", "name_embedding", "email"]
        driver = _driver([{"uuid": "alice", "keys": keys}], [])

        await recall_hide.scrub_entities(driver, ["u1"], [])

        cleared = driver.execute_query.await_args_list[1].kwargs["entities"][0]
        assert cleared["cleared"] == {"role": None, "attributes": None, "email": None}

    @pytest.mark.asyncio
    async def test_no_entity_found_writes_nothing(self) -> None:
        driver = _driver([])

        await recall_hide.scrub_entities(driver, ["u1"], [])

        assert driver.execute_query.await_count == 1


class TestRedactQuery:
    def test_redacts_every_episode_the_policy_now_hides(self) -> None:
        """Any episode naming a forgotten fact, not only one left with no
        live fact: the redaction and the read side share one predicate."""
        query = recall_hide.REDACT_EPISODES_QUERY
        assert query.startswith(forgotten_facts_clause())
        assert "any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)" in query
        assert f"NOT ({recallable_episode_predicate('ep')})" in query
        assert "SET ep.redacted_at = coalesce(ep.redacted_at, $now)" in query
        assert "content" not in query, "the text stays for audit"
