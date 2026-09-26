"""Unit tests for ``recall_hide``: what a forget scrubs and redacts beyond
the edge, against a mock driver.

The live runs are ``recall_forget_integration_test.py`` (the episodes) and
``ingest_recall_integration_test.py`` (graphiti's own prompts).
"""

from unittest.mock import AsyncMock

import pytest

from . import recall_hide
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT, forgotten_facts_clause, recallable_episode_predicate
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-abc")
_NOW = "2026-09-26T12:00:00+00:00"


def _driver(*results) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, [], None) for r in results
    ]
    return driver


class TestHide:
    @pytest.mark.asyncio
    async def test_scrubs_the_facts_then_redacts_their_episodes(self) -> None:
        driver = _driver([], [{"uuid": "ep1"}])
        result = ForgetResult()

        hidden = await recall_hide.hide(driver, _SCOPE, ["u1", "u2"], _NOW, result)

        assert hidden is True
        assert result.redacted_episodes == ["ep1"] and result.failures == []
        scrub, redact = driver.execute_query.await_args_list
        assert scrub.args == (recall_hide.SCRUB_FACTS_QUERY,)
        assert scrub.kwargs == {"uuids": ["u1", "u2"], "placeholder": FORGOTTEN_FACT}
        assert redact.args == (recall_hide.REDACT_EPISODES_QUERY,)
        assert redact.kwargs == {"uuids": ["u1", "u2"], "now": _NOW}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing", [0, 1], ids=["scrub", "redaction"])
    async def test_a_failed_write_is_each_edges_cleanup_error(
        self, failing: int
    ) -> None:
        results: list = [[], []]
        results[failing] = RuntimeError("down")
        driver = _driver(*results)
        result = ForgetResult()

        hidden = await recall_hide.hide(driver, _SCOPE, ["u1", "u2"], _NOW, result)

        assert hidden is False
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("u1", MemoryForgetFailureCode.CLEANUP_ERROR),
            ("u2", MemoryForgetFailureCode.CLEANUP_ERROR),
        ]
        assert "Forgetting it again is safe" in result.failures[0].reason

    @pytest.mark.asyncio
    async def test_nothing_retracted_means_nothing_to_hide(self) -> None:
        driver = _driver()

        assert await recall_hide.hide(driver, _SCOPE, [], _NOW, ForgetResult())
        driver.execute_query.assert_not_awaited()


class TestScrubQuery:
    def test_moves_fact_and_name_to_their_audit_copies(self) -> None:
        """A repeat reads the audit copies back, so it changes nothing."""
        query = recall_hide.SCRUB_FACTS_QUERY
        assert "WHERE e.uuid IN $uuids" in query
        assert "coalesce(e.fact_redacted, e.fact) AS sentence" in query
        assert "coalesce(e.name_redacted, e.name) AS relation" in query
        assert "SET e.fact_redacted = sentence," in query
        assert "e.name_redacted = relation," in query
        assert "e.fact = $placeholder," in query
        assert "e.name = $placeholder," in query

    def test_blanks_the_summaries_graphiti_built_from_the_sentence(self) -> None:
        query = recall_hide.SCRUB_FACTS_QUERY
        assert "source.summary = ''" in query
        assert "target.summary = ''" in query
        assert "OPTIONAL MATCH (c:Community)-[:HAS_MEMBER]->(member)" in query
        assert "WHERE member IN ends" in query
        assert "SET c.summary = ''" in query

    def test_writes_no_status_or_marker(self) -> None:
        """The retraction owns those; the scrub only moves text."""
        query = recall_hide.SCRUB_FACTS_QUERY
        for field in ("status", "forgotten_at", "expired_at", "invalid_at"):
            assert field not in query


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
