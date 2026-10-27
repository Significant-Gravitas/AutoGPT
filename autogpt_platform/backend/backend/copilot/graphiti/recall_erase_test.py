"""Unit tests for a hard forget's erasure (``recall_erase.py``), run by the
cascade over the in-memory graph of ``recall_cascade_fake.py``: what it
erases, the user's episode it never empties, and that a soft forget erases
nothing. The Cypher runs on FalkorDB in
``recall_cascade_erase_integration_test.py``.
"""

import pytest

from . import recall_cascade, recall_erase
from .memory_model import ForgetResult
from .recall_cascade_fake import CascadeGraph, chain

_NOW = "2026-09-28T00:00:00+00:00"


async def _cascade(graph: CascadeGraph, *, erase: bool) -> ForgetResult:
    result = ForgetResult(redacted_episodes=["e0"])
    await recall_cascade.cascade(graph, "user_a", ["f"], _NOW, result, erase=erase)
    return result


class TestHardForget:
    @pytest.mark.asyncio
    async def test_it_erases_each_derived_fact_and_dream_text_it_reaches(
        self,
    ) -> None:
        """Softly retracted, as ever, and each fact's text and each dream
        episode's body erased as the walk reaches them; the user's episode
        ``e0``, hidden for the forgotten fact, keeps its text."""
        graph = chain()

        result = await _cascade(graph, erase=True)

        assert (result.derived, result.failures) == (["c", "p", "q"], [])
        assert graph.erased == ["c", "dc", "p", "dp", "q", "dq"]
        assert "e0" in result.redacted_episodes and "e0" not in graph.erased

    @pytest.mark.asyncio
    async def test_it_erases_a_derived_fact_it_only_walks_through(self) -> None:
        graph = chain()
        graph.facts["c"].live = False

        result = await _cascade(graph, erase=True)

        assert result.derived == ["p", "q"]
        assert graph.facts["c"].reason is None, "not retracted"
        assert "c" in graph.scrubbed and "c" in graph.erased

    @pytest.mark.asyncio
    async def test_forgetting_again_erases_what_an_earlier_try_retracted(
        self,
    ) -> None:
        graph = chain()
        graph.facts["c"].live = False
        graph.facts["c"].reason = "derived_from_forgotten:f"

        await _cascade(graph, erase=True)

        assert graph.erased[0] == "c"

    @pytest.mark.asyncio
    async def test_a_soft_forget_erases_nothing(self) -> None:
        graph = chain()
        graph.facts["c"].live = False

        await _cascade(graph, erase=False)

        assert graph.erased == []
        assert graph.scrubbed == ["p", "q"], "a fact walked through keeps its text"
        erasing = {
            recall_erase.ERASE_FACTS_QUERY,
            recall_erase.ERASE_DREAM_EPISODES_QUERY,
        }
        assert not erasing & set(graph.queries)


class TestQueries:
    def test_a_fact_keeps_the_placeholder_and_blank_audit_copies(self) -> None:
        query = recall_erase.ERASE_FACTS_QUERY
        assert "e.fact = $placeholder" in query and "e.name = $placeholder" in query
        assert "e.fact_redacted = ''" in query and "e.name_redacted = ''" in query
        assert "e.fact_embedding = NULL" in query, "the vector encodes the sentence"

    def test_only_the_dreams_episodes_are_emptied(self) -> None:
        query = recall_erase.ERASE_DREAM_EPISODES_QUERY
        assert "ep.derived_from_facts IS NOT NULL" in query
        assert "ep.content = ''" in query
