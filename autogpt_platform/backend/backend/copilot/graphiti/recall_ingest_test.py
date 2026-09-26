"""Unit tests for ``recall_ingest``: graphiti's edge dedup never names a
forgotten edge, checked against graphiti's own dedup prompt.

The live runs, through the production worker, are
``recall_ingest_integration_test.py`` and ``recall_reteach_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.models import Message

from .client import _build_graphiti
from .recall import FORGOTTEN_FACT
from .recall_ingest import ForgetAwareLLMClient, forgotten_candidates, off_forgotten


def _dedup_prompt(existing: list[str], candidates: list[str]) -> list[Message]:
    """graphiti's edge dedup prompt, one idx numbering across both lists."""
    return prompt_library.dedupe_edges.resolve_edge(
        {
            "existing_edges": [{"idx": i, "fact": f} for i, f in enumerate(existing)],
            "new_edge": "Alice is assigned to work on the Atlas project",
            "edge_invalidation_candidates": [
                {"idx": len(existing) + i, "fact": f} for i, f in enumerate(candidates)
            ],
        }
    )


def _inner(answer: dict[str, Any]) -> MagicMock:
    inner = MagicMock(spec=LLMClient)
    inner.config = LLMConfig()
    inner.token_tracker = TokenUsageTracker()
    inner.generate_response = AsyncMock(return_value=answer)
    return inner


class TestForgottenCandidates:
    def test_finds_every_forgotten_edge_in_both_lists(self) -> None:
        prompt = _dedup_prompt(
            [FORGOTTEN_FACT, "Alice owns the Atlas budget"],
            ["Bob leads Atlas", FORGOTTEN_FACT],
        )

        assert forgotten_candidates(prompt) == {0, 3}

    def test_a_prompt_without_one_names_none(self) -> None:
        assert (
            forgotten_candidates(_dedup_prompt(["Alice works on Atlas"], [])) == set()
        )

    def test_quotes_in_other_facts_do_not_confuse_it(self) -> None:
        prompt = _dedup_prompt(['Alice\'s "big" budget', FORGOTTEN_FACT], [])

        assert forgotten_candidates(prompt) == {1}

    def test_lists_it_cannot_read_are_reported_as_such(self) -> None:
        prompt = [Message(role="user", content=f"<EXISTING FACTS>\n{FORGOTTEN_FACT}")]

        assert forgotten_candidates(prompt) is None


class TestOffForgotten:
    def test_a_forgotten_edge_is_neither_duplicate_nor_contradiction(self) -> None:
        answer = {"duplicate_facts": [0, 1], "contradicted_facts": [1, 3]}

        assert off_forgotten(answer, {0, 3}) == {
            "duplicate_facts": [1],
            "contradicted_facts": [1],
        }

    def test_unreadable_lists_keep_the_new_fact_new(self) -> None:
        answer = {"duplicate_facts": [1], "contradicted_facts": [2]}

        assert off_forgotten(answer, None) == {
            "duplicate_facts": [],
            "contradicted_facts": [],
        }

    def test_a_malformed_answer_is_left_for_graphiti_to_reject(self) -> None:
        assert off_forgotten({"duplicate_facts": "x"}, {0}) == {"duplicate_facts": "x"}


class TestForgetAwareLLMClient:
    @pytest.mark.asyncio
    async def test_edge_dedup_answers_are_kept_off_forgotten_edges(self) -> None:
        inner = _inner({"duplicate_facts": [0], "contradicted_facts": [1]})
        client = ForgetAwareLLMClient(inner)
        prompt = _dedup_prompt([FORGOTTEN_FACT], ["Bob leads Atlas"])

        answer = await client.generate_response(
            prompt,
            EdgeDuplicate,
            model_size=ModelSize.small,
            prompt_name="dedupe_edges.resolve_edge",
        )

        assert answer == {"duplicate_facts": [], "contradicted_facts": [1]}
        inner.generate_response.assert_awaited_once_with(
            prompt,
            EdgeDuplicate,
            None,
            ModelSize.small,
            None,
            "dedupe_edges.resolve_edge",
            attribute_extraction=False,
        )

    @pytest.mark.asyncio
    async def test_every_other_call_passes_through(self) -> None:
        answer = {"edges": [], "note": FORGOTTEN_FACT}
        client = ForgetAwareLLMClient(_inner(answer))
        prompt = [Message(role="user", content=f"extract: {FORGOTTEN_FACT}")]

        assert await client.generate_response(prompt, ExtractedEdges) == answer

    def test_it_shares_the_inner_clients_tracer_and_token_counts(self) -> None:
        inner = _inner({})
        client = ForgetAwareLLMClient(inner)
        tracer = MagicMock()

        client.set_tracer(tracer)

        inner.set_tracer.assert_called_once_with(tracer)
        assert client.token_tracker is inner.token_tracker


def test_every_graphiti_client_is_built_forget_aware() -> None:
    inner = _inner({})
    with patch("graphiti_core.Graphiti") as graphiti:
        _build_graphiti(
            "user_abc",
            inner,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            graph_driver=MagicMock(),
        )

    wrapped = graphiti.call_args.kwargs["llm_client"]
    assert isinstance(wrapped, ForgetAwareLLMClient) and wrapped.inner is inner
