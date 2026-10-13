"""Tests for the OpenAI-compat cross-encoder reranker patch.

Dev outage: graphiti-core 0.28.2's stock ``OpenAIRerankerClient.rank``
sends ``max_tokens=1`` per boolean-classifier call; OpenAI/OpenRouter
upstreams now reject anything below 16 with a 400, which crashed every
warm-context fetch AND starved the hit-time ratification loop (the
hit hook runs after the search gather that raised).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.llm_client import LLMConfig

from backend.copilot.graphiti.reranker import CompatOpenAIRerankerClient


def _logprob_response(token: str, logprob: float) -> SimpleNamespace:
    top = SimpleNamespace(token=token, logprob=logprob)
    content = SimpleNamespace(top_logprobs=[top])
    logprobs = SimpleNamespace(content=[content])
    choice = SimpleNamespace(logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


def _empty_logprob_response() -> SimpleNamespace:
    """A response whose first token has no top_logprobs entries — the
    classifier returned nothing usable for that passage."""
    content = SimpleNamespace(top_logprobs=[])
    logprobs = SimpleNamespace(content=[content])
    choice = SimpleNamespace(logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


def _client_with(responses: list[SimpleNamespace]) -> tuple[MagicMock, list[dict]]:
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        return responses[len(captured) - 1]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return client, captured


@pytest.mark.asyncio
async def test_rank_requests_at_least_sixteen_max_tokens():
    """The whole point of the subclass: OpenAI rejects max_tokens < 16
    on logprob calls, so every classifier request must send >= 16."""
    client, captured = _client_with(
        [_logprob_response("True", -0.1), _logprob_response("False", -0.2)]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )

    await reranker.rank("query", ["passage a", "passage b"])

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["max_tokens"] >= 16
        # The boolean-classifier mechanics must survive the patch —
        # logit_bias pins output to True/False and logprobs drive scores.
        assert kwargs["logit_bias"] == {"6432": 1, "7983": 1}
        assert kwargs["logprobs"] is True


@pytest.mark.asyncio
async def test_rank_scores_true_above_false_and_sorts():
    """Scoring parity with upstream: 'True' top-token scores exp(lp),
    'False' scores 1-exp(lp); results sorted descending."""
    client, _ = _client_with(
        [_logprob_response("False", -0.05), _logprob_response("True", -0.05)]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )

    ranked = await reranker.rank("query", ["irrelevant", "relevant"])

    assert [passage for passage, _ in ranked] == ["relevant", "irrelevant"]
    assert ranked[0][1] > ranked[1][1]


@pytest.mark.asyncio
async def test_rank_handles_empty_logprobs_without_desync():
    """An empty ``top_logprobs`` must not be silently skipped — doing so
    desyncs scores from passages and makes ``zip(..., strict=True)`` raise
    ``ValueError``, aborting the entire rerank. The passage with a missing
    classifier response should score lowest and sort last instead."""
    client, _ = _client_with(
        [_logprob_response("True", -0.05), _empty_logprob_response()]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )

    ranked = await reranker.rank("query", ["relevant", "no_response"])

    assert [passage for passage, _ in ranked] == ["relevant", "no_response"]
    assert dict(ranked)["no_response"] == 0.0


@pytest.mark.asyncio
async def test_rank_scores_every_candidate_at_once():
    """Forty candidates are scored in one wave, not two of twenty: a second
    wave adds a classifier round trip to the chat's first turn."""
    in_flight = 0
    most_in_flight = 0

    async def fake_create(**_kwargs):
        nonlocal in_flight, most_in_flight
        in_flight += 1
        most_in_flight = max(most_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _logprob_response("True", -0.1)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )

    ranked = await reranker.rank("query", [f"passage {n}" for n in range(40)])

    assert len(ranked) == 40
    assert most_in_flight == 40


def test_build_graphiti_uses_compat_reranker(mocker):
    """client.py must construct the patched reranker, not the stock
    one — otherwise the 400s come back on the next refactor."""
    from backend.copilot.graphiti import client as client_mod

    captured: dict = {}

    class _FakeGraphiti:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mocker.patch("graphiti_core.Graphiti", _FakeGraphiti)
    mocker.patch(
        "backend.copilot.graphiti.falkordb_driver.AutoGPTFalkorDriver",
        MagicMock(),
    )

    client_mod._build_graphiti("user_test", llm_client=MagicMock())

    assert isinstance(captured["cross_encoder"], CompatOpenAIRerankerClient)
