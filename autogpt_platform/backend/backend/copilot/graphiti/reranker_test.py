"""Tests for the OpenAI-compat cross-encoder reranker patch.

Dev outage: graphiti-core 0.28.2's stock ``OpenAIRerankerClient.rank``
sends ``max_tokens=1`` per boolean-classifier call; OpenAI/OpenRouter
upstreams now reject anything below 16 with a 400, which crashed every
warm-context fetch AND starved the hit-time ratification loop (the
hit hook runs after the search gather that raised).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.llm_client import LLMConfig

from backend.copilot.graphiti import client as client_mod
from backend.copilot.graphiti.config import GraphitiConfig
from backend.copilot.graphiti.reranker import CompatOpenAIRerankerClient


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-luna", "openai/gpt-6-luna"])
async def test_luna_reranker_disables_reasoning_without_legacy_token_bias(model):
    client, captured = _client_with([_logprob_response("True", -0.1)])
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model=model), client=client
    )

    ranked = await reranker.rank("query", ["passage"])

    assert ranked[0][1] > 0.9
    assert captured[0]["reasoning_effort"] == "none"
    assert captured[0]["max_completion_tokens"] == 16
    assert captured[0]["logprobs"] is True
    assert captured[0]["top_logprobs"] == 2
    assert "max_tokens" not in captured[0]
    assert "logit_bias" not in captured[0]


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
        config=LLMConfig(api_key="k", model="gpt-4.1-mini"), client=client
    )

    await reranker.rank("query", ["passage a", "passage b"])

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["max_tokens"] >= 16
        # The boolean-classifier mechanics must survive the patch —
        # Legacy logit bias nudges True/False and logprobs drive scores.
        assert kwargs["logit_bias"] == {"6432": 1, "7983": 1}
        assert kwargs["logprobs"] is True


@pytest.mark.asyncio
async def test_rank_scores_true_above_false_and_sorts(caplog):
    """Scoring parity with upstream: 'True' top-token scores exp(lp),
    'False' scores 1-exp(lp); results sorted descending."""
    client, _ = _client_with(
        [_logprob_response("False", -0.05), _logprob_response("True", -0.05)]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-mini"), client=client
    )

    ranked = await reranker.rank("query", ["irrelevant", "relevant"])

    assert [passage for passage, _ in ranked] == ["relevant", "irrelevant"]
    assert ranked[0][1] > ranked[1][1]
    assert not any(
        record.name == "backend.copilot.graphiti.reranker" for record in caplog.records
    )


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
        config=LLMConfig(api_key="k", model="gpt-4.1-mini"), client=client
    )

    ranked = await reranker.rank("query", ["relevant", "no_response"])

    assert [passage for passage, _ in ranked] == ["relevant", "no_response"]
    assert dict(ranked)["no_response"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-4.1-mini"])
@pytest.mark.parametrize("token", ["Yes", "No", "\n", '"', "True.", "True enough"])
async def test_rank_rejects_non_boolean_tokens(model, token):
    client, _ = _client_with(
        [
            _logprob_response(token, -0.7),
            _logprob_response("True", -0.05),
            _logprob_response("False", -0.05),
        ]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model=model), client=client
    )

    ranked = await reranker.rank("query", ["unscorable", "relevant", "irrelevant"])

    assert [passage for passage, _ in ranked] == [
        "relevant",
        "irrelevant",
        "unscorable",
    ]
    assert dict(ranked)["unscorable"] == 0.0
    assert dict(ranked)["relevant"] > 0.9
    assert 0 < dict(ranked)["irrelevant"] < 0.1


@pytest.mark.asyncio
async def test_rank_handles_empty_token_content():
    empty_response = _empty_logprob_response()
    empty_response.choices[0].logprobs.content = []
    client, _ = _client_with([empty_response, _logprob_response("True", -0.05)])
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-6-luna"), client=client
    )

    ranked = await reranker.rank("query", ["unscorable", "relevant"])

    assert [passage for passage, _ in ranked] == ["relevant", "unscorable"]
    assert dict(ranked)["unscorable"] == 0.0


@pytest.mark.asyncio
async def test_rank_aggregates_warnings_without_logging_token_content(caplog):
    tokens = ["private-output-one", "private-output-two"]
    client, _ = _client_with(
        [_logprob_response(token, -0.1) for token in tokens]
        + [_logprob_response("True", -0.05)]
    )
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-6-luna"), client=client
    )

    ranked = await reranker.rank("query", ["unscorable1", "unscorable2", "relevant"])

    warnings = [
        record
        for record in caplog.records
        if record.name == "backend.copilot.graphiti.reranker"
    ]
    assert len(warnings) == 1
    assert "2 non-boolean" in warnings[0].message
    assert "gpt-6-luna" in warnings[0].message
    assert all(token not in warnings[0].message for token in tokens)
    assert [passage for passage, _ in ranked] == [
        "relevant",
        "unscorable1",
        "unscorable2",
    ]
    assert dict(ranked)["unscorable1"] == dict(ranked)["unscorable2"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, expected_model",
    [
        (None, "gpt-6-luna"),
        ("", "gpt-6-luna"),
        ("custom-reranker", "custom-reranker"),
    ],
)
async def test_rank_uses_configured_model_or_default(model, expected_model):
    client, captured = _client_with([_logprob_response("True", -0.1)])
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model=model), client=client
    )

    await reranker.rank("query", ["passage"])

    assert captured[0]["model"] == expected_model


@pytest.mark.asyncio
async def test_build_graphiti_uses_compat_reranker(mocker):
    """client.py must construct the patched reranker, not the stock
    one — otherwise the 400s come back on the next refactor."""
    captured: dict = {}

    class _FakeGraphiti:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mocker.patch("graphiti_core.Graphiti", _FakeGraphiti)
    mocker.patch.object(
        client_mod,
        "graphiti_config",
        GraphitiConfig.model_construct(
            llm_api_key="k", llm_base_url="https://openrouter.ai/api/v1"
        ),
    )

    client_mod._build_graphiti(
        "user_test",
        llm_client=MagicMock(),
        embedder=MagicMock(),
        graph_driver=MagicMock(),
    )

    reranker = captured["cross_encoder"]
    assert isinstance(reranker, CompatOpenAIRerankerClient)
    client, requests = _client_with([_logprob_response("True", -0.1)])
    reranker.client = client

    await reranker.rank("query", ["passage"])

    assert requests[0]["model"] == "gpt-6-luna"
