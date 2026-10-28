"""Public reranker concurrency, response association, and cleanup contracts."""

from __future__ import annotations

import asyncio
import gc
from math import log
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.llm_client import LLMConfig

from backend.copilot.graphiti.reranker import (
    MAX_PARALLEL_SCORES,
    CompatOpenAIRerankerClient,
)


@pytest.mark.parametrize(
    "candidates, peak",
    [(40, 40), (MAX_PARALLEL_SCORES + 36, MAX_PARALLEL_SCORES)],
)
@pytest.mark.asyncio
async def test_rank_scores_every_candidate_at_once_up_to_the_cap(candidates, peak):
    """Forty candidates are scored in one wave, not two of twenty: a second
    wave adds a classifier round trip to the chat's first turn. Past the cap,
    the rest wait for a free slot."""
    in_flight = 0
    most_in_flight = 0
    full_wave = asyncio.Event()
    release_scores = asyncio.Event()

    async def fake_create(**_kwargs):
        nonlocal in_flight, most_in_flight
        in_flight += 1
        most_in_flight = max(most_in_flight, in_flight)
        if in_flight == peak:
            full_wave.set()
        await release_scores.wait()
        in_flight -= 1
        return _logprob_response()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )

    ranking = asyncio.create_task(
        reranker.rank("query", [f"passage {n}" for n in range(candidates)])
    )
    try:
        await asyncio.wait_for(full_wave.wait(), timeout=5)
        await asyncio.sleep(0)
    finally:
        release_scores.set()
        ranked = await ranking

    assert len(ranked) == candidates
    assert most_in_flight == peak


@pytest.mark.asyncio
async def test_rank_cancels_and_drains_siblings_after_provider_error(recwarn):
    started = 0
    finished = 0
    cancelled = 0
    active_wave_ready = asyncio.Event()
    fail_first = asyncio.Event()
    release_siblings = asyncio.Event()

    async def fake_create(**_kwargs):
        nonlocal started, finished, cancelled
        call_index = started
        started += 1
        if started == MAX_PARALLEL_SCORES:
            active_wave_ready.set()
        try:
            if call_index == 0:
                await fail_first.wait()
                raise RuntimeError("provider failed")
            await release_siblings.wait()
            return _logprob_response()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            finished += 1

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )
    baseline_tasks = asyncio.all_tasks()
    ranking = asyncio.create_task(
        reranker.rank(
            "query",
            [f"passage {n}" for n in range(MAX_PARALLEL_SCORES + 1)],
        )
    )

    try:
        await asyncio.wait_for(active_wave_ready.wait(), timeout=5)
        fail_first.set()
        with pytest.raises(RuntimeError, match="provider failed"):
            await asyncio.wait_for(ranking, timeout=5)

        started_when_rank_returned = started
        pending_when_rank_returned = asyncio.all_tasks() - baseline_tasks
        release_siblings.set()
        await asyncio.gather(*pending_when_rank_returned, return_exceptions=True)
    finally:
        fail_first.set()
        release_siblings.set()
        if not ranking.done():
            ranking.cancel()
        await asyncio.gather(ranking, return_exceptions=True)
    gc.collect()

    assert started_when_rank_returned == MAX_PARALLEL_SCORES
    assert started == started_when_rank_returned
    assert finished == started
    assert cancelled == MAX_PARALLEL_SCORES - 1
    assert not (asyncio.all_tasks() - baseline_tasks)
    assert not [
        warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)
    ]


@pytest.mark.asyncio
async def test_rank_cancellation_drains_active_and_queued_requests(recwarn):
    started = 0
    finished = 0
    cancelled = 0
    active_wave_ready = asyncio.Event()
    release_scores = asyncio.Event()

    async def fake_create(**_kwargs):
        nonlocal started, finished, cancelled
        started += 1
        if started == MAX_PARALLEL_SCORES:
            active_wave_ready.set()
        try:
            await release_scores.wait()
            return _logprob_response()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            finished += 1

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )
    baseline_tasks = asyncio.all_tasks()
    ranking = asyncio.create_task(
        reranker.rank(
            "query",
            [f"passage {n}" for n in range(MAX_PARALLEL_SCORES + 36)],
        )
    )

    try:
        await asyncio.wait_for(active_wave_ready.wait(), timeout=5)
        ranking.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(ranking, timeout=5)
    finally:
        release_scores.set()
        if not ranking.done():
            ranking.cancel()
        await asyncio.gather(ranking, return_exceptions=True)
    del ranking
    gc.collect()

    assert started == MAX_PARALLEL_SCORES
    assert finished == started
    assert cancelled == started
    assert not (asyncio.all_tasks() - baseline_tasks)
    assert not [
        warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)
    ]


@pytest.mark.asyncio
async def test_rank_preserves_passage_scores_when_responses_finish_in_reverse():
    scores = {"passage a": 0.2, "passage b": 0.9, "passage c": 0.6}
    started = {passage: asyncio.Event() for passage in scores}
    release = {passage: asyncio.Event() for passage in scores}
    finished = {passage: asyncio.Event() for passage in scores}
    completion_order = []

    async def fake_create(**kwargs):
        content = kwargs["messages"][1]["content"]
        passage = content.split("<PASSAGE>", 1)[1].split("</PASSAGE>", 1)[0].strip()
        started[passage].set()
        await release[passage].wait()
        completion_order.append(passage)
        finished[passage].set()
        return _logprob_response(log(scores[passage]))

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    reranker = CompatOpenAIRerankerClient(
        config=LLMConfig(api_key="k", model="gpt-4.1-nano"), client=client
    )
    ranking = asyncio.create_task(reranker.rank("query", list(scores)))

    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())), timeout=5
        )
        for passage in reversed(scores):
            release[passage].set()
            await asyncio.wait_for(finished[passage].wait(), timeout=5)
        ranked = await asyncio.wait_for(ranking, timeout=5)
    finally:
        for event in release.values():
            event.set()
        if not ranking.done():
            ranking.cancel()
        await asyncio.gather(ranking, return_exceptions=True)

    assert completion_order == list(reversed(scores))
    assert dict(ranked) == pytest.approx(scores)
    assert [passage for passage, _ in ranked] == ["passage b", "passage c", "passage a"]


def _logprob_response(logprob: float = -0.1) -> SimpleNamespace:
    top = SimpleNamespace(token="True", logprob=logprob)
    content = SimpleNamespace(top_logprobs=[top])
    logprobs = SimpleNamespace(content=[content])
    choice = SimpleNamespace(logprobs=logprobs)
    return SimpleNamespace(choices=[choice])
