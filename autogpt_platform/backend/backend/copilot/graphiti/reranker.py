"""Boolean/logprob reranking with a 16-token floor and non-reasoning Luna.

Luna omits the legacy tokenizer's logit bias; only True/False first-token
probabilities supply relevance scores. Unusable responses score zero.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import openai
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client import LLMConfig, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .models import DEFAULT_LLM_MODEL, is_luna_model

logger = logging.getLogger(__name__)

MIN_COMPLETION_TOKENS = 16

DEFAULT_MODEL = DEFAULT_LLM_MODEL

# Every candidate is scored at once, up to this many. graphiti's own default
# (``SEMAPHORE_LIMIT``, 20) splits a typical 40-candidate warm-context rerank
# into two waves, the second adding a classifier round trip (~1 s) to the
# chat's first turn.
MAX_PARALLEL_SCORES = 64


class CompatOpenAIRerankerClient(OpenAIRerankerClient):
    """Graphiti reranker with compatible request parameters per model."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        model = self.config.model or DEFAULT_MODEL
        options: dict[str, Any] = (
            {"reasoning_effort": "none", "max_completion_tokens": MIN_COMPLETION_TOKENS}
            if is_luna_model(model)
            else {
                "max_tokens": MIN_COMPLETION_TOKENS,
                "logit_bias": {"6432": 1, "7983": 1},
            }
        )
        openai_messages_list: list[list[ChatCompletionMessageParam]] = [
            [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=(
                        "You are an expert tasked with determining whether the "
                        "passage is relevant to the query"
                    ),
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        'Respond with "True" if PASSAGE is relevant to QUERY'
                        ' and "False" otherwise.\n'
                        f"<PASSAGE>\n{passage}\n</PASSAGE>\n"
                        f"<QUERY>\n{query}\n</QUERY>"
                    ),
                ),
            ]
            for passage in passages
        ]
        try:
            responses = await self._request_responses(
                model, options, openai_messages_list
            )

            responses_top_logprobs = [
                (
                    response.choices[0].logprobs.content[0].top_logprobs
                    if response.choices[0].logprobs is not None
                    and response.choices[0].logprobs.content
                    else []
                )
                for response in responses
            ]
            scores: list[float] = []
            non_boolean_count = 0
            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    # Keep scores aligned 1:1 with passages — a skipped
                    # entry desyncs the lists and makes the
                    # ``zip(..., strict=True)`` below raise ``ValueError``,
                    # aborting the whole rerank. Treat a missing classifier
                    # response as least-relevant (0.0).
                    scores.append(0.0)
                    continue
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                token = top_logprobs[0].token.strip().lower()
                if token == "true":
                    scores.append(norm_logprobs)
                elif token == "false":
                    scores.append(1 - norm_logprobs)
                else:
                    non_boolean_count += 1
                    scores.append(0.0)

            if non_boolean_count:
                logger.warning(
                    f"Reranker model {model} returned {non_boolean_count} "
                    "non-boolean responses; scoring zero"
                )

            results = [
                (passage, score)
                for passage, score in zip(passages, scores, strict=True)
            ]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise

    async def _request_responses(
        self,
        model: str,
        options: dict[str, Any],
        openai_messages_list: list[list[ChatCompletionMessageParam]],
    ) -> list[Any]:
        responses: list[Any] = [None] * len(openai_messages_list)
        requests = iter(enumerate(openai_messages_list))
        stopped = asyncio.Event()

        async def worker() -> None:
            while not stopped.is_set():
                try:
                    index, openai_messages = next(requests)
                except StopIteration:
                    return
                try:
                    responses[index] = await self.client.chat.completions.create(
                        model=model,
                        messages=openai_messages,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=2,
                        **options,
                    )
                except (Exception, asyncio.CancelledError):
                    stopped.set()
                    raise

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(len(openai_messages_list), MAX_PARALLEL_SCORES))
        ]
        try:
            await asyncio.gather(*tasks)
        except (Exception, asyncio.CancelledError):
            stopped.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return responses


__all__ = ["MIN_COMPLETION_TOKENS", "CompatOpenAIRerankerClient", "LLMConfig"]
