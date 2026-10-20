"""Boolean/logprob reranking with a 16-token floor and non-reasoning Luna.

Luna omits the legacy tokenizer's logit bias; only True/False first-token
probabilities supply relevance scores. Unusable responses score zero.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any

import numpy as np
import openai
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.helpers import SEMAPHORE_LIMIT, semaphore_gather
from graphiti_core.llm_client import LLMConfig, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from backend.copilot import perf_legs as _perf_legs  # do-not-merge probe (#15739)

from .config import graphiti_config
from .models import DEFAULT_LLM_MODEL, is_luna_model

logger = logging.getLogger(__name__)

_PROBE_RANKS = itertools.count()
_PROBE_ARMS = ("or20", "or1w", "direct1w", "mini1w")
_probe_direct: openai.AsyncOpenAI | None = None


def _probe_direct_client() -> openai.AsyncOpenAI:
    """OpenAI directly, with the key the embedder already uses."""
    global _probe_direct
    if _probe_direct is None:
        _probe_direct = openai.AsyncOpenAI(
            api_key=graphiti_config.resolve_embedder_api_key()
        )
    return _probe_direct

MIN_COMPLETION_TOKENS = 16

DEFAULT_MODEL = DEFAULT_LLM_MODEL


class CompatOpenAIRerankerClient(OpenAIRerankerClient):
    """Graphiti reranker with compatible request parameters per model."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        model = self.config.model or DEFAULT_MODEL
        # do-not-merge probe (#15739): successive pairs of ranks rotate through
        # graphiti's default cap of 20 at a time, one wave (#15492's cap), one
        # wave sent to OpenAI directly, and one wave on GPT-4.1 mini.
        arm = _PROBE_ARMS[(next(_PROBE_RANKS) // 2) % len(_PROBE_ARMS)]
        client = self.client
        if arm == "direct1w":
            client = _probe_direct_client()
            model = model.removeprefix("openai/")
        elif arm == "mini1w":
            model = "openai/gpt-4.1-mini"
        cap = None if arm == "or20" else max(1, min(len(passages), 64))
        _perf_legs.note("rr_arm", arm)
        _perf_legs.note("rr_model", model)
        _perf_legs.note("rr_passages", len(passages))
        _perf_legs.note("rr_cap", cap or SEMAPHORE_LIMIT)
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
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]
        async def _probe_call(call):
            started = time.time()
            try:
                response = await call
                provider = (getattr(response, "model_extra", None) or {}).get(
                    "provider"
                )
                _perf_legs.note_in_set("rr_provider", str(provider))
                return response
            finally:
                _perf_legs.call_span("rr", started)

        try:
            responses = await semaphore_gather(
                *[
                    _probe_call(
                        client.chat.completions.create(
                            model=model,
                            messages=openai_messages,
                            temperature=0,
                            logprobs=True,
                            top_logprobs=2,
                            **options,
                        )
                    )
                    for openai_messages in openai_messages_list
                ],
                max_coroutines=cap,
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


__all__ = ["MIN_COMPLETION_TOKENS", "CompatOpenAIRerankerClient", "LLMConfig"]
