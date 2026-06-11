"""OpenAI-compat cross-encoder reranker for warm-context retrieval.

graphiti-core 0.28.2's stock ``OpenAIRerankerClient.rank`` sends
``max_tokens=1`` on each boolean-classifier call. OpenAI (and
OpenRouter's OpenAI/Azure upstreams) now reject completions below 16
output tokens ("Invalid 'max_output_tokens': integer below minimum
value. Expected a value >= 16, but got 1"), which turned every
warm-context fetch into a 400 — and, because the ratification hit
hook runs after the search gather that raised, also starved the
hit-time ratification loop.

``CompatOpenAIRerankerClient.rank`` is a faithful copy of the
upstream method (pinned at graphiti-core 0.28.2) with ``max_tokens``
raised to a compliant floor. Scoring is unchanged: the ``logit_bias``
pins the output to True/False, and only the FIRST token's
top-logprobs are read, so the extra token budget never affects the
ranking (same seam-level patch pattern as
``communities._bounded_label_propagation``).
"""

from __future__ import annotations

import logging

import numpy as np
import openai
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

logger = logging.getLogger(__name__)

# OpenAI's documented minimum for max output tokens on logprob
# completions. The classifier still only ever emits one token (the
# logit_bias restricts the vocabulary to "True"/"False").
MIN_COMPLETION_TOKENS = 16

DEFAULT_MODEL = "gpt-4.1-nano"


class CompatOpenAIRerankerClient(OpenAIRerankerClient):
    """Stock reranker with an OpenAI-compliant ``max_tokens`` floor."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
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
        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=MIN_COMPLETION_TOKENS,
                        logit_bias={"6432": 1, "7983": 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                (
                    response.choices[0].logprobs.content[0].top_logprobs
                    if response.choices[0].logprobs is not None
                    and response.choices[0].logprobs.content is not None
                    else []
                )
                for response in responses
            ]
            scores: list[float] = []
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
                if top_logprobs[0].token.strip().split(" ")[0].lower() == "true":
                    scores.append(norm_logprobs)
                else:
                    scores.append(1 - norm_logprobs)

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


__all__ = ["CompatOpenAIRerankerClient", "LLMConfig", "MIN_COMPLETION_TOKENS"]
