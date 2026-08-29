"""Graphiti LLM client that runs on OpenAI's flex tier.

Subclasses graphiti-core's ``OpenAIClient`` to inject
``service_tier="flex"`` on every underlying LLM call. Used by the
nightly community-rebuild path: ~50% discount, best-effort latency
(worst-case ~15min queue) — acceptable for scheduled non-interactive
work, not for live chat ingest.

``service_tier`` rides through ``extra_body`` so the same client works
against both backends:

- **OpenAI native** picks the param off the request body and routes
  the call onto the flex queue.
- **OpenRouter** forwards ``extra_body`` keys to the upstream model,
  reaching the same flex tier for OpenAI/Google-backed models and
  silently ignoring it for providers without a flex equivalent.

We intentionally do NOT pass ``service_tier`` as a top-level SDK
kwarg — the openai-python SDK doesn't expose it as a typed kwarg on
every release, so ``extra_body`` is the forward-compatible path
(mirrors what ``backend.util.llm.providers._call_openai_responses``
does).
"""

from typing import Any

from graphiti_core.llm_client import OpenAIClient
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class FlexOpenAIClient(OpenAIClient):
    """Same as graphiti's ``OpenAIClient`` with ``service_tier`` injected.

    Constructor mirrors ``OpenAIClient`` exactly so it slots into
    graphiti's ``Graphiti(llm_client=...)`` without surprises.
    """

    SERVICE_TIER = "flex"

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        is_reasoning_model = (
            model.startswith("gpt-5")
            or model.startswith("o1")
            or model.startswith("o3")
        )

        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "text_format": response_model,
            "extra_body": {"service_tier": self.SERVICE_TIER},
        }

        temperature_value = temperature if not is_reasoning_model else None
        if temperature_value is not None:
            request_kwargs["temperature"] = temperature_value

        if is_reasoning_model and reasoning is not None:
            request_kwargs["reasoning"] = {"effort": reasoning}

        if is_reasoning_model and verbosity is not None:
            request_kwargs["text"] = {"verbosity": verbosity}

        return await self.client.responses.parse(**request_kwargs)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        is_reasoning_model = (
            model.startswith("gpt-5")
            or model.startswith("o1")
            or model.startswith("o3")
        )

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"service_tier": self.SERVICE_TIER},
        )
