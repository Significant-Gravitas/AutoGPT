from typing import Any

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .extraction_prompts import prepare_extraction_messages
from .models import is_luna_model


class GraphitiOpenAIClient(OpenAIClient):
    """Preserve non-reasoning extraction on Luna with graphiti-core 0.30.2."""

    SERVICE_TIER: str | None = None

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        """Apply Luna's extraction policy while bypassing attribute extraction."""
        if (
            is_luna_model(self._get_model_for_size(model_size))
            and not attribute_extraction
        ):
            messages = prepare_extraction_messages(
                messages, response_model, prompt_name
            )
        return await super().generate_response(
            messages,
            response_model,
            max_tokens,
            model_size,
            group_id,
            prompt_name,
            attribute_extraction=attribute_extraction,
        )

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
        if not is_luna_model(model):
            return await super()._create_structured_completion(
                model,
                messages,
                temperature,
                max_tokens,
                response_model,
                reasoning,
                verbosity,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "text_format": response_model,
            "reasoning": {"effort": "none"},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.SERVICE_TIER:
            kwargs["extra_body"] = {"service_tier": self.SERVICE_TIER}
        return await self.client.responses.parse(**kwargs)

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
        if not is_luna_model(model):
            return await super()._create_completion(
                model,
                messages,
                temperature,
                max_tokens,
                response_model,
                reasoning,
                verbosity,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.SERVICE_TIER:
            kwargs["extra_body"] = {"service_tier": self.SERVICE_TIER}
        return await self.client.chat.completions.create(**kwargs)
