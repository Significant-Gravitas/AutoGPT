"""The brain-dump pipeline's one way of asking a model for a JSON answer."""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.api.features.onboarding_dump.parsing import parse_response_json

# Haiku 5.5 thinks adaptively, and the thinking counts against max_tokens: on
# a real brain dump it spent the whole cap and the JSON came back cut off.
_NO_REASONING = {"reasoning": {"enabled": False}}


class StructuredReply(BaseModel):
    """The parsed answer, or ``None``, and enough to log why it is missing."""

    data: dict | None
    finish_reason: str | None
    chars: int

    def describe(self) -> str:
        """Why a reply failed to parse, without any of its content."""
        return f"finish_reason={self.finish_reason}, {self.chars} chars"


async def request_json(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> StructuredReply:
    """One JSON-only completion, with reasoning off.

    Raises whatever the client or the timeout raises, so each caller keeps
    its own degradation.
    """
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=_NO_REASONING,
        ),
        timeout=timeout,
    )
    if not response.choices:
        return StructuredReply(data=None, finish_reason=None, chars=0)
    choice = response.choices[0]
    content = choice.message.content or ""
    return StructuredReply(
        data=parse_response_json(content),
        finish_reason=str(choice.finish_reason),
        chars=len(content),
    )
