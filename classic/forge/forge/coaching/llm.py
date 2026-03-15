"""Claude (Anthropic) LLM wrapper for the ABN Co-Navigator coaching module."""
from __future__ import annotations

from typing import List


def chat_completion(messages: List[dict], model: str, temperature: float) -> str:
    """
    Send a list of messages to Claude and return the assistant reply.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Claude's API separates the system prompt from the conversation turns,
    so we extract the first system message (if any) and pass the rest as
    the messages list.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # Split system prompt from conversation turns
    system_prompt = ""
    turns = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            turns.append({"role": msg["role"], "content": msg["content"]})

    # Claude requires at least one human turn
    if not turns:
        turns = [{"role": "user", "content": "Hello"}]

    kwargs = dict(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        messages=turns,
    )
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text
