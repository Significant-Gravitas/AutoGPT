from math import ceil, floor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.llm.prompting import ChatPrompt
    from forge.llm.providers.schema import ChatMessage


SEPARATOR_LENGTH = 42


def dump_prompt(prompt: "ChatPrompt | list[ChatMessage]") -> str:
    def separator(text: str):
        half_sep_len = (SEPARATOR_LENGTH - 2 - len(text)) / 2
        return f"{floor(half_sep_len)*'-'} {text.upper()} {ceil(half_sep_len)*'-'}"

    if not isinstance(prompt, list):
        prompt = prompt.messages

    formatted_messages = "\n".join(
        [f"{separator(m.role)}\n{m.content}" for m in prompt]
    )
    return f"""
============== {prompt.__class__.__name__} ==============
Length: {len(prompt)} messages
{formatted_messages}
==========================================
"""
