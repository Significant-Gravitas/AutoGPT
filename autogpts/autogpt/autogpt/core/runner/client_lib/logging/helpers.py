from math import ceil, floor

from autogpt.core.planning import LanguageModelPrompt

SEPARATOR_LENGTH = 42


def dump_prompt(prompt: LanguageModelPrompt) -> str:
    def separator(text: str):
        half_sep_len = (SEPARATOR_LENGTH - 2 - len(text)) / 2
        return f"{floor(half_sep_len)*'-'} {text.upper()} {ceil(half_sep_len)*'-'}"

    formatted_messages = "\n".join(
        [f"{separator(m.role)}\n{m.content}" for m in prompt.messages]
    )
    return f"""
============== {prompt.__class__.__name__} ==============
Length: {len(prompt.messages)} messages
{formatted_messages}
==========================================
"""
