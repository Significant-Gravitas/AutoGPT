from math import ceil, floor
from typing import Any

from forge.llm.prompting.schema import ChatPrompt

SEPARATOR_LENGTH = 42


def dump_prompt(prompt: ChatPrompt) -> str:
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


def format_numbered_list(items: list[Any], start_at: int = 1) -> str:
    return "\n".join(f"{i}. {str(item)}" for i, item in enumerate(items, start_at))


def indent(content: str, indentation: int | str = 4) -> str:
    if type(indentation) is int:
        indentation = " " * indentation
    return indentation + content.replace("\n", f"\n{indentation}")  # type: ignore


def to_numbered_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    if items:
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response
