from typing import Any


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
