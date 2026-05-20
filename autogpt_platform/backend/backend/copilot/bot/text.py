"""Text formatting helpers — message batching and chunk splitting."""

import re

# Matches a triple-backtick fence with an optional language tag. Used to tell
# whether a cut falls inside an open Markdown code block.
_CODE_FENCE = re.compile(r"```(\w*)")


def format_batch(batch: list[tuple[str, str, str]], platform: str) -> str:
    """Format one or more pending messages into a single prompt for AutoPilot.

    Each batch entry is (username, user_id, text). When multiple messages are
    batched together (because they arrived while the bot was streaming a prior
    response), they're labelled individually so the LLM can address each.
    """
    platform_display = platform.capitalize()
    if len(batch) == 1:
        username, user_id, text = batch[0]
        return (
            f"[Message sent by {username} ({platform_display} user ID: {user_id})]\n"
            f"{text}"
        )

    lines = ["[Multiple messages — please address them together]"]
    for username, user_id, text in batch:
        lines.append(
            f"\n[From {username} ({platform_display} user ID: {user_id})]\n{text}"
        )
    return "\n".join(lines)


def split_at_boundary(text: str, flush_at: int) -> tuple[str, str]:
    """Split text at a natural boundary to fit within a length limit.

    Returns (postable_chunk, remaining_text).
    Prefers: paragraph > newline > sentence end > space > hard cut.
    If the cut lands inside a Markdown code fence (``\\`\\`\\``), the fence is
    closed in the chunk and reopened at the start of the remainder so both
    sides render correctly.
    """
    if len(text) <= flush_at:
        return text, ""

    search_start = max(0, flush_at - 200)
    search_region = text[search_start:flush_at]

    for sep in ("\n\n", "\n"):
        idx = search_region.rfind(sep)
        if idx != -1:
            cut = search_start + idx
            return _balance_code_fences(text[:cut].rstrip(), text[cut:].lstrip("\n"))

    for sep in (". ", "! ", "? "):
        idx = search_region.rfind(sep)
        if idx != -1:
            cut = search_start + idx + len(sep)
            return _balance_code_fences(text[:cut], text[cut:])

    idx = search_region.rfind(" ")
    if idx != -1:
        cut = search_start + idx
        return _balance_code_fences(text[:cut], text[cut:].lstrip())

    return _balance_code_fences(text[:flush_at], text[flush_at:])


def _balance_code_fences(before: str, after: str) -> tuple[str, str]:
    """If ``before`` ends inside an open ``\\`\\`\\`` fence, close and reopen it.

    Preserves the language tag from the opening fence so syntax highlighting
    survives the split.
    """
    fences = _CODE_FENCE.findall(before)
    if len(fences) % 2 == 0:
        return before, after
    lang = fences[-1]
    closed_before = f"{before.rstrip()}\n```"
    reopened_after = f"```{lang}\n{after.lstrip()}"
    return closed_before, reopened_after
