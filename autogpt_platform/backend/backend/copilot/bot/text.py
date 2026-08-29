"""Text formatting helpers — message batching and chunk splitting."""

import re
from collections import Counter
from typing import Callable, Iterator

from backend.data.sharing.workspace_refs import cut_lands_inside_artifact_link

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
            cut = _guarded_cut(text, search_start + idx)
            return _balance_code_fences(text[:cut].rstrip(), text[cut:].lstrip("\n"))

    for sep in (". ", "! ", "? "):
        idx = search_region.rfind(sep)
        if idx != -1:
            cut = _guarded_cut(text, search_start + idx + len(sep))
            return _balance_code_fences(text[:cut], text[cut:])

    idx = search_region.rfind(" ")
    if idx != -1:
        cut = _guarded_cut(text, search_start + idx)
        return _balance_code_fences(text[:cut], text[cut:].lstrip())

    cut = _guarded_cut(text, flush_at)
    return _balance_code_fences(text[:cut], text[cut:])


def iter_chunks(text: str, flush_at: int) -> Iterator[str]:
    """Yield ``text`` split into postable chunks, each under ``flush_at``.

    Wraps the ``split_at_boundary`` drain loop so any adapter sending a whole
    long message at once (proactive posts) shares one splitter instead of
    re-implementing the loop.
    """
    remaining = text.strip()
    while remaining:
        chunk, remaining = split_at_boundary(remaining, flush_at)
        if not chunk:
            break
        yield chunk


def resolve_mentions(
    text: str,
    mentionable_users: tuple[tuple[str, str], ...],
    render_token: Callable[[str, str], str],
) -> tuple[str, list[str]]:
    """Substitute ``@DisplayName`` with a platform mention token, but only for
    users on ``mentionable_users``. Returns ``(rendered_text, pinged_user_ids)``.

    Security-sensitive shared policy: only allowlisted names are ever
    converted, so the bot never pings a user the LLM invented (``@everyone``,
    ``@here``, a hallucinated or elsewhere-learned name) — those stay plain
    text. Longest names first so ``@John Smith`` matches before ``@John``, and
    the match is word-bounded so ``@Name`` inside an email/URL is left alone.
    ``render_token(display_name, user_id)`` produces the platform's mention
    markup; the adapter turns ``pinged_user_ids`` into its own ping-safety
    object.
    """
    if not mentionable_users:
        return text, []

    # Names that two different allowlisted users share case-insensitively are
    # ambiguous — we can't know which the author meant, so leave them plain
    # rather than ping whichever happens to sort first.
    name_counts = Counter(name.casefold() for name, _ in mentionable_users)
    users_by_name = {
        name.casefold(): (name, user_id)
        for name, user_id in mentionable_users
        if name_counts[name.casefold()] == 1
    }
    if not users_by_name:
        return text, []

    # ONE combined pattern + ONE sub() pass over the original text. re.sub never
    # re-scans replacement output, so a rendered token can't be matched again —
    # e.g. a display name equal to another user's ID inside an emitted <@U123>.
    # Longest alternative first so "@John Smith" wins over "@John"; the same
    # boundaries as before keep emails/URLs and "@John-Smith" prefixes unmatched.
    alternation = "|".join(
        re.escape(name)
        for name, _ in sorted(users_by_name.values(), key=lambda pair: -len(pair[0]))
    )
    pattern = re.compile(
        rf"(?<![\w@])@({alternation})(?![\w-])",
        re.IGNORECASE,
    )

    pinged: list[str] = []

    def _render(match: re.Match[str]) -> str:
        display_name, user_id = users_by_name[match.group(1).casefold()]
        if user_id not in pinged:
            pinged.append(user_id)
        return render_token(display_name, user_id)

    return pattern.sub(_render, text), pinged


def _guarded_cut(text: str, cut: int) -> int:
    """Keep a workspace artifact markdown link from being split across chunks.

    ``split_at_boundary`` picks a cut purely from prose boundaries, so it can
    land inside ``[name](workspace://...)`` — which would stop the downstream
    artifact extractor matching it. Pull the cut back to the link's start so the
    whole link travels intact into the remainder.
    """
    return cut_lands_inside_artifact_link(text, cut)


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
