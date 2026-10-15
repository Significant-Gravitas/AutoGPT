"""Text formatting helpers — message batching and chunk splitting."""

import re
from collections import defaultdict
from typing import Callable, Iterator

from backend.data.sharing.workspace_refs import cut_lands_inside_artifact_link

# Matches a triple-backtick fence with an optional language tag. Used to tell
# whether a cut falls inside an open Markdown code block.
_CODE_FENCE = re.compile(r"```(\w*)")


def format_batch(batch: list[tuple[str, str, str]], platform: str) -> str:
    """Format one or more pending messages into a single prompt for Otto.

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
    """Turn the mentions the model wrote into platform mention tokens, for
    allowlisted people only. Returns ``(rendered_text, pinged_ids)``.

    The model mentions someone in one of two ways, and both resolve here:

    - by id, as ``<@ID>`` (or ``<@!ID>``). Every turn tells the model each
      sender's user id, so this is the exact form and the one it reaches for.
    - by name, as ``@Name``. Longest name first, so ``@John Smith`` wins over
      ``@John``, and word-bounded, so an email or URL is left alone.

    Security-sensitive shared policy: only ids and names on
    ``mentionable_users`` ever become a ping. An id or name the model invented,
    ``@everyone`` and ``@here`` stay exactly as written. A name that belongs to
    two *different* ids is ambiguous and pings neither; the same person listed
    under a name twice is not a clash. ``render_token(name, id)`` produces the
    platform's markup, and the adapter turns ``pinged_ids`` into its own
    ping-safety object.
    """
    if not mentionable_users:
        return text, []

    ids_by_name: dict[str, set[str]] = defaultdict(set)
    for name, user_id in mentionable_users:
        ids_by_name[name.casefold()].add(user_id)
    users_by_name = {
        name.casefold(): (name, user_id)
        for name, user_id in mentionable_users
        if len(ids_by_name[name.casefold()]) == 1
    }
    names_by_id: dict[str, str] = {}
    for name, user_id in mentionable_users:
        names_by_id.setdefault(user_id, name)

    # ONE combined pattern + ONE sub() pass over the original text. re.sub never
    # re-scans replacement output, so a rendered token can't be matched again,
    # e.g. a display name equal to another user's id inside an emitted <@U123>.
    alternatives: list[str] = [
        r"<@!?(?P<id>"
        + "|".join(
            re.escape(user_id) for user_id in sorted(names_by_id, key=len, reverse=True)
        )
        + r")>"
    ]
    if users_by_name:
        alternatives.append(
            r"(?<![\w@])@(?P<name>"
            + "|".join(
                re.escape(name)
                for name, _ in sorted(
                    users_by_name.values(), key=lambda pair: -len(pair[0])
                )
            )
            + r")(?![\w-])"
        )
    pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    pinged: list[str] = []

    def _render(match: re.Match[str]) -> str:
        if match.group("id") is not None:
            user_id = _canonical_id(match.group("id"), names_by_id)
            display_name = names_by_id[user_id]
        else:
            display_name, user_id = users_by_name[match.group("name").casefold()]
        if user_id not in pinged:
            pinged.append(user_id)
        return render_token(display_name, user_id)

    return pattern.sub(_render, text), pinged


def _canonical_id(matched: str, names_by_id: dict[str, str]) -> str:
    """The allowlisted id a case-insensitive match stands for. Ids are matched
    ignoring case only because names share the pattern; the id itself is
    returned exactly as it was allowlisted."""
    if matched in names_by_id:
        return matched
    return next(
        user_id for user_id in names_by_id if user_id.casefold() == matched.casefold()
    )


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
