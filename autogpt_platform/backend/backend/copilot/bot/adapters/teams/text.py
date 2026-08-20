"""Convert the bot's canonical CommonMark output into what Teams renders.

Teams supports a noticeably smaller markdown subset than Discord or Telegram:
headings, tables and horizontal rules are simply **not rendered at all** and
leak through as literal ``#``/``|`` characters. LLM output uses all three
constantly, so they are downgraded here rather than shipped raw.

Bold, italic, inline code, fenced code and links render fine and pass through
untouched.
"""

import re
from typing import Any

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)
_RULE_RE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")

# Placeholder for stashed code so downgrades never rewrite code content.
_STASH = "\x00{}\x00"


def to_teams_markdown(text: str) -> str:
    """Downgrade canonical CommonMark to Teams' rendered subset."""
    # NUL is the stash delimiter below, so input carrying one could otherwise
    # steer a restore into the wrong position.
    text = text.replace("\x00", "")
    stashed: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stashed.append(match.group(0))
        return _STASH.format(len(stashed) - 1)

    text = _FENCE_RE.sub(_stash, text)

    # Headings render as literal '#' in Teams — bold is the closest thing that
    # survives on every client.
    text = _HEADING_RE.sub(lambda m: f"**{m.group(1).strip()}**", text)
    # Horizontal rules render as literal dashes.
    text = _RULE_RE.sub("", text)
    # Inline images are not rendered in message text; keep them reachable.
    text = _IMAGE_RE.sub(lambda m: f"[{m.group(1) or 'image'}]({m.group(2)})", text)
    text = _fence_tables(text)

    for index, original in enumerate(stashed):
        text = text.replace(_STASH.format(index), original)
    return text


def _fence_tables(text: str) -> str:
    """Wrap markdown tables in a code fence so they keep their alignment.

    Teams renders no table markup at all, so an unfenced table collapses into
    an unreadable run of pipes. Monospace preserves the columns.
    """
    lines = text.split("\n")
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if not block:
            return
        # A single piped line is more likely prose than a table.
        if len(block) > 1:
            out.extend(["```", *block, "```"])
        else:
            out.extend(block)
        block.clear()

    for line in lines:
        if _TABLE_ROW_RE.match(line):
            block.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


def mention_token(display_name: str, _user_id: str) -> str:
    """Teams mention markup, for ``resolve_mentions``' ``render_token``.

    The tag alone does nothing — a mention only pings when a matching entity
    accompanies the activity (see :func:`mention_entities`). That is why LLM
    output containing a literal ``<at>…</at>`` cannot forge a ping.
    """
    return f"<at>{display_name}</at>"


def mention_entities(
    pinged_user_ids: list[str],
    mentionable_users: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Build the ``entities`` array that makes rendered mentions actually ping.

    Only ids that ``resolve_mentions`` allowlisted are included, so the entity
    array inherits the shared mention-safety policy rather than restating it.
    """
    names_by_id = {user_id: name for name, user_id in mentionable_users}
    entities: list[dict[str, Any]] = []
    for user_id in pinged_user_ids:
        name = names_by_id.get(user_id)
        if not name:
            continue
        entities.append(
            {
                "type": "mention",
                "text": mention_token(name, user_id),
                "mentioned": {"id": user_id, "name": name},
            }
        )
    return entities
