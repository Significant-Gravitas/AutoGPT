"""Text rules for a task's record: how long each line may be, how a long
reply becomes a short result, and how the source a worker names as proof is
matched against what is on record. Pure functions, no database."""

import os
import re
from urllib.parse import urlsplit

# People skim a task: every line on it is short.
GOAL_MAX_CHARS = 120
DONE_WHEN_MAX_CHARS = 160
PROGRESS_MAX_CHARS = 80
CLAIM_MAX_CHARS = 120
SUMMARY_MAX_CHARS = 400

_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"[*`#>|~]+")
_SENTENCE_END = re.compile(r"[.!?](?=\s)")
_TOOL_PREFIXES = ("mcp__copilot__", "functions.", "tool:")
_NOT_A_TOOL = frozenset({"and", "then", "the", "tool", "tools"})
_CAPABILITY_KINDS = frozenset({"mcp", "service", "block", "skill"})


def clip(text: str | None, limit: int) -> str | None:
    """One plain line of at most ``limit`` characters, cut at a word."""
    line = " ".join((text or "").split())
    if len(line) <= limit:
        return line or None
    cut = line[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{cut or line[: limit - 1]}…"


def too_long(text: str, limit: int, what: str) -> str | None:
    """What to tell a worker whose line is over its cap, or None when it fits."""
    length = len(" ".join(text.split()))
    if length <= limit:
        return None
    return (
        f"That {what} is {length} characters; the user skims it, so keep it to "
        f"{limit}. Say only the main point and call again."
    )


def short_result(text: str | None, limit: int = SUMMARY_MAX_CHARS) -> str | None:
    """A long reply as a short result: its opening as plain text, cut at the
    end of a sentence. Used when a worker finished without a summary."""
    plain = _LINK.sub(r"\1", text or "")
    lines = [" ".join(_MARKUP.sub(" ", line).split()) for line in plain.splitlines()]
    body = " ".join(line for line in lines if line and not set(line) <= set("-: "))
    if len(body) <= limit:
        return body or None
    ends = [m.end() for m in _SENTENCE_END.finditer(body[:limit])]
    if ends and ends[-1] >= limit // 3:
        return body[: ends[-1]]
    return clip(body, limit)


def tool_names(text: str | None) -> set[str]:
    """The tools a text names, however it writes them: 'write_file + bash_exec',
    'mcp__copilot__bash_exec' and 'tool:bash_exec' all name bash_exec, and
    'mcp:gmail/send_email' also answers to 'gmail' and 'send_email'."""
    names: set[str] = set()
    for part in re.split(r"[\s,+&]+", (text or "").lower()):
        name = part.strip("`'\"()[].")
        for prefix in _TOOL_PREFIXES:
            name = name.removeprefix(prefix)
        if not name or name in _NOT_A_TOOL:
            continue
        names.add(name)
        names.update(re.split(r"[/:]", name))
    return names - _NOT_A_TOOL - _CAPABILITY_KINDS - {""}


def recorded_tool_names(tool: str | None) -> set[str]:
    """The names a call on record answers to. One reached by id (a capability)
    also answers to ``run_capability``, the tool the worker really called."""
    names = tool_names(tool)
    return names | {"run_capability"} if ":" in (tool or "") else names


def file_key(name: str | None) -> str:
    """A file as named in a path, a link or on its own: its bare name."""
    text = (name or "").strip().strip("`'\"")
    text = text.removeprefix("workspace://").split("#", 1)[0]
    return os.path.basename(text.rstrip("/")).casefold()


def url_key(url: str | None) -> str:
    """A page, whatever the scheme, ``www.`` or trailing slash it is given with."""
    parts = urlsplit((url or "").strip())
    if not parts.netloc:
        return ""
    host = parts.netloc.lower().removeprefix("www.")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{parts.path.rstrip('/')}{query}"
