"""What a business owner should never read in an expert's reply (SECRT-3158).

The style judge grades voice; this is the plain-language check beside it: the
tool names, file paths, shell commands and internal ids a reply leaks. It is
deterministic, so every scored response gets it at no extra model cost, and it
reads the reply as written, so a regression shows as a count, not a vibe.
"""

import re
from collections.abc import Iterable

from backend.copilot.tools import TOOL_REGISTRY

from .models import ExpertSummary

# SDK built-ins with compound names; the single-word ones are below.
TOOL_NAMES = frozenset(TOOL_REGISTRY) | {
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "MultiEdit",
    "NotebookEdit",
}

_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_CAPABILITY_ID = re.compile(r"\btool:[a-z_]+")
# SDK built-ins named by one word. "Glob" and "Grep" are never English as
# written; "Read", "Edit" and the rest are, so they count only when shown as a
# tool: as inline code, or before "tool" ("the Read tool").
_SDK_TOOL = re.compile(
    r"\b(Glob|Grep)\b|`(Read|Write|Edit|Bash|Task)`"
    r"|\b(Read|Write|Edit|Bash|Task)(?=\s+tool\b)"
)
# A path segment never ends in a dot, so a sentence's full stop stays out.
_SEGMENT = r"\.?[\w-]+(?:\.[\w-]+)*"
# Starts a line, or follows a space, quote, bracket or inline-code backtick:
# two or more segments, one after ~ or ./, or a system root alone ("/tmp").
# "and/or", "24/7" and "Rs 899 / 30ml" match none of these.
_PATH = re.compile(
    rf"(?:(?<=[\s`'\"(])|^)(?:(?:~|\.{{1,2}})?/{_SEGMENT}(?:/{_SEGMENT})+"
    rf"|(?:~|\.{{1,2}})/{_SEGMENT}"
    r"|/(?:tmp|etc|home|usr|var|opt|root|bin|dev|mnt|srv|workspace)\b)",
    re.MULTILINE,
)
# A script counts by its file name ("index.js"); "Next.js" or "Node.js" is a
# product name, not a file. A document counts by its extension ("report.pdf");
# "the PDF report" names a kind of file, not a file.
_FILE_NAME = re.compile(
    r"\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|log|sql|xml|ipynb"
    r"|xlsx?|docx?|pptx?|pdf|png|jpe?g|gif|svg|webp|zip)\b"
    r"|\b[a-z0-9_][\w-]*\.(?:js|ts)\b"
)
_COMMAND = re.compile(
    r"\b(?:(?:sed|awk|grep|cat|cd|ls|mkdir|rm|cp|mv|curl|chmod|pip3?|npm|python3?|git)"
    r"\s+(?:-{1,2}[\w-]+|[~./]\S*)"
    r"|git\s+(?:status|add|commit|push|pull|clone|checkout|diff|log|init)\b"
    r"|(?:pip3?|npm|pnpm|yarn)\s+(?:install|run)\b)"
)
# Commands that are also everyday words ("sort", "export", "find") count only
# before an argument, never before a function word: "Running sort on items" is
# prose, "Running sort orders" is a command.
_WORDLIKE = "cat|head|tail|less|sort|cut|tr|find|touch|ln|zip|node|make|echo|export"
_CODE_ONLY = (
    "wc|uniq|xargs|rm|cp|mv|tar|unzip|gzip|wget|curl|ssh|scp|rsync|python3?|"
    "pip3?|npm|pnpm|yarn|git|docker|chmod|chown|mkdir|sudo|sed|awk|grep|bash|sh|"
    "ls|cd|printf|jq"
)
_FUNCTION_WORDS = (
    "on|of|for|to|the|a|an|and|or|in|at|by|from|with|into|about|over|through|"
    "this|that|these|those|my|your|our|their|its|all|any|some"
)
_ARG = r"[\w~/-]+(?:\.[\w~/-]+)*"
# Command position: opening a line or sentence, or right after a run verb.
# Prepositions such as "with" are left out: "listings with cat food" is prose.
_POSITION = (
    r"(?:^|(?<=[.!?]\s)|\b(?:[Rr]un(?:s|ning)?|[Rr]an|[Ee]xecut(?:e|es|ed|ing)"
    r"|[Cc]all(?:s|ed|ing)?)\s+)"
)
# Certain, and so gating: a code-only command with its argument, bare or
# quoted ("I'm running grep notes"), or any command quoted as inline code
# ("I ran `cat brief`").
_COMMAND_POSITION = re.compile(
    _POSITION + rf"(?:[`'\"]((?:{_WORDLIKE}|{_CODE_ONLY})\s+{_ARG})"
    rf"|((?:{_CODE_ONLY})\s+{_ARG}))",
    re.MULTILINE,
)
# Possible, and so only reported: a bare everyday-word command before an
# argument. "Running cat brief" and "ran head office numbers" read the same,
# so a gate on this would fail runs on prose.
_POSSIBLE_COMMAND = re.compile(
    _POSITION + rf"((?:{_WORDLIKE})\s+(?!(?:{_FUNCTION_WORDS})\b){_ARG})",
    re.MULTILINE,
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


def leaking_experts(summaries: Iterable[ExpertSummary]) -> list[str]:
    """The experts any of whose replies showed the user a technical term. The
    eval fails on any: a leak is a defect, not a style score to average away."""
    return [s.expert for s in summaries if s.technical_replies]


def technical_terms(text: str, tool_names: Iterable[str]) -> list[str]:
    """Every technical token in ``text``, in the order the checks run, once each."""
    remaining = _LINK.sub(_link_label, text)
    found: list[str] = []
    found += [m[1] or m[2] or m[3] for m in _SDK_TOOL.finditer(remaining)]
    remaining = _SDK_TOOL.sub(" ", remaining)
    for pattern in (_CAPABILITY_ID, _tool_pattern(tool_names), _PATH, _FILE_NAME):
        found += pattern.findall(remaining)
        remaining = pattern.sub(" ", remaining)
    found += _COMMAND.findall(remaining)
    found += [m[1] or m[2] for m in _COMMAND_POSITION.finditer(remaining)]
    found += _UUID.findall(remaining)
    return list(dict.fromkeys(found))


def possible_technical_terms(text: str) -> list[str]:
    """Commands that are also everyday words, bare in command position: worth
    reading, not proof of a leak, so the eval reports them without gating."""
    return list(dict.fromkeys(_POSSIBLE_COMMAND.findall(_LINK.sub(_link_label, text))))


def _link_label(link: re.Match[str]) -> str:
    """A shared workspace file renders as a card, so it shows nothing; any
    other link shows its label, which is read like the rest of the reply."""
    return " " if link[2].startswith("workspace://") else f" {link[1]} "


def _tool_pattern(tool_names: Iterable[str]) -> re.Pattern[str]:
    compound = sorted(
        (n for n in tool_names if "_" in n or re.search(r"[a-z][A-Z]", n)),
        key=len,
        reverse=True,
    )
    if not compound:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(map(re.escape, compound)) + r")\b")
