"""What a business owner should never read in an expert's reply (SECRT-3158).

The style judge grades voice; this is the plain-language check beside it: the
tool names, file paths, shell commands and internal ids a reply leaks. It is
deterministic, so every scored response gets it at no extra model cost, and it
reads the reply as written, so a regression shows as a count, not a vibe.
"""

import re
from collections.abc import Iterable

from backend.copilot.tools import TOOL_REGISTRY

# SDK built-ins with compound names; "Read" or "Write" alone are English words.
TOOL_NAMES = frozenset(TOOL_REGISTRY) | {"TodoWrite", "WebFetch", "WebSearch"}

_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_CAPABILITY_ID = re.compile(r"\btool:[a-z_]+")
# Starts a line, or follows a space, quote, bracket or inline-code backtick.
_PATH = re.compile(
    r"(?:(?<=[\s`'\"(])|^)(?:~|\.{1,2})?/[\w.-]+(?:/[\w.-]+)+", re.MULTILINE
)
_FILE_NAME = re.compile(
    r"\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|js|ts|log|xlsx)\b"
)
_COMMAND = re.compile(
    r"\b(?:(?:sed|awk|grep|cat|cd|ls|mkdir|rm|cp|mv|curl|chmod|pip3?|npm|python3?|git)"
    r"\s+(?:-{1,2}[\w-]+|[~./]\S*)"
    r"|git\s+(?:status|add|commit|push|pull|clone|checkout|diff|log|init)\b"
    r"|(?:pip3?|npm|pnpm|yarn)\s+(?:install|run)\b)"
)
_SHELL = (
    "cat|head|tail|less|wc|sort|uniq|cut|tr|find|xargs|touch|rm|cp|mv|ln|tar|"
    "zip|unzip|gzip|wget|curl|ssh|scp|rsync|python3?|node|pip3?|npm|pnpm|yarn|"
    "git|docker|make|chmod|chown|mkdir|sudo|sed|awk|grep|bash|sh|ls|cd|echo|"
    "printf|export|jq"
)
# A command and its argument in command position: opening a line or sentence
# ("cat brief shows…") or right after a run verb ("I'm running cat brief").
# Prepositions such as "with" are left out: "listings with cat food" is prose.
_COMMAND_POSITION = re.compile(
    r"(?:^|(?<=[.!?]\s)|\b(?:[Rr]un(?:s|ning)?|[Rr]an|[Ee]xecut(?:e|es|ed|ing)"
    r"|[Cc]all(?:s|ed|ing)?)\s+)"
    rf"((?:{_SHELL})\s+[\w~/-]+(?:\.[\w~/-]+)*)",
    re.MULTILINE,
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


def technical_terms(text: str, tool_names: Iterable[str]) -> list[str]:
    """Every technical token in ``text``, in the order the checks run, once each."""
    remaining = _LINK.sub(_link_label, text)
    found: list[str] = []
    for pattern in (_CAPABILITY_ID, _tool_pattern(tool_names), _PATH, _FILE_NAME):
        found += pattern.findall(remaining)
        remaining = pattern.sub(" ", remaining)
    found += _COMMAND.findall(remaining) + _COMMAND_POSITION.findall(remaining)
    found += _UUID.findall(remaining)
    return list(dict.fromkeys(found))


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
