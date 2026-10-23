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

# A file the reply shares as a link renders as a card, not as a path.
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_CAPABILITY_ID = re.compile(r"\btool:[a-z_]+")
_PATH = re.compile(r"(?:(?<=\s)|^)(?:~|\.{1,2})?/[\w.-]+(?:/[\w.-]+)+", re.MULTILINE)
_FILE_NAME = re.compile(
    r"\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|js|ts|log|xlsx)\b"
)
_COMMAND = re.compile(
    r"\b(?:sed|awk|grep|cat|cd|ls|mkdir|rm|cp|mv|curl|chmod|pip3?|npm|python3?|git)"
    r"\s+(?:-{1,2}[\w-]+|[~./]\S*)"
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


def technical_terms(text: str, tool_names: Iterable[str]) -> list[str]:
    """Every technical token in ``text``, in the order the checks run, once each."""
    remaining = _LINK.sub(" ", text)
    found: list[str] = []
    for pattern in (_CAPABILITY_ID, _tool_pattern(tool_names), _PATH, _FILE_NAME):
        found += pattern.findall(remaining)
        remaining = pattern.sub(" ", remaining)
    found += _COMMAND.findall(remaining) + _UUID.findall(remaining)
    return list(dict.fromkeys(found))


def _tool_pattern(tool_names: Iterable[str]) -> re.Pattern[str]:
    compound = sorted(
        (n for n in tool_names if "_" in n or re.search(r"[a-z][A-Z]", n)),
        key=len,
        reverse=True,
    )
    if not compound:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(map(re.escape, compound)) + r")\b")
