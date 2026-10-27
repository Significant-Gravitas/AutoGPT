"""What a business owner should never read in an expert's reply (SECRT-3158).

The style judge grades voice; this is the plain-language check beside it: the
tool names, file paths, shell commands and internal ids a reply leaks. It is
deterministic, so every scored response gets it at no extra model cost, and it
reads the reply as written, so a regression shows as a count, not a vibe.
"""

import json
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from backend.copilot.tools import TOOL_REGISTRY

from .models import ExpertSummary

# The chat's own rule set (plainText.ts), so the eval fails on what the chat
# would hide and the two cannot drift. The eval runs from a checkout; the
# file is not in the backend image.
RULES_PATH = (
    Path(__file__).resolve().parents[5]
    / "frontend/src/app/(platform)/copilot/components/ToolChain/technicalTerms.json"
)

_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_PLACEHOLDER = re.compile(r"\{([A-Z_]+)(?::(exact|noabbr))?\}")
_ABBREVIATION_LETTERS = 3


@cache
def _rules() -> dict:
    return json.loads(RULES_PATH.read_text())


@cache
def _patterns(tier: str) -> tuple[re.Pattern[str], ...]:
    rules = _rules()
    return tuple(
        re.compile(_render(rule["pattern"], rules), re.MULTILINE | re.ASCII)
        for rule in rules[tier]
    )


TOOL_NAMES = frozenset(TOOL_REGISTRY) | frozenset(_rules()["words"]["SDK_TOOL_NAMES"])


def leaking_experts(summaries: Iterable[ExpertSummary]) -> list[str]:
    """The experts any of whose replies showed the user a technical term. The
    eval fails on any: a leak is a defect, not a style score to average away."""
    return [s.expert for s in summaries if s.technical_replies]


def technical_terms(text: str, tool_names: Iterable[str]) -> list[str]:
    """Every technical term in ``text`` the shared rules are certain of, plus
    any registry tool name, in the order the checks run, once each."""
    remaining = _LINK.sub(_link_label, text)
    registry = _tool_pattern(tool_names)
    found = registry.findall(remaining)
    remaining = registry.sub(" ", remaining)
    return list(dict.fromkeys(found + _find(_patterns("certain"), remaining)))


def possible_technical_terms(text: str) -> list[str]:
    """Terms the shared rules call possible: commands that are also everyday
    words, inline code, slashed tokens. Worth reading, not proof of a leak,
    so the eval reports them without gating."""
    return list(
        dict.fromkeys(_find(_patterns("possible"), _LINK.sub(_link_label, text)))
    )


def _find(patterns: Iterable[re.Pattern[str]], text: str) -> list[str]:
    """Each pattern's terms in turn (its first capture group that matched,
    else the match), taking each match out so later patterns don't see it."""
    found: list[str] = []
    for pattern in patterns:
        found += [
            next((g for g in m.groups() if g), m[0]) for m in pattern.finditer(text)
        ]
        text = pattern.sub(" ", text)
    return found


def _render(pattern: str, rules: dict) -> str:
    for name, fragment in rules["fragments"].items():
        pattern = pattern.replace("{" + name + "}", fragment)
    return _PLACEHOLDER.sub(
        lambda m: _alternation(rules["words"][m[1]], m[2] or "any"), pattern
    )


def _alternation(words: list[str], mode: str) -> str:
    return "|".join(_word(w, mode) for w in sorted(words, key=len, reverse=True))


def _word(word: str, mode: str) -> str:
    """``word`` as a pattern: as written, or in any case. A short word in
    capitals ("CD", "RM") reads as an abbreviation, so ``noabbr`` matches it
    only in lower or title case."""
    if mode == "exact":
        return re.escape(word)
    if mode == "noabbr" and sum(c.isalpha() for c in word) <= _ABBREVIATION_LETTERS:
        lower = word.lower()
        return f"(?:{re.escape(lower)}|{re.escape(lower[:1].upper() + lower[1:])})"
    return "".join(
        f"[{c.lower()}{c.upper()}]" if c.isalpha() else re.escape(c) for c in word
    )


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
