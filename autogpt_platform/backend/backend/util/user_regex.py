"""Compile a regular expression a user wrote, within bounds on what it costs.

The ``timeout=`` that ``regex`` takes covers matching, not compiling, and what
compiling costs grows with a pattern's repeat counts. So the size a pattern
compiles to is counted on ``regex``'s own parse of it, and a pattern over the
limit is refused before it is compiled.
"""

import importlib
from typing import Any

import regex

MAX_PATTERN_LENGTH = 10_000
# About 3 MB and 0.1 s of compiling at most.
MAX_COMPILED_ITEMS = 10_000

# The type stubs for `regex` leave its parser out.
_regex_core: Any = importlib.import_module("regex._regex_core")


def compile_user_regex(pattern: str, flags: int = 0) -> "regex.Pattern[str]":
    """Compile ``pattern``, or raise ``ValueError`` if it is too large to compile.

    An invalid pattern raises ``regex.error``. The compiled pattern is not kept in
    the module's cache, so a large one does not outlive its caller.
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError(
            f"Regex pattern is too large to evaluate safely "
            f"({len(pattern)} characters; maximum {MAX_PATTERN_LENGTH})."
        )
    if _count_compiled_items(_parse(pattern, flags)) > MAX_COMPILED_ITEMS:
        raise ValueError(
            "Regex pattern repeats too much to evaluate safely "
            f"(over {MAX_COMPILED_ITEMS} items once its repeats are expanded)."
        )
    return regex.compile(pattern, flags, cache_pattern=False)


def _parse(pattern: str, flags: int) -> list[Any]:
    """Parse ``pattern`` the way ``regex.compile`` does, up to building its code."""
    while True:
        source = _regex_core.Source(pattern)
        info = _regex_core.Info(flags, source.char_type)
        info.guess_encoding = regex.UNICODE
        source.ignore_space = bool(info.flags & regex.VERBOSE)
        try:
            parsed = _regex_core._parse_pattern(source, info)
            break
        except _regex_core._UnscopedFlagSet:
            # A global inline flag such as (?x) applies from the pattern's start.
            flags = info.global_flags
    if not info.flags & _regex_core._ALL_ENCODINGS:
        info.flags |= regex.UNICODE
    reverse = bool(info.flags & regex.REVERSE)
    parsed.fix_groups(pattern, reverse, False)
    parsed = parsed.optimise(info, reverse)
    # Groups called from a lookbehind or a fuzzy section are compiled once more.
    _regex_core._check_group_features(info, parsed)
    return [parsed, info.additional_groups]


def _count_compiled_items(item: Any) -> int:
    if isinstance(item, dict):
        # `_key` holds a second reference to the members of a character set.
        return sum(
            _count_compiled_items(value)
            for name, value in item.items()
            if name != "_key"
        )
    if isinstance(item, (list, tuple)):
        return sum(_count_compiled_items(child) for child in item)
    if not isinstance(item, _regex_core.RegexBase):
        return 0
    body = _count_compiled_items(vars(item))
    if isinstance(item, _regex_core.GreedyRepeat):
        # Compiled as `min_count` copies of its body, plus one more to loop over.
        body *= item.min_count + 1
    return 1 + body
