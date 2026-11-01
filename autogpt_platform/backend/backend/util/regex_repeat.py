"""Read the repetition counts ({m}, {m,}, {m,n}) a regex pattern asks for."""

import re

_COUNTED_REPEAT = re.compile(r"\{(\d*)(?:,(\d*))?\}")
# \p{...}, \P{...} and \N{...} name a character; their braces are not a repeat
_NAMED_ESCAPES = frozenset("pPN")


def largest_repeat_count(pattern: str, limit: int) -> int:
    """Return the largest repetition count in ``pattern``, with the counts of
    enclosing groups multiplied in, so ``(a{10}){20}`` is 200.

    Any count above ``limit`` is returned as ``limit + 1``. Escaped braces and
    braces inside a character class are literals, not repeats.
    """
    over = limit + 1
    # The largest count seen so far inside each open group
    largest = [1]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            i = _skip_escape(pattern, i)
        elif char == "[":
            i = _skip_class(pattern, i)
        elif char == "(":
            largest.append(1)
            i += 1
        elif char == ")" and len(largest) > 1:
            inner = largest.pop()
            count, i = _read_count(pattern, i + 1, over)
            largest[-1] = max(largest[-1], min(inner * max(count, 1), over))
        elif char == "{":
            count, end = _read_count(pattern, i, over)
            largest[-1] = max(largest[-1], count)
            i = end if end > i else i + 1
        else:
            i += 1
    return max(largest)


def _read_count(pattern: str, start: int, over: int) -> tuple[int, int]:
    """Read a counted repeat at ``start``; return its largest count (capped at
    ``over``) and the index after it, or ``(0, start)`` when there is none."""
    match = _COUNTED_REPEAT.match(pattern, start)
    if not match:
        return 0, start
    counts = [_to_count(digits, over) for digits in match.groups() if digits]
    return max(counts, default=0), match.end()


def _to_count(digits: str, over: int) -> int:
    digits = digits.lstrip("0") or "0"
    # Long digit strings are over the limit; skip converting them
    if len(digits) > len(str(over)):
        return over
    return min(int(digits), over)


def _skip_escape(pattern: str, start: int) -> int:
    end = start + 2
    if pattern[start + 1 : end] in _NAMED_ESCAPES and pattern[end : end + 1] == "{":
        close = pattern.find("}", end)
        return close + 1 if close != -1 else len(pattern)
    return end


def _skip_class(pattern: str, start: int) -> int:
    i = start + 1
    if pattern[i : i + 1] == "^":
        i += 1
    # A "]" straight after the opening bracket is a literal
    if pattern[i : i + 1] == "]":
        i += 1
    while i < len(pattern):
        if pattern[i] == "\\":
            i += 2
        elif pattern[i] == "]":
            return i + 1
        else:
            i += 1
    return len(pattern)
