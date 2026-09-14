import sys
from typing import Any

# ---------------------------------------------------------------------------
#  String helpers
# ---------------------------------------------------------------------------


def _truncate_string_middle(value: str, limit: int) -> str:
    """Shorten *value* to at most *limit* chars by removing the **middle**.

    The omission marker is budgeted first; any remaining characters are split
    between a head and a tail (head-only when the budget is tiny).  The result
    never exceeds ``max(0, limit)``.  When the marker itself cannot fit, the
    value is hard-prefixed to *limit* (or empty for ``limit <= 0``).
    """

    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value

    n = len(value)

    def marker(omitted: int) -> str:
        return f"… (omitted {omitted} chars)…"

    def fits(content_budget: int) -> bool:
        omitted = n - content_budget
        return omitted >= 1 and content_budget + len(marker(omitted)) <= limit

    # Fitting content budgets form ``[0, T]`` or are empty (marker too large).
    if not fits(0):
        return value[:limit]

    lo, hi = 0, min(limit, n - 1)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    omitted = n - best
    m = marker(omitted)

    # Head-only for tiny budgets; otherwise split head/tail.
    # Never slice with tail_len == 0 via ``value[-0:]`` (that yields the full string).
    if best <= 1:
        head_len, tail_len = best, 0
    else:
        head_len = best // 2
        tail_len = best - head_len

    if tail_len == 0:
        return f"{value[:head_len]}{m}"
    return f"{value[:head_len]}{m}{value[-tail_len:]}"


# ---------------------------------------------------------------------------
#  List helpers
# ---------------------------------------------------------------------------


def _truncate_list_middle(lst: list[Any], str_lim: int, list_lim: int) -> list[Any]:
    """Return *lst* truncated to *list_lim* items, removing from the middle.

    Each retained element is itself recursively truncated via
    :func:`_truncate_value` so we don’t blow the budget with long strings nested
    inside.
    """

    if len(lst) <= list_lim:
        return [_truncate_value(v, str_lim, list_lim) for v in lst]

    # If the limit is very small (<3) fall back to head‑only + sentinel to avoid
    # degenerate splits.
    if list_lim < 3:
        kept = [_truncate_value(v, str_lim, list_lim) for v in lst[:list_lim]]
        kept.append(f"… (omitted {len(lst) - list_lim} items)…")
        return kept

    head_len = list_lim // 2
    tail_len = list_lim - head_len

    head = [_truncate_value(v, str_lim, list_lim) for v in lst[:head_len]]
    tail = [_truncate_value(v, str_lim, list_lim) for v in lst[-tail_len:]]

    omitted = len(lst) - (head_len + tail_len)
    sentinel = f"… (omitted {omitted} items)…"
    return head + [sentinel] + tail


# ---------------------------------------------------------------------------
#  Recursive truncation
# ---------------------------------------------------------------------------


def _truncate_value(value: Any, str_limit: int, list_limit: int) -> Any:
    """Recursively truncate *value* using the current per‑type limits."""

    if isinstance(value, str):
        return _truncate_string_middle(value, str_limit)

    if isinstance(value, list):
        return _truncate_list_middle(value, str_limit, list_limit)

    if isinstance(value, dict):
        return {k: _truncate_value(v, str_limit, list_limit) for k, v in value.items()}

    return value


def truncate(value: Any, size_limit: int) -> Any:
    """
    Truncate the given value (recursively) so that its string representation
    does not exceed size_limit characters. Uses binary search to find the
    largest str_limit and list_limit that fit.
    """

    # Fast path: plain strings don't need the binary search machinery.
    if isinstance(value, str):
        return _truncate_string_middle(value, size_limit)

    def measure(val):
        try:
            return len(str(val))
        except Exception:
            return sys.getsizeof(val)

    # Reasonable bounds for string and list limits
    STR_MIN, STR_MAX = min(8, size_limit), size_limit
    LIST_MIN, LIST_MAX = 1, 2**12

    # Binary search for the largest str_limit and list_limit that fit
    best = None

    # We'll search str_limit first, then list_limit, but can do both together
    # For practical purposes, do a grid search with binary search on str_limit for each list_limit
    # (since lists are usually the main source of bloat)
    # We'll do binary search on list_limit, and for each, binary search on str_limit

    # Outer binary search on list_limit
    l_lo, l_hi = LIST_MIN, LIST_MAX
    while l_lo <= l_hi:
        l_mid = (l_lo + l_hi) // 2

        # Inner binary search on str_limit
        s_lo, s_hi = STR_MIN, STR_MAX
        local_best = None
        while s_lo <= s_hi:
            s_mid = (s_lo + s_hi) // 2
            truncated = _truncate_value(value, s_mid, l_mid)
            size = measure(truncated)
            if size <= size_limit:
                local_best = truncated
                s_lo = s_mid + 1  # try to increase str_limit
            else:
                s_hi = s_mid - 1  # decrease str_limit

        if local_best is not None:
            best = local_best
            l_lo = l_mid + 1  # try to increase list_limit
        else:
            l_hi = l_mid - 1  # decrease list_limit

    # If nothing fits, fall back to the most aggressive truncation
    if best is None:
        best = _truncate_value(value, STR_MIN, LIST_MIN)

    return best
