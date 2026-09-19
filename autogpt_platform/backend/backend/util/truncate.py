import sys
from typing import Any

# ---------------------------------------------------------------------------
#  String helpers
# ---------------------------------------------------------------------------


def _truncate_string_middle(value: str, limit: int) -> str:
    """Shorten *value* to *limit* chars by removing the **middle** portion."""

    if len(value) <= limit:
        return value

    if limit == 0:
        return ""

    omitted = len(value)
    marker = f"… (omitted {omitted} chars)…"
    while len(marker) <= limit:
        retained = limit - len(marker)
        if retained < 2:
            break
        actual_omitted = len(value) - retained
        if actual_omitted == omitted:
            head_len = (retained + 1) // 2
            tail_len = retained - head_len
            tail = value[-tail_len:] if tail_len else ""
            return f"{value[:head_len]}{marker}{tail}"
        omitted = actual_omitted
        marker = f"… (omitted {omitted} chars)…"

    retained = limit - 1
    head_len = (retained + 1) // 2
    tail_len = retained - head_len
    tail = value[-tail_len:] if tail_len else ""
    return f"{value[:head_len]}…{tail}"


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
#  Dict helpers
# ---------------------------------------------------------------------------


def _truncate_dict_middle(
    dct: dict[Any, Any], str_lim: int, dict_lim: int
) -> dict[Any, Any]:
    """Return *dct* truncated to *dict_lim* entries, removing from the middle.

    Mirrors :func:`_truncate_list_middle`. Without an entry bound a dict cannot
    be shrunk at all, so a wide dict stays over the requested size limit no
    matter how far the string and list limits are lowered.
    """

    if len(dct) <= dict_lim:
        return {k: _truncate_value(v, str_lim, dict_lim) for k, v in dct.items()}

    items = list(dct.items())

    if dict_lim < 3:
        kept = {k: _truncate_value(v, str_lim, dict_lim) for k, v in items[:dict_lim]}
        kept[f"… (omitted {len(dct) - dict_lim} keys)…"] = ""
        return kept

    head_len = dict_lim // 2
    tail_len = dict_lim - head_len

    kept = {k: _truncate_value(v, str_lim, dict_lim) for k, v in items[:head_len]}
    kept[f"… (omitted {len(dct) - head_len - tail_len} keys)…"] = ""
    kept |= {k: _truncate_value(v, str_lim, dict_lim) for k, v in items[-tail_len:]}
    return kept


def _drop_last_entry(container: Any) -> Any | None:
    """Return *container* without its last entry, or ``None`` if it has none."""

    if isinstance(container, dict) and container:
        kept = dict(container)
        kept.pop(next(reversed(kept)))
        return kept

    if isinstance(container, list) and container:
        return container[:-1]

    return None


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
        return _truncate_dict_middle(value, str_limit, list_limit)

    return value


def truncate(value: Any, size_limit: int) -> Any:
    """
    Truncate the given value (recursively) so that its string representation
    does not exceed size_limit characters. Uses binary search to find the
    largest str_limit and list_limit that fit.
    """

    if size_limit < 0:
        raise ValueError("size_limit must be non-negative")

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

    # If nothing fits, fall back to the most aggressive truncation. The search
    # starts at STR_MIN, so walk the remaining string budget down to 0 first: a
    # one-character value can still make a container fit where STR_MIN cannot,
    # and keeping the key is preferable to dropping the entry.
    if best is None:
        for str_limit in range(STR_MIN - 1, -1, -1):
            candidate = _truncate_value(value, str_limit, LIST_MIN)
            if measure(candidate) <= size_limit:
                best = candidate
                break

    if best is None:
        best = _truncate_value(value, STR_MIN, LIST_MIN)

    # The fallback above still keeps every top‑level entry, so a value whose
    # outermost keys or brackets alone exceed size_limit can come back over
    # budget. Drop entries until it fits so the documented bound holds.
    while measure(best) > size_limit:
        smaller = _drop_last_entry(best)
        if smaller is None:
            break
        best = smaller

    return best
