import sys
from bisect import bisect_right
from itertools import accumulate
from typing import Any, Callable

LIST_MAX = 2**12
# A list's "… (omitted N items)…" item: 19 characters plus N's digits, quotes and ", ".
_LIST_MARKER_COST = 23

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

    # Head and tail both grow with the limit, so a larger limit keeps every item a
    # smaller one kept; `_price` relies on that.
    head_len = max(1, list_lim // 2)
    tail_len = list_lim - head_len

    head = [_truncate_value(v, str_lim, list_lim) for v in lst[:head_len]]
    tail = [_truncate_value(v, str_lim, list_lim) for v in lst[len(lst) - tail_len :]]

    omitted = len(lst) - list_lim
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
    does not exceed size_limit characters, keeping as many list items as fit
    and then the longest strings that fit.
    """

    if size_limit < 0:
        raise ValueError("size_limit must be non-negative")

    # Fast path: plain strings don't need the limits search.
    if isinstance(value, str):
        return _truncate_string_middle(value, size_limit)

    def measure(val):
        try:
            return len(str(val))
        except Exception:
            return sys.getsizeof(val)

    # Most payloads fit with the maximal limits; this still caps lists at LIST_MAX,
    # which returning `value` would not.
    maximal = _truncate_value(value, size_limit, LIST_MAX)
    if measure(maximal) <= size_limit:
        return maximal

    if not isinstance(value, (list, dict)):
        return value

    return _truncate_to_fit(value, size_limit, measure)


def _truncate_to_fit(
    value: list | dict, size_limit: int, measure: Callable[[Any], int]
) -> Any:
    """Keep as many list items as fit with strings cut to their minimum, then
    the longest strings that fit, as a search over both limits would, but
    with one walk of *value* and a few builds instead of one build per probe.
    """
    str_min = min(8, size_limit)
    added, strings = _price(value, str_min)
    sizes = list(accumulate(added))  # sizes[n]: size with lists cut to n items
    fitting = [n for n in range(1, LIST_MAX + 1) if sizes[n] <= size_limit]
    if not fitting:
        smallest = min(range(1, LIST_MAX + 1), key=lambda n: (sizes[n], -n))
        return _truncate_value(value, str_min, smallest)
    list_limit = fitting[-1]

    kept = sorted((m, r, cut) for at, m, r, cut in strings if at <= list_limit)
    lengths = [m for m, _, _ in kept]
    rest = sizes[list_limit] - sum(cut for _, _, cut in kept)
    whole = list(accumulate((r for _, r, _ in kept), initial=0))
    density = list(
        accumulate(((r - m - 2) / (m or 1) for m, r, _ in kept), initial=0.0)
    )

    def estimate(str_limit: int) -> float:
        i = bisect_right(lengths, str_limit)
        cut = len(kept) - i
        escapes = str_limit * (density[-1] - density[i])
        return rest + whole[i] + cut * (str_limit + 2) + escapes

    lo, hi = str_min, size_limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(mid) <= size_limit:
            lo = mid
        else:
            hi = mid - 1

    # The estimate spreads each string's escapes evenly; correct it by measuring.
    str_limit = lo
    for _ in range(3):
        if str_limit <= str_min:
            break
        result = _truncate_value(value, str_limit, list_limit)
        over = measure(result) - size_limit
        if over <= 0:
            return result
        cut = max(1, len(kept) - bisect_right(lengths, str_limit))
        str_limit = max(str_min, str_limit - (over + cut - 1) // cut)
    return _truncate_value(value, str_min, list_limit)


def _price(
    value: list | dict, str_min: int
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Return what *value*'s size grows by at each list limit with strings cut to
    *str_min* (exact but for list markers, priced at their widest), and each
    string as (the list limit that keeps it, length, repr size, cut repr size).
    """
    added = [0] * (LIST_MAX + 1)
    strings: list[tuple[int, int, int, int]] = []

    def leaf(node: Any, kept_from: int) -> int:
        if not isinstance(node, str):
            return _repr_len(node)
        size = len(repr(node))
        cut = size
        if len(node) > str_min:
            cut = len(repr(_truncate_string_middle(node, str_min)))
        strings.append((kept_from, len(node), size, cut))
        return cut

    def walk(node: list | dict, kept_from: int) -> None:
        if isinstance(node, dict):
            size = 2 + 2 * max(len(node) - 1, 0)  # braces and ", " separators
            for key, child in node.items():
                size += _repr_len(key) + 2
                if isinstance(child, (list, dict)):
                    walk(child, kept_from)
                else:
                    size += leaf(child, kept_from)
            added[kept_from] += size
            return

        n = len(node)
        if n > kept_from:  # the marker shows while the limit is under n
            marker = _LIST_MARKER_COST + len(str(n - 1))
            added[kept_from] += marker
            if n <= LIST_MAX:
                added[n] -= marker
        added[kept_from] += 2
        for i in _indices_ever_kept(n):
            # Kept once its head or tail slot exists, or the whole list fits.
            from_head = 1 if i == 0 else 2 * i + 2
            from_tail = max(2, 2 * (n - 1 - i) + 1)
            at = max(kept_from, min(n, from_head, from_tail))
            child = node[i]
            if isinstance(child, (list, dict)):
                added[at] += 2 if i else 0
                walk(child, at)
            else:
                added[at] += (2 if i else 0) + leaf(child, at)

    walk(value, 1)
    return added, strings


def _indices_ever_kept(n: int) -> range | list[int]:
    if n <= LIST_MAX:
        return range(n)
    half = LIST_MAX // 2
    return [*range(half), *range(n - (LIST_MAX - half), n)]


def _repr_len(value: Any) -> int:
    try:
        return len(repr(value))
    except Exception:
        return sys.getsizeof(value)
