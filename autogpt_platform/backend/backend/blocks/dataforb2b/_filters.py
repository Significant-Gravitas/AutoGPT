"""Filter-building helpers for DataForB2B search blocks.

Mirrors the logic of the Dify integration so the two stay in parity. A search
filter is ``{op, conditions: [{column, type, value, value2?}]}``.
"""

from enum import Enum
from typing import Any, Optional


def _to_str(x: Any) -> str:
    """Return an enum member's value, or the stringified value otherwise."""
    if isinstance(x, Enum):
        return str(x.value)
    return str(x)


def coerce_scalar(value: Any) -> Any:
    """Coerce a filter value string into bool/number when it clearly is one."""
    if isinstance(value, (int, float, bool)):
        return value
    s = str(value).strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return s


def build_slot_condition(column: Any, operator: Any, raw: Any) -> Optional[dict]:
    """Build one filter condition from a (column, operator, value) slot.

    Returns None when the slot is empty.
    - ``in``      -> value is a comma-separated list
    - ``between`` -> value is "min,max"
    - ``like``    -> raw string kept as-is (text pattern)
    - others      -> single value, coerced to bool/number when applicable
    """
    if not column or raw is None or str(raw).strip() == "":
        return None
    column = _to_str(column)
    op = (_to_str(operator).strip() if operator else "=") or "="

    if op == "in":
        items = [x.strip() for x in str(raw).split(",") if x.strip()]
        if not items:
            return None
        return {
            "column": column,
            "type": "in",
            "value": [coerce_scalar(x) for x in items],
        }

    if op == "between":
        parts = [x.strip() for x in str(raw).split(",") if x.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"Operator 'between' on '{column}' needs two comma-separated "
                "values, e.g. 3,7"
            )
        return {
            "column": column,
            "type": "between",
            "value": coerce_scalar(parts[0]),
            "value2": coerce_scalar(parts[1]),
        }

    if op == "like":
        return {"column": column, "type": "like", "value": str(raw)}

    return {"column": column, "type": op, "value": coerce_scalar(raw)}


def finalize_filters(conditions: list, match: Any, advanced: Any) -> Optional[dict]:
    """Combine slot conditions (with AND/OR) and optional advanced JSON filters."""
    op = str(match).strip().lower() if match else "and"
    if op not in ("and", "or"):
        op = "and"

    group = {"op": op, "conditions": conditions} if conditions else None

    if advanced:
        if isinstance(advanced, dict) and "conditions" in advanced:
            adv = advanced
        elif isinstance(advanced, list):
            adv = {"op": "and", "conditions": advanced}
        else:  # a single bare condition dict
            adv = {"op": "and", "conditions": [advanced]}
        return {"op": "and", "conditions": [group, adv]} if group else adv

    return group
