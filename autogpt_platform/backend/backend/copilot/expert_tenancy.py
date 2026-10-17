"""An expert's tenancy, checked once per turn.

An expert's turn starts by checking that the expert is still the owner's own
active PRIVATE expert, and which personal organization and team its work
belongs to (``resolve_private_expert_tenancy``: three queries, ~0.45 s in
prod). Its identity block checks the same thing again a moment later, before
the model is called. Inside :func:`sharing_expert_tenancy` the turn remembers
the first answer and the identity block uses it. Each turn opens its own
share, so no turn sees another's answer, and a check made outside a turn
always reads.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

Tenancy = tuple[str, str | None]
"""The owner's personal (organization id, team id)."""

_checked: ContextVar[dict[tuple[str, str], Tenancy] | None] = ContextVar(
    "copilot_checked_expert_tenancy", default=None
)


@contextmanager
def sharing_expert_tenancy() -> Iterator[None]:
    """Let the rest of the turn reuse each expert tenancy it checks."""
    token = _checked.set({})
    try:
        yield
    finally:
        _checked.reset(token)


def remember_expert_tenancy(user_id: str, expert_id: str, tenancy: Tenancy) -> None:
    """Note what this turn's tenancy check of *expert_id* found."""
    checked = _checked.get()
    if checked is not None:
        checked[(user_id, expert_id)] = tenancy


def checked_expert_tenancy(user_id: str, expert_id: str) -> Tenancy | None:
    """What this turn's tenancy check of *expert_id* found, or ``None`` when
    it hasn't checked (or isn't a turn)."""
    checked = _checked.get()
    return None if checked is None else checked.get((user_id, expert_id))
