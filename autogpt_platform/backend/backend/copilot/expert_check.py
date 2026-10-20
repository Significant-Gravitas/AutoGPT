"""An expert's turn checks its expert once.

Before an expert's turn runs, the executor checks that the expert is still
the owner's own active PRIVATE expert and which personal organization and
team its work belongs to. It then waits for a fresh hire's setup, and the
engine builds the expert's identity prompt. Each of those used to read the
expert again (four database-manager calls, ~1.1 s of every expert turn on a
preview). ``check_private_expert`` answers all three in one call; inside
:func:`sharing_expert_checks` the turn remembers what it found and the setup
wait and the identity prompt use that. Each turn opens its own share, so no
turn sees another's check, and a lookup made outside a turn always reads.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from backend.api.features.experts.models import CheckedExpert

_checked: ContextVar[dict[tuple[str, str], CheckedExpert] | None] = ContextVar(
    "copilot_checked_expert", default=None
)


@contextmanager
def sharing_expert_checks() -> Iterator[None]:
    """Let the rest of the turn reuse each expert check it makes."""
    token = _checked.set({})
    try:
        yield
    finally:
        _checked.reset(token)


def remember_expert_check(user_id: str, expert_id: str, check: CheckedExpert) -> None:
    """Note what this turn's check of *expert_id* found."""
    checked = _checked.get()
    if checked is not None:
        checked[(user_id, expert_id)] = check


def checked_expert(user_id: str, expert_id: str) -> CheckedExpert | None:
    """What this turn's check of *expert_id* found, or ``None`` when it
    hasn't checked (or isn't a turn)."""
    checked = _checked.get()
    return None if checked is None else checked.get((user_id, expert_id))
