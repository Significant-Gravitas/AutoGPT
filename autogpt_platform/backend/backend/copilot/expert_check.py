"""An expert's turn checks its expert once.

Before an expert's turn runs, the executor checks the expert with
``check_private_expert``: that it is still the owner's own active PRIVATE
expert, which personal organization and team its work belongs to, how far a
fresh hire's setup got, and the columns its identity prompt is built from.
Inside :func:`sharing_expert_checks` the turn remembers that check, and the
hire-setup wait and the identity prompt use it instead of reading the expert
again. Each turn opens its own share, so no turn sees another's check, and a
lookup made outside a turn always reads. An auto-continue drops the turn's
checks first (:func:`forget_expert_checks`): it runs after the turn's own
work, by when the expert may have been archived.
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


def forget_expert_checks() -> None:
    """Make the turn's next look at any expert read it again."""
    checked = _checked.get()
    if checked is not None:
        checked.clear()
