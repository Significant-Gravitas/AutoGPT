"""Telling a chat-message insert on an id that is already saved apart from
the other unique violation a ``ChatMessage`` insert can raise."""

import re

from prisma.errors import UniqueViolationError

_UNIQUE_FIELDS_IN_MESSAGE = re.compile(r"fields: \(([^)]*)\)")


def _unique_violation_fields(exc: UniqueViolationError) -> list[str] | None:
    """The fields a unique violation failed on, or None when unknown.

    Prisma reports them as ``meta["target"]`` (e.g. ``["id"]``) and in the
    message as ``Unique constraint failed on the fields: (`id`)``. Only the
    message survives the database manager's RPC, which rebuilds the error
    from it with ``code`` and ``meta`` empty, so read both."""
    target = exc.meta.get("target") if isinstance(exc.meta, dict) else None
    if isinstance(target, list):
        return [str(field) for field in target]
    match = _UNIQUE_FIELDS_IN_MESSAGE.search(str(exc))
    if match is None:
        return None
    return re.findall(r"`([^`]+)`", match.group(1))


def is_duplicate_chat_message_id_error(exc: BaseException) -> bool:
    """True iff *exc* is a ``UniqueViolationError`` on the ``ChatMessage``
    row ``id``, i.e. the caller supplied an id that is already saved.

    ``ChatMessage`` has two unique constraints: ``id`` (the idempotency
    signal, which callers drop quietly) and ``(sessionId, sequence)`` (a
    sequence race, which callers retry). Only call this on errors from a
    ``ChatMessage`` insert."""
    if not isinstance(exc, UniqueViolationError):
        return False
    return _unique_violation_fields(exc) == ["id"]
