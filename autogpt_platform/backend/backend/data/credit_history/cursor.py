import base64
import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CreditHistoryCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    snapshot_at: datetime
    transaction_time: datetime
    group_id: str
    scope: str


def cursor_scope(user_id: str, organization_id: str | None, kind: str | None) -> str:
    ledger = (
        f"org:{organization_id}" if organization_id is not None else f"user:{user_id}"
    )
    return hashlib.sha256(f"{ledger}|{kind or ''}".encode()).hexdigest()


def as_utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def decode_cursor(cursor: str, scope: str) -> CreditHistoryCursor:
    try:
        if len(cursor) > 4096:
            raise ValueError("Cursor too long")
        decoded = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True
        )
        position = CreditHistoryCursor.model_validate_json(decoded)
        if (
            position.scope != scope
            or position.snapshot_at.tzinfo is None
            or position.transaction_time.tzinfo is None
            or position.transaction_time > position.snapshot_at
            or not position.group_id
        ):
            raise ValueError("Invalid cursor fields")
        return position
    except ValueError as exc:
        raise ValueError("Invalid credit history cursor") from exc


def encode_cursor(position: CreditHistoryCursor) -> str:
    # Cursors are unsigned, untrusted pagination state. The server derives the
    # expected scope and query's ledger owner separately from the authorized
    # request; the scope hash is only a reuse check, not authorization.
    return (
        base64.urlsafe_b64encode(position.model_dump_json().encode())
        .decode()
        .rstrip("=")
    )
