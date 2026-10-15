import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.data.credit_history.cursor import (
    CreditHistoryCursor,
    as_utc,
    cursor_scope,
    decode_cursor,
    encode_cursor,
)


def test_cursor_preserves_snapshot_and_tie_breaker():
    position = CreditHistoryCursor(
        snapshot_at=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        transaction_time=datetime(2026, 9, 4, 9, 0, 0, 123000, tzinfo=timezone.utc),
        group_id="execution:run-2",
        scope=cursor_scope("user", None, "USAGE"),
    )
    assert decode_cursor(encode_cursor(position), position.scope) == position


@pytest.mark.parametrize(
    "user_id,org_id,kind",
    [("other", None, None), ("user", "org", None), ("user", None, "USAGE")],
)
def test_cursor_is_bound_to_wallet_and_filter(user_id, org_id, kind):
    time = datetime(2026, 9, 5, tzinfo=timezone.utc)
    position = CreditHistoryCursor(
        snapshot_at=time,
        transaction_time=time,
        group_id="transaction:key",
        scope=cursor_scope("user", None, None),
    )
    with pytest.raises(ValueError, match="Invalid credit history cursor"):
        decode_cursor(encode_cursor(position), cursor_scope(user_id, org_id, kind))


@pytest.mark.parametrize("cursor", ["", "!!!", "a" * 5000, "e30", "bm90LWpzb24"])
def test_malformed_cursor(cursor):
    with pytest.raises(ValueError, match="Invalid credit history cursor"):
        decode_cursor(cursor, "scope")


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},
        {"snapshot_at": "2026-09-01T00:00:00"},
        {"transaction_time": "2026-09-10T00:00:00Z"},
        {"group_id": ""},
        {"extra": "unexpected"},
    ],
)
def test_cursor_rejects_invalid_fields(overrides):
    fields = {
        "version": 1,
        "snapshot_at": "2026-09-05T00:00:00Z",
        "transaction_time": "2026-09-04T00:00:00Z",
        "group_id": "execution:run",
        "scope": "scope",
        **overrides,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(fields).encode()).decode()
    with pytest.raises(ValueError, match="Invalid credit history cursor"):
        decode_cursor(encoded, "scope")


def test_timezone_conversion_preserves_instant():
    paris = datetime(2026, 9, 5, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert as_utc(paris) == datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
