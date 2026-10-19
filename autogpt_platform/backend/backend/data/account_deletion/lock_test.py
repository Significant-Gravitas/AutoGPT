"""The lock records when the account was stopped, once; the unlock lifts only
what the deletion did (its ban, its marketing opt-out) and clears the lock.
Dry runs only count."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_deletion import lock
from backend.data.account_deletion.activity import BAN_REASON, OPT_OUT_SOURCE
from backend.data.account_deletion.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
NOW = datetime(2026, 10, 15, 9, 30, tzinfo=timezone.utc)


def _table(count: int = 1) -> MagicMock:
    table = MagicMock()
    table.prisma.return_value.count = AsyncMock(return_value=count)
    table.prisma.return_value.update_many = AsyncMock(return_value=count)
    return table


@pytest.fixture
def tables(monkeypatch):
    users, identities = _table(), _table()
    monkeypatch.setattr(lock, "User", users)
    monkeypatch.setattr(lock, "AuthUser", identities)
    monkeypatch.setattr(lock, "_now", lambda: NOW)
    return MagicMock(
        users=users.prisma.return_value, ids=identities.prisma.return_value
    )


@pytest.mark.asyncio
async def test_the_lock_is_recorded_once_and_names_the_due_date(tables):
    subject = Subject(user_id=USER_ID, has_user_row=True)

    report = await lock.record_lock(subject, dry_run=False)

    tables.users.update_many.assert_awaited_once_with(
        where={"id": USER_ID, "deletionLockedAt": None},
        data={"deletionLockedAt": NOW},
    )
    assert report.status == StepStatus.DONE
    assert report.counts == {"locked": 1}
    assert "2026-10-29" in report.note


@pytest.mark.asyncio
async def test_locking_again_keeps_the_first_lock(tables):
    earlier = NOW - timedelta(days=3)
    subject = Subject(user_id=USER_ID, has_user_row=True, locked_at=earlier)

    report = await lock.record_lock(subject, dry_run=False)

    tables.users.update_many.assert_not_awaited()
    assert report.counts == {"locked": 0}
    assert "2026-10-26" in report.note


@pytest.mark.asyncio
async def test_a_dry_run_lock_writes_nothing(tables):
    report = await lock.record_lock(
        Subject(user_id=USER_ID, has_user_row=True), dry_run=True
    )

    tables.users.update_many.assert_not_awaited()
    assert report.status == StepStatus.WOULD_DO


@pytest.mark.asyncio
async def test_without_an_account_row_the_lock_is_skipped(tables):
    report = await lock.record_lock(Subject(user_id=USER_ID), dry_run=False)

    assert report.status == StepStatus.SKIPPED
    assert "--force" in report.note
    tables.users.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_unlock_lifts_only_what_the_deletion_did(tables):
    report = await lock.lift_lock(Subject(user_id=USER_ID), dry_run=False)

    (ban,) = tables.ids.update_many.await_args_list
    # Only the ban the lock wrote: its reason, and no expiry. The lock never
    # overwrites a ban already in force (`activity.disable_sign_in`), so any
    # other ban, such as one for abuse, stays.
    assert ban.kwargs["where"] == {
        "id": USER_ID,
        "banned": True,
        "banReason": BAN_REASON,
        "banExpires": None,
    }
    assert ban.kwargs["data"] == {
        "banned": False,
        "banReason": None,
        "banExpires": None,
    }
    opt_out, cleared = tables.users.update_many.await_args_list
    assert opt_out.kwargs["where"] == {
        "id": USER_ID,
        "marketingOptOutSource": OPT_OUT_SOURCE,
    }
    assert opt_out.kwargs["data"] == {
        "marketingOptOutAt": None,
        "marketingOptOutSource": None,
    }
    assert cleared.kwargs["data"] == {"deletionLockedAt": None}
    assert report.counts == {
        "ban_lifted": 1,
        "marketing_opt_out_reverted": 1,
        "lock_cleared": 1,
    }
    assert "stay cancelled" in report.note
    assert "the requester sets them up again" in report.note


@pytest.mark.asyncio
async def test_a_dry_run_unlock_only_counts(tables):
    report = await lock.lift_lock(Subject(user_id=USER_ID), dry_run=True)

    tables.ids.update_many.assert_not_awaited()
    tables.users.update_many.assert_not_awaited()
    assert report.status == StepStatus.WOULD_DO
