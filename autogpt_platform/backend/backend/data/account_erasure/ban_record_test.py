"""The ban record kept when a banned account is erased: only for a ban still
in force, keyed so the table alone names nobody, as long as the ban would
have lasted and 3 years at most, checked at sign-up and purged after."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountBanCategory, AccountDeletionSource

from backend.data.account_erasure import ban_record
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
EMAIL = "sam@example.com"
BANNED = Subject(
    user_id=USER_ID,
    email=EMAIL,
    has_auth_identity=True,
    request_source=AccountDeletionSource.SETTINGS,
    banned=True,
)


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(
        ban_record.settings.secrets, "encryption_key", "test-encryption-key"
    )
    table = MagicMock()
    table.prisma.return_value.upsert = AsyncMock()
    table.prisma.return_value.find_first = AsyncMock(return_value=None)
    table.prisma.return_value.delete_many = AsyncMock(return_value=3)
    monkeypatch.setattr(ban_record, "AccountBanRecord", table)
    return table.prisma.return_value


def test_the_hash_is_keyed_and_alike_in_any_case(monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "key-one")
    first = ban_record.email_hash(" Sam@Example.com ")
    assert first == ban_record.email_hash(EMAIL)
    assert first is not None and EMAIL not in first
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "key-two")
    assert ban_record.email_hash(EMAIL) != first


def test_without_a_key_there_is_no_hash(monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "")

    assert ban_record.email_hash(EMAIL) is None


@pytest.mark.asyncio
async def test_a_ban_for_good_is_recorded_for_3_years(records):
    before = datetime.now(UTC)

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    data = records.upsert.await_args.kwargs["data"]["create"]
    assert data["emailHash"] == ban_record.email_hash(EMAIL)
    assert data["category"] == AccountBanCategory.ADMIN
    assert EMAIL not in str(records.upsert.await_args)
    lasts = data["deleteAfter"] - data["recordedAt"]
    assert lasts == timedelta(days=3 * 365)
    assert data["recordedAt"] >= before
    assert report.counts == {"ban_records": 1}


@pytest.mark.asyncio
async def test_a_ban_that_ends_sooner_is_kept_only_until_then(records):
    ends = datetime.now(UTC) + timedelta(days=30)
    temporary = BANNED.model_copy(update={"ban_expires": ends})

    await ban_record.keep_ban_record(temporary, dry_run=False)

    assert records.upsert.await_args.kwargs["data"]["create"]["deleteAfter"] == ends


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        {"banned": False},
        {"ban_expires": datetime(2020, 1, 1, tzinfo=UTC)},
        {"email": None},
    ],
)
async def test_no_record_without_a_ban_in_force(records, update):
    report = await ban_record.keep_ban_record(
        BANNED.model_copy(update=update), dry_run=False
    )

    records.upsert.assert_not_awaited()
    assert report.counts == {"ban_records": 0}


@pytest.mark.asyncio
async def test_a_dry_run_writes_no_record(records):
    report = await ban_record.keep_ban_record(BANNED, dry_run=True)

    records.upsert.assert_not_awaited()
    assert report.status == StepStatus.WOULD_DO
    assert report.counts == {"ban_records": 1}


@pytest.mark.asyncio
async def test_without_a_key_the_step_is_skipped(records, monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "")

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    assert report.status == StepStatus.SKIPPED
    records.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_sign_up_finds_only_a_record_still_in_force(records):
    now = datetime(2026, 11, 2, tzinfo=UTC)
    records.find_first.return_value = object()

    assert await ban_record.is_banned_address("SAM@example.com", now=now)
    assert records.find_first.await_args.kwargs["where"] == {
        "emailHash": ban_record.email_hash(EMAIL),
        "deleteAfter": {"gt": now},
    }
    records.find_first.return_value = None
    assert not await ban_record.is_banned_address(EMAIL, now=now)


@pytest.mark.asyncio
async def test_records_past_their_date_are_purged(records):
    now = datetime(2026, 11, 2, tzinfo=UTC)

    assert await ban_record.purge_expired_ban_records(now=now) == 3
    assert records.delete_many.await_args.kwargs["where"] == {
        "deleteAfter": {"lte": now}
    }
