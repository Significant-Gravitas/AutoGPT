"""The ban record kept when a banned account is erased: only for a ban still
in force, keyed so the table alone names nobody, as long as the ban would
have lasted and 3 years at most, checked at sign-up and purged after."""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountBanCategory, AccountDeletionSource
from prisma.errors import UniqueViolationError

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
        ban_record.settings.secrets, "ban_record_hash_key", "test-ban-record-key"
    )
    table = MagicMock()
    table.prisma.return_value.create = AsyncMock()
    table.prisma.return_value.update_many = AsyncMock(return_value=0)
    table.prisma.return_value.find_first = AsyncMock(return_value=None)
    table.prisma.return_value.delete_many = AsyncMock(return_value=3)
    monkeypatch.setattr(ban_record, "AccountBanRecord", table)
    return table.prisma.return_value


def test_the_hash_is_keyed_and_alike_in_any_case(monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "key-one")
    first = ban_record.email_hash(" Sam@Example.com ")
    assert first == ban_record.email_hash(EMAIL)
    assert first is not None and EMAIL not in first
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "key-two")
    assert ban_record.email_hash(EMAIL) != first


def test_the_hash_does_not_follow_the_encryption_key(monkeypatch):
    """Rotating ENCRYPTION_KEY must not make the records stop matching."""
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "key")
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "old")
    first = ban_record.email_hash(EMAIL)
    monkeypatch.setattr(ban_record.settings.secrets, "encryption_key", "new")

    assert ban_record.email_hash(EMAIL) == first


def test_without_a_key_there_is_no_hash(monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "")

    assert ban_record.email_hash(EMAIL) is None
    assert ban_record.email_hashes(EMAIL) == []


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        (" Sam@Example.com ", "sam@example.com"),
        ("sam+shop@example.com", "sam@example.com"),
        ("sam+a+b@example.com", "sam@example.com"),
        ("s.a.m@example.com", "s.a.m@example.com"),
        ("S.am+x@Gmail.com", "sam@gmail.com"),
        ("s.am@googlemail.com", "sam@gmail.com"),
        ("+only@example.com", "+only@example.com"),
        ("...@gmail.com", "...@gmail.com"),
        ("no-at-sign", "no-at-sign"),
    ],
)
def test_the_spellings_of_one_mailbox_share_a_canonical_form(spelling, canonical):
    """`+tag` goes on every domain; dots go only on Gmail, where they reach
    the same mailbox. A part that would come out empty is left as it is."""
    assert ban_record.canonical_address(spelling) == canonical


def test_the_hash_is_of_the_canonical_form(monkeypatch):
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "key")

    assert ban_record.email_hash("s.am+x@googlemail.com") == ban_record.email_hash(
        "sam@gmail.com"
    )
    assert ban_record.email_hash("s.am@example.com") != ban_record.email_hash(
        "sam@example.com"
    )


def test_an_address_folded_by_the_rule_also_has_its_old_hash(monkeypatch):
    """Records written before the canonical form hold the hash of the
    trimmed, lower-cased address, so the lookup asks for both."""
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "key")
    old = hmac.new(b"key", b"sam+x@example.com", hashlib.sha256).hexdigest()

    assert ban_record.email_hashes(" Sam+X@example.com ") == [
        ban_record.email_hash("sam@example.com"),
        old,
    ]
    assert ban_record.email_hashes(EMAIL) == [ban_record.email_hash(EMAIL)]


@pytest.mark.asyncio
async def test_a_ban_for_good_is_recorded_for_3_years(records):
    before = datetime.now(UTC)

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    data = records.create.await_args.kwargs["data"]
    assert data["emailHash"] == ban_record.email_hash(EMAIL)
    assert data["category"] == AccountBanCategory.ADMIN
    assert EMAIL not in str(records.create.await_args)
    lasts = data["deleteAfter"] - data["recordedAt"]
    assert lasts == timedelta(days=3 * 365)
    assert data["recordedAt"] >= before
    assert report.counts == {"ban_records": 1}


@pytest.mark.asyncio
async def test_a_ban_that_ends_sooner_is_kept_only_until_then(records):
    ends = datetime.now(UTC) + timedelta(days=30)
    temporary = BANNED.model_copy(update={"ban_expires": ends})

    await ban_record.keep_ban_record(temporary, dry_run=False)

    assert records.create.await_args.kwargs["data"]["deleteAfter"] == ends


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

    records.create.assert_not_awaited()
    assert report.counts == {"ban_records": 0}


@pytest.mark.asyncio
async def test_a_dry_run_writes_no_record(records):
    report = await ban_record.keep_ban_record(BANNED, dry_run=True)

    records.create.assert_not_awaited()
    assert report.status == StepStatus.WOULD_DO
    assert report.counts == {"ban_records": 1}


@pytest.mark.asyncio
async def test_without_a_key_the_step_fails_so_the_ban_is_not_lost(
    records, monkeypatch
):
    """Skipping would let the identity, and the ban with it, go for good."""
    monkeypatch.setattr(ban_record.settings.secrets, "ban_record_hash_key", "")

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    assert report.status == StepStatus.FAILED
    assert "BAN_RECORD_HASH_KEY" in report.note
    records.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_rerun_keeps_the_first_record_and_its_end(records):
    """A retry must not push the record's end out; only a record that ran out
    and was not purged yet starts over."""
    records.create.side_effect = _conflict()
    records.find_first.return_value = MagicMock()

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    assert report.status == StepStatus.DONE
    where = records.update_many.await_args.kwargs["where"]
    assert where["emailHash"] == ban_record.email_hash(EMAIL)
    assert "lte" in where["deleteAfter"]
    records.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_record_purged_in_between_is_created_again(records):
    """The expired record went between the conflict and the restart, so the
    restart touched nothing and nothing is kept: the record is created again."""
    records.create.side_effect = [_conflict(), None]

    report = await ban_record.keep_ban_record(BANNED, dry_run=False)

    assert report.status == StepStatus.DONE
    assert records.create.await_count == 2


@pytest.mark.asyncio
async def test_a_second_conflict_fails_rather_than_report_a_record(records):
    records.create.side_effect = [_conflict(), _conflict()]

    with pytest.raises(UniqueViolationError):
        await ban_record.keep_ban_record(BANNED, dry_run=False)


def _conflict() -> UniqueViolationError:
    return UniqueViolationError(
        {"user_facing_error": {"message": "Unique constraint failed"}}
    )


@pytest.mark.asyncio
async def test_sign_up_finds_only_a_record_still_in_force(records):
    now = datetime(2026, 11, 2, tzinfo=UTC)
    records.find_first.return_value = object()

    assert await ban_record.is_banned_address("SAM@example.com", now=now)
    assert records.find_first.await_args.kwargs["where"] == {
        "emailHash": {"in": [ban_record.email_hash(EMAIL)]},
        "deleteAfter": {"gt": now},
    }
    records.find_first.return_value = None
    assert not await ban_record.is_banned_address(EMAIL, now=now)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spelling",
    ["sam+again@example.com", "SAM+x@Example.com", " sam@example.com "],
)
async def test_sign_up_with_another_spelling_finds_the_record(records, spelling):
    records.find_first.return_value = object()

    assert await ban_record.is_banned_address(spelling)
    digests = records.find_first.await_args.kwargs["where"]["emailHash"]["in"]
    assert ban_record.email_hash(EMAIL) in digests


@pytest.mark.asyncio
async def test_a_gmail_record_matches_its_dotted_and_googlemail_forms(records):
    """The record of an address banned on Gmail is written under the
    canonical hash, which every spelling of that mailbox asks for."""
    gmail = BANNED.model_copy(update={"email": "Sam.Lee+work@gmail.com"})
    await ban_record.keep_ban_record(gmail, dry_run=False)
    written = records.create.await_args.kwargs["data"]["emailHash"]
    records.find_first.return_value = object()

    for spelling in ("samlee@gmail.com", "s.a.m.lee@googlemail.com"):
        await ban_record.is_banned_address(spelling)
        assert (
            written in records.find_first.await_args.kwargs["where"]["emailHash"]["in"]
        )


@pytest.mark.asyncio
async def test_a_record_written_before_the_rule_still_matches(records):
    """An old record holds the hash of the address as it was banned, trimmed
    and lower-cased; that exact address still finds it."""
    old = hmac.new(
        b"test-ban-record-key", b"sam+old@example.com", hashlib.sha256
    ).hexdigest()
    records.find_first.return_value = object()

    assert await ban_record.is_banned_address("Sam+Old@example.com")
    assert old in records.find_first.await_args.kwargs["where"]["emailHash"]["in"]


@pytest.mark.asyncio
async def test_a_different_mailbox_asks_for_no_banned_hash(records):
    """Outside Gmail a dot makes another mailbox: its lookup never asks for
    the banned address's hash."""
    await ban_record.is_banned_address("s.am@example.com")

    digests = records.find_first.await_args.kwargs["where"]["emailHash"]["in"]
    assert ban_record.email_hash(EMAIL) not in digests


@pytest.mark.asyncio
async def test_records_past_their_date_are_purged(records):
    now = datetime(2026, 11, 2, tzinfo=UTC)

    assert await ban_record.purge_expired_ban_records(now=now) == 3
    assert records.delete_many.await_args.kwargs["where"] == {
        "deleteAfter": {"lte": now}
    }
