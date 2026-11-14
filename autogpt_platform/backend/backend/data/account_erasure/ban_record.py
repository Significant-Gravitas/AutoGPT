"""The ban record: what is kept of an account erased while banned.

Deleting their account is a banned person's right, and the erasure takes the
auth identity, and its ban with it. So that the address can't simply sign up
again, the erasure first keeps an `AccountBanRecord`: a keyed hash of the
address, the ban's category (never the admin's free-text reason) and the
dates. Sign-up asks `is_banned_address` (through
`api/features/auth_email/routes.py`) and refuses a match.

The hash is taken of the address's canonical form (`canonical_address`), so
the other spellings of one mailbox match its record: any case, `+tag` dropped
on every domain, and for Gmail the dots dropped and `googlemail.com` read as
`gmail.com`. Records written before that rule hold the hash of the trimmed,
lower-cased address; the lookup matches those too, and they stay as they are.
Two accounts on spellings of one mailbox, both erased while banned, share one
record: the second only moves its end out, to the longer of the two bans.

A record lasts as long as the ban would have, and 3 years at most from when
it was first kept; the sweep purges the ones past their date
(`purge_expired_ban_records`). The hash is keyed with `BAN_RECORD_HASH_KEY`,
a key of its own that never reaches the database, so the table alone does not
tell whose addresses it holds. It is never rotated: the records kept under
the old key would stop matching.
"""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

from prisma.enums import AccountBanCategory
from prisma.errors import UniqueViolationError
from prisma.models import AccountBanRecord

from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.util.settings import Settings

BAN_RECORD_RETENTION = timedelta(days=3 * 365)

settings = Settings()


async def keep_ban_record(subject: Subject, *, dry_run: bool) -> StepReport:
    """Record a ban still in force before the auth identity goes. Idempotent:
    a rerun keeps the record the first run wrote, dates included, so retries
    never push its end out. A longer ban on another spelling of the mailbox
    moves it out (`_outlast`)."""
    now = datetime.now(UTC)
    if not _ban_in_force(subject, now) or not subject.email:
        return StepReport(
            name="ban_record",
            status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
            counts={"ban_records": 0},
        )
    digest = email_hash(subject.email)
    if digest is None:
        # Fails rather than skips: once the identity goes, the ban is lost.
        return StepReport(
            name="ban_record",
            status=StepStatus.FAILED,
            note="no BAN_RECORD_HASH_KEY on this service to key the hash with; "
            "set it and run again",
        )
    if not dry_run:
        await _create_once(digest, now, subject.ban_expires)
    return StepReport(
        name="ban_record",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"ban_records": 1},
    )


async def is_banned_address(email: str, *, now: datetime | None = None) -> bool:
    """Whether an erased, banned account had this address, or another
    spelling of its mailbox, and its record has not run out. False on a
    service with no key to hash with."""
    digests = email_hashes(email)
    if not digests:
        return False
    record = await AccountBanRecord.prisma().find_first(
        where={
            "emailHash": {"in": digests},
            "deleteAfter": {"gt": now or datetime.now(UTC)},
        }
    )
    return record is not None


async def purge_expired_ban_records(*, now: datetime | None = None) -> int:
    """Delete the records past their date; how many went."""
    return await AccountBanRecord.prisma().delete_many(
        where={"deleteAfter": {"lte": now or datetime.now(UTC)}}
    )


async def _create_once(
    digest: str, now: datetime, ban_expires: datetime | None
) -> None:
    try:
        await _create(digest, now, _delete_after(ban_expires, now))
    except UniqueViolationError:
        # Kept from an earlier run, or for another spelling of this mailbox
        # also erased while banned. One that ran out but was not purged yet
        # starts over, for this ban.
        restarted = await AccountBanRecord.prisma().update_many(
            where={"emailHash": digest, "deleteAfter": {"lte": now}},
            data={"recordedAt": now, "deleteAfter": _delete_after(ban_expires, now)},
        )
        if restarted:
            return
        kept = await AccountBanRecord.prisma().find_first(
            where={"emailHash": digest, "deleteAfter": {"gt": now}}
        )
        if kept is not None and (
            await _outlast(digest, kept.recordedAt, ban_expires)
            or await _in_force(digest, now)
        ):
            return
        # The purge took the record in between: create it again, and
        # a second conflict fails the step rather than report a record that
        # is not there.
        await _create(digest, now, _delete_after(ban_expires, now))


async def _outlast(
    digest: str, recorded_at: datetime, ban_expires: datetime | None
) -> bool:
    """Move the kept record's end out to this ban's, counted from when the
    record was first kept, so a longer ban on another spelling is not lost
    and a rerun of the same ban, which ends where the record does, changes
    nothing. Never moves an end in. Whether it moved the end."""
    delete_after = _delete_after(ban_expires, recorded_at)
    moved = await AccountBanRecord.prisma().update_many(
        where={"emailHash": digest, "deleteAfter": {"lt": delete_after}},
        data={"deleteAfter": delete_after},
    )
    return moved > 0


async def _in_force(digest: str, now: datetime) -> bool:
    record = await AccountBanRecord.prisma().find_first(
        where={"emailHash": digest, "deleteAfter": {"gt": now}}
    )
    return record is not None


async def _create(digest: str, now: datetime, delete_after: datetime) -> None:
    await AccountBanRecord.prisma().create(
        data={
            "emailHash": digest,
            "category": AccountBanCategory.ADMIN,
            "recordedAt": now,
            "deleteAfter": delete_after,
        }
    )


def email_hash(email: str) -> str | None:
    """The keyed hash of an address's canonical form, the one a new record
    holds; None without a key."""
    return _keyed_hash(canonical_address(email))


def email_hashes(email: str) -> list[str]:
    """Every hash a record of this address may hold: the canonical one, and
    the trimmed, lower-cased one records written before the canonical form
    hold. Empty without a key."""
    digests = [_keyed_hash(canonical_address(email)), _keyed_hash(_trimmed(email))]
    return list(dict.fromkeys(d for d in digests if d is not None))


def canonical_address(email: str) -> str:
    """One spelling for the addresses that reach the same mailbox: trimmed,
    lower-cased, the `+tag` dropped, and on Gmail the dots too."""
    address = _trimmed(email)
    local, at, domain = address.rpartition("@")
    if not at:
        return address
    local = local.split("+", 1)[0] or local
    if domain in _GMAIL_DOMAINS:
        local = local.replace(".", "") or local
        domain = "gmail.com"
    return f"{local}@{domain}"


_GMAIL_DOMAINS = frozenset({"gmail.com", "googlemail.com"})


def _trimmed(email: str) -> str:
    return email.strip().lower()


def _keyed_hash(address: str) -> str | None:
    key = settings.secrets.ban_record_hash_key
    if not key:
        return None
    return hmac.new(key.encode(), address.encode(), hashlib.sha256).hexdigest()


def _ban_in_force(subject: Subject, now: datetime) -> bool:
    return subject.banned and (subject.ban_expires is None or subject.ban_expires > now)


def _delete_after(ban_expires: datetime | None, since: datetime) -> datetime:
    longest = since + BAN_RECORD_RETENTION
    if ban_expires is None:
        return longest
    return min(ban_expires, longest)
