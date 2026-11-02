"""The ban record: what is kept of an account erased while banned.

Deleting their account is a banned person's right, and the erasure takes the
auth identity, and its ban with it. So that the address can't simply sign up
again, the erasure first keeps an `AccountBanRecord`: a keyed hash of the
address, the ban's category (never the admin's free-text reason) and the
dates. Sign-up asks `is_banned_address` (through
`api/features/auth_email/routes.py`) and refuses a match.

A record lasts as long as the ban would have, and 3 years at most; the sweep
purges the ones past their date (`purge_expired_ban_records`). The hash is
keyed with a key derived from the backend's `ENCRYPTION_KEY`, which never
reaches the database, so the table alone does not tell whose addresses it
holds.
"""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

from prisma.enums import AccountBanCategory
from prisma.models import AccountBanRecord

from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.util.settings import Settings

BAN_RECORD_RETENTION = timedelta(days=3 * 365)
_KEY_LABEL = b"account-ban-record/v1"

settings = Settings()


async def keep_ban_record(subject: Subject, *, dry_run: bool) -> StepReport:
    """Record a ban still in force before the auth identity goes. Idempotent:
    a rerun writes the same record again."""
    now = datetime.now(UTC)
    if not _ban_in_force(subject, now) or not subject.email:
        return StepReport(
            name="ban_record",
            status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
            counts={"ban_records": 0},
        )
    digest = email_hash(subject.email)
    if digest is None:
        return StepReport(
            name="ban_record",
            status=StepStatus.SKIPPED,
            note="no ENCRYPTION_KEY on this service to key the hash with",
        )
    if not dry_run:
        delete_after = _delete_after(subject, now)
        await AccountBanRecord.prisma().upsert(
            where={"emailHash": digest},
            data={
                "create": {
                    "emailHash": digest,
                    "category": AccountBanCategory.ADMIN,
                    "recordedAt": now,
                    "deleteAfter": delete_after,
                },
                "update": {"deleteAfter": delete_after},
            },
        )
    return StepReport(
        name="ban_record",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"ban_records": 1},
    )


async def is_banned_address(email: str, *, now: datetime | None = None) -> bool:
    """Whether an erased, banned account had this address and its record has
    not run out. False on a service with no key to hash with."""
    digest = email_hash(email)
    if digest is None:
        return False
    record = await AccountBanRecord.prisma().find_first(
        where={"emailHash": digest, "deleteAfter": {"gt": now or datetime.now(UTC)}}
    )
    return record is not None


async def purge_expired_ban_records(*, now: datetime | None = None) -> int:
    """Delete the records past their date; how many went."""
    return await AccountBanRecord.prisma().delete_many(
        where={"deleteAfter": {"lte": now or datetime.now(UTC)}}
    )


def email_hash(email: str) -> str | None:
    """The keyed hash of an address, alike in any case and without the
    surrounding spaces; None without a key."""
    secret = settings.secrets.encryption_key
    if not secret:
        return None
    key = hmac.new(secret.encode(), _KEY_LABEL, hashlib.sha256).digest()
    address = email.strip().lower().encode()
    return hmac.new(key, address, hashlib.sha256).hexdigest()


def _ban_in_force(subject: Subject, now: datetime) -> bool:
    return subject.banned and (subject.ban_expires is None or subject.ban_expires > now)


def _delete_after(subject: Subject, now: datetime) -> datetime:
    longest = now + BAN_RECORD_RETENTION
    if subject.ban_expires is None:
        return longest
    return min(subject.ban_expires, longest)
