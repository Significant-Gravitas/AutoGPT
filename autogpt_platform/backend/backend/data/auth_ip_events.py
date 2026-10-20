"""Where each account signed up and signed in from, and which other accounts
share those IPs or its email domain.

The frontend's Better Auth hooks write the log (`frontend/src/lib/auth/ip-log.ts`);
the scheduler deletes rows older than `auth_ip_event_retention_days`.
"""

import ipaddress
import logging
from datetime import datetime, timedelta, timezone

from prisma.enums import AuthAccessBlockKind, AuthIpEventKind
from prisma.models import AuthIpEvent, AuthUser
from pydantic import BaseModel

from backend.data.access_blocks import (
    IPV6_SUBNET_PREFIX,
    PUBLIC_EMAIL_DOMAINS,
    AccessBlock,
    find_active_access_blocks,
    get_email_domain,
)
from backend.data.db import query_raw_with_schema
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

ACCOUNT_IP_LIMIT = 50
EVENTS_PER_IP = 20
RELATED_ACCOUNT_LIMIT = 200


class IpEventEntry(BaseModel):
    kind: AuthIpEventKind
    user_agent: str | None
    created_at: datetime


class AccountIp(BaseModel):
    ip_address: str
    display_address: str
    is_ipv6_prefix: bool
    has_sign_up: bool
    sign_in_count: int
    first_seen: datetime
    last_seen: datetime
    last_user_agent: str | None
    other_account_count: int
    block: AccessBlock | None
    events: list[IpEventEntry]
    event_count: int


class EmailDomainInfo(BaseModel):
    value: str
    is_public_provider: bool
    other_account_count: int
    block: AccessBlock | None


class RelatedAccount(BaseModel):
    id: str
    email: str
    name: str
    role: str | None
    created_at: datetime
    banned: bool
    ban_expires: datetime | None
    has_profile: bool
    via_ips: list[str]
    via_domain: bool


class AccountIpReport(BaseModel):
    retention_days: int
    ips: list[AccountIp]
    domain: EmailDomainInfo
    related: list[RelatedAccount]
    related_total: int


class _IpRow(BaseModel):
    ip_address: str
    has_sign_up: bool
    sign_in_count: int
    event_count: int
    first_seen: datetime
    last_seen: datetime
    last_user_agent: str | None
    other_account_count: int


class _EventRow(BaseModel):
    ip_address: str
    kind: AuthIpEventKind
    user_agent: str | None
    created_at: datetime


class _RelatedRow(BaseModel):
    id: str
    email: str
    name: str
    role: str | None
    created_at: datetime
    banned: bool | None
    ban_expires: datetime | None
    has_profile: bool
    via_ips: list[str]
    via_domain: bool
    total: int


async def get_account_ip_report(user_id: str, retention_days: int) -> AccountIpReport:
    """Read from the auth identity table, so an account with no platform
    `User` row (one that never finished signing up) still shows. A public
    mail provider's domain links no accounts: that would be the user base."""
    user = await AuthUser.prisma().find_unique(where={"id": user_id})
    if user is None:
        raise NotFoundError(f"User {user_id} not found")

    ip_rows = await _summarise_account_ips(user_id)
    ip_addresses = [row.ip_address for row in ip_rows]
    events = await _latest_events_per_ip(user_id, ip_addresses)
    email_domain = get_email_domain(user.email)
    is_public = email_domain in PUBLIC_EMAIL_DOMAINS
    linked_domain = None if is_public else email_domain
    related_rows = await _find_related_accounts(user_id, ip_addresses, linked_domain)
    domain_count = await _count_domain_accounts(user_id, linked_domain)
    blocks = await find_active_access_blocks(ip_addresses, email_domain)
    ip_blocks = {b.value: b for b in blocks if b.kind == AuthAccessBlockKind.IP}
    domain_block = next(
        (b for b in blocks if b.kind == AuthAccessBlockKind.EMAIL_DOMAIN), None
    )

    return AccountIpReport(
        retention_days=retention_days,
        ips=[
            _to_account_ip(row, events.get(row.ip_address, []), ip_blocks)
            for row in ip_rows
        ],
        domain=EmailDomainInfo(
            value=email_domain,
            is_public_provider=is_public,
            other_account_count=domain_count,
            block=domain_block,
        ),
        related=[_to_related_account(row) for row in related_rows],
        related_total=related_rows[0].total if related_rows else 0,
    )


async def delete_expired_ip_events(retention_days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = await AuthIpEvent.prisma().delete_many(
        where={"createdAt": {"lt": cutoff}}
    )
    if deleted > 0:
        logger.info(f"Deleted {deleted} sign-in IP records past their retention")
    return deleted


async def _summarise_account_ips(user_id: str) -> list[_IpRow]:
    # The sign-up address first, whenever it was last used: "where was this
    # account created" is the first question.
    return await query_raw_with_schema(
        """
        WITH mine AS (
            SELECT "ipAddress",
                   bool_or("kind" = 'SIGN_UP') AS has_sign_up,
                   count(*) FILTER (WHERE "kind" = 'SIGN_IN') AS sign_in_count,
                   count(*) AS event_count,
                   min("createdAt") AS first_seen,
                   max("createdAt") AS last_seen,
                   (array_agg("userAgent" ORDER BY "createdAt" DESC))[1]
                       AS last_user_agent
            FROM {schema_prefix}"UserAuthIpEvent"
            WHERE "userId" = $1
            GROUP BY "ipAddress"
            ORDER BY has_sign_up DESC, last_seen DESC
            LIMIT $2
        )
        SELECT mine."ipAddress" AS ip_address, has_sign_up, sign_in_count,
               event_count, first_seen, last_seen, last_user_agent,
               (SELECT count(DISTINCT other."userId")
                FROM {schema_prefix}"UserAuthIpEvent" other
                WHERE other."ipAddress" = mine."ipAddress"
                  AND other."userId" <> $1) AS other_account_count
        FROM mine
        ORDER BY has_sign_up DESC, last_seen DESC
        """,
        user_id,
        ACCOUNT_IP_LIMIT,
        model=_IpRow,
    )


async def _latest_events_per_ip(
    user_id: str, ip_addresses: list[str]
) -> dict[str, list[IpEventEntry]]:
    if not ip_addresses:
        return {}
    rows = await query_raw_with_schema(
        """
        SELECT ip_address, kind, user_agent, created_at
        FROM (
            SELECT "ipAddress" AS ip_address, "kind" AS kind,
                   "userAgent" AS user_agent, "createdAt" AS created_at,
                   row_number() OVER (
                       PARTITION BY "ipAddress" ORDER BY "createdAt" DESC
                   ) AS position
            FROM {schema_prefix}"UserAuthIpEvent"
            WHERE "userId" = $1 AND "ipAddress" = ANY($2::text[])
        ) ranked
        WHERE position <= $3
        ORDER BY created_at DESC
        """,
        user_id,
        ip_addresses,
        EVENTS_PER_IP,
        model=_EventRow,
    )
    events: dict[str, list[IpEventEntry]] = {}
    for row in rows:
        events.setdefault(row.ip_address, []).append(
            IpEventEntry(
                kind=row.kind, user_agent=row.user_agent, created_at=row.created_at
            )
        )
    return events


async def _find_related_accounts(
    user_id: str, ip_addresses: list[str], email_domain: str | None
) -> list[_RelatedRow]:
    # The domain is an exact match on the part after the @, not a LIKE, so `_`
    # and `%` in a domain match only themselves. Strongest link first: same IP
    # and domain, then same IP, then same domain.
    return await query_raw_with_schema(
        """
        WITH ip_links AS (
            SELECT "userId" AS id, array_agg(DISTINCT "ipAddress") AS via_ips
            FROM {schema_prefix}"UserAuthIpEvent"
            WHERE "ipAddress" = ANY($2::text[]) AND "userId" <> $1
            GROUP BY "userId"
        ),
        domain_links AS (
            SELECT "id"
            FROM {schema_prefix}"UserAuthIdentity"
            WHERE $3::text IS NOT NULL
              AND split_part("email", '@', 2) = $3::text
              AND "id" <> $1
        ),
        related AS (
            SELECT id FROM ip_links UNION SELECT id FROM domain_links
        )
        SELECT u."id" AS id, u."email" AS email, u."name" AS name, u."role" AS role,
               u."createdAt" AS created_at, u."banned" AS banned,
               u."banExpires" AS ban_expires,
               EXISTS (SELECT 1 FROM {schema_prefix}"User" p WHERE p."id" = u."id")
                   AS has_profile,
               COALESCE(i.via_ips, ARRAY[]::text[]) AS via_ips,
               d."id" IS NOT NULL AS via_domain,
               count(*) OVER () AS total
        FROM related r
        JOIN {schema_prefix}"UserAuthIdentity" u ON u."id" = r.id
        LEFT JOIN ip_links i ON i.id = r.id
        LEFT JOIN domain_links d ON d."id" = r.id
        ORDER BY (i.id IS NOT NULL AND d."id" IS NOT NULL) DESC,
                 (i.id IS NOT NULL) DESC,
                 u."createdAt" DESC
        LIMIT $4
        """,
        user_id,
        ip_addresses,
        email_domain,
        RELATED_ACCOUNT_LIMIT,
        model=_RelatedRow,
    )


async def _count_domain_accounts(user_id: str, email_domain: str | None) -> int:
    if email_domain is None:
        return 0
    rows = await query_raw_with_schema(
        """
        SELECT count(*) AS total
        FROM {schema_prefix}"UserAuthIdentity"
        WHERE split_part("email", '@', 2) = $1 AND "id" <> $2
        """,
        email_domain,
        user_id,
    )
    return int(rows[0]["total"])


def _to_account_ip(
    row: _IpRow, events: list[IpEventEntry], ip_blocks: dict[str, AccessBlock]
) -> AccountIp:
    display = _display_address(row.ip_address)
    return AccountIp(
        ip_address=row.ip_address,
        display_address=display,
        is_ipv6_prefix=display != row.ip_address,
        has_sign_up=row.has_sign_up,
        sign_in_count=row.sign_in_count,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        last_user_agent=row.last_user_agent,
        other_account_count=row.other_account_count,
        block=ip_blocks.get(row.ip_address),
        events=events,
        event_count=row.event_count,
    )


def _display_address(stored: str) -> str:
    """`2001:0db8:…:0000` as `2001:db8::/64`; anything else as stored."""
    try:
        address = ipaddress.ip_address(stored)
    except ValueError:
        return stored
    if not isinstance(address, ipaddress.IPv6Address):
        return stored
    return str(ipaddress.IPv6Network((address, IPV6_SUBNET_PREFIX), strict=False))


def _to_related_account(row: _RelatedRow) -> RelatedAccount:
    # Better Auth clears a lapsed ban only at the account's next sign-in.
    banned = bool(row.banned) and (
        row.ban_expires is None or row.ban_expires > datetime.now(timezone.utc)
    )
    return RelatedAccount(
        id=row.id,
        email=row.email,
        name=row.name,
        role=row.role,
        created_at=row.created_at,
        banned=banned,
        ban_expires=row.ban_expires if banned else None,
        has_profile=row.has_profile,
        via_ips=sorted(row.via_ips),
        via_domain=row.via_domain,
    )
