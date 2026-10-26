"""IPs and email domains refused at sign-up (and, for an IP, at sign-in).

Admins write them here; the frontend's Better Auth hooks enforce them
(`frontend/src/lib/auth/access-blocks.ts`), so the two must agree on how a
value is normalised.
"""

import functools
import ipaddress
import re
from datetime import datetime, timezone

from prisma.enums import AuthAccessBlockKind
from prisma.models import AuthAccessBlock, AuthUser
from prisma.types import AuthAccessBlockWhereInput
from publicsuffixlist import PublicSuffixList
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.exceptions import ConflictError, NotFoundError

# Better Auth stores an IPv6 client by its /64 (`advanced.ipAddress.ipv6Subnet`
# defaults to 64), so a block has to name the same network to match.
IPV6_SUBNET_PREFIX = 64

MAX_REASON_LENGTH = 500
DEFAULT_PAGE_SIZE = 50

# Consumer mail providers: blocking one would refuse a large share of real
# sign-ups, and sharing one links no two accounts.
PUBLIC_EMAIL_DOMAINS = frozenset(
    {
        "126.com",
        "139.com",
        "163.com",
        "aim.com",
        "alice.it",
        "aliyun.com",
        "aol.com",
        "att.net",
        "bellsouth.net",
        "bigpond.com",
        "bk.ru",
        "bol.com.br",
        "btinternet.com",
        "charter.net",
        "comcast.net",
        "cox.net",
        "daum.net",
        "earthlink.net",
        "email.com",
        "fastmail.com",
        "fastmail.fm",
        "foxmail.com",
        "free.fr",
        "freenet.de",
        "gmail.com",
        "gmx.at",
        "gmx.ch",
        "gmx.com",
        "gmx.de",
        "gmx.fr",
        "gmx.net",
        "googlemail.com",
        "hanmail.net",
        "hey.com",
        "hotmail.be",
        "hotmail.ca",
        "hotmail.co.jp",
        "hotmail.co.uk",
        "hotmail.com",
        "hotmail.com.au",
        "hotmail.de",
        "hotmail.es",
        "hotmail.fr",
        "hotmail.it",
        "hotmail.nl",
        "icloud.com",
        "inbox.ru",
        "interia.pl",
        "juno.com",
        "laposte.net",
        "libero.it",
        "list.ru",
        "live.be",
        "live.ca",
        "live.co.uk",
        "live.com",
        "live.com.au",
        "live.de",
        "live.fr",
        "live.ie",
        "live.it",
        "live.jp",
        "live.nl",
        "mac.com",
        "mail.com",
        "mail.de",
        "mail.ru",
        "mailbox.org",
        "me.com",
        "msn.com",
        "naver.com",
        "o2.pl",
        "onet.pl",
        "optonline.net",
        "optusnet.com.au",
        "orange.fr",
        "outlook.be",
        "outlook.co.uk",
        "outlook.com",
        "outlook.com.au",
        "outlook.de",
        "outlook.es",
        "outlook.fr",
        "outlook.ie",
        "outlook.in",
        "outlook.it",
        "outlook.jp",
        "pm.me",
        "posteo.de",
        "proton.me",
        "protonmail.ch",
        "protonmail.com",
        "qq.com",
        "rambler.ru",
        "rediffmail.com",
        "rocketmail.com",
        "rogers.com",
        "sbcglobal.net",
        "seznam.cz",
        "sfr.fr",
        "shaw.ca",
        "sina.com",
        "sky.com",
        "sohu.com",
        "t-online.de",
        "telenet.be",
        "terra.com.br",
        "tuta.io",
        "tutanota.com",
        "tutanota.de",
        "uol.com.br",
        "verizon.net",
        "virgilio.it",
        "virginmedia.com",
        "wanadoo.fr",
        "web.de",
        "windowslive.com",
        "wp.pl",
        "ya.ru",
        "yahoo.ca",
        "yahoo.co.in",
        "yahoo.co.jp",
        "yahoo.co.nz",
        "yahoo.co.uk",
        "yahoo.com",
        "yahoo.com.ar",
        "yahoo.com.au",
        "yahoo.com.br",
        "yahoo.com.hk",
        "yahoo.com.mx",
        "yahoo.com.sg",
        "yahoo.com.tw",
        "yahoo.de",
        "yahoo.es",
        "yahoo.fr",
        "yahoo.gr",
        "yahoo.ie",
        "yahoo.in",
        "yahoo.it",
        "yahoo.nl",
        "yahoo.se",
        "yandex.by",
        "yandex.com",
        "yandex.kz",
        "yandex.ru",
        "yandex.ua",
        "yeah.net",
        "ymail.com",
        "ziggo.nl",
        "zoho.com",
        "zohomail.com",
    }
)

_DOMAIN_PATTERN = re.compile(
    r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}$"
)


class AccessBlock(BaseModel):
    id: str
    kind: AuthAccessBlockKind
    value: str
    reason: str
    created_by_id: str | None
    created_by_email: str | None
    created_at: datetime
    expires_at: datetime | None


class AccessBlockPage(BaseModel):
    blocks: list[AccessBlock]
    total: int


async def list_active_access_blocks(
    limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
) -> AccessBlockPage:
    """The blocks in force, newest first. A lapsed block stays in the table
    until it is replaced or lifted, but is neither listed nor enforced."""
    where = _active_where()
    rows = await AuthAccessBlock.prisma().find_many(
        where=where, order={"createdAt": "desc"}, take=limit, skip=offset
    )
    total = await AuthAccessBlock.prisma().count(where=where)
    return AccessBlockPage(blocks=await _to_access_blocks(rows), total=total)


async def find_active_access_blocks(
    ip_addresses: list[str], email_domain: str
) -> list[AccessBlock]:
    """The blocks in force on any of these IPs, or on this domain or a parent
    of it, as sign-up checks them."""
    rows = await AuthAccessBlock.prisma().find_many(
        where={
            "AND": [
                _active_where(),
                {
                    "OR": [
                        {
                            "kind": AuthAccessBlockKind.IP,
                            "value": {"in": ip_addresses},
                        },
                        {
                            "kind": AuthAccessBlockKind.EMAIL_DOMAIN,
                            "value": {"in": email_domain_and_parents(email_domain)},
                        },
                    ]
                },
            ]
        },
        order={"createdAt": "desc"},
    )
    return await _to_access_blocks(rows)


async def create_access_block(
    *,
    kind: AuthAccessBlockKind,
    value: str,
    reason: str,
    expires_at: datetime | None,
    created_by_id: str,
) -> AccessBlock:
    """Block an IP or an email domain. A lapsed block on the same value is
    replaced; one still in force is a conflict."""
    normalized = (
        normalize_ip(value)
        if kind == AuthAccessBlockKind.IP
        else normalize_email_domain(value)
    )
    reason = reason.strip()
    if not reason:
        raise ValueError("Give a reason for the block.")
    if len(reason) > MAX_REASON_LENGTH:
        raise ValueError(f"Keep the reason under {MAX_REASON_LENGTH} characters.")
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        raise ValueError("The block would expire before it starts.")

    # One statement, so two admins blocking the same value at once can't both
    # pass a check and then overwrite each other: only a lapsed row is replaced.
    rows = await query_raw_with_schema(
        """
        INSERT INTO {schema_prefix}"UserAuthAccessBlock"
            ("kind", "value", "reason", "expiresAt", "createdById")
        VALUES ($1::{schema_prefix}"AuthAccessBlockKind", $2, $3, $4::timestamptz, $5)
        ON CONFLICT ("kind", "value") DO UPDATE SET
            "reason" = EXCLUDED."reason",
            "expiresAt" = EXCLUDED."expiresAt",
            "createdById" = EXCLUDED."createdById",
            "createdAt" = CURRENT_TIMESTAMP
        WHERE "UserAuthAccessBlock"."expiresAt" IS NOT NULL
          AND "UserAuthAccessBlock"."expiresAt" <= CURRENT_TIMESTAMP
        RETURNING "id"
        """,
        kind.value,
        normalized,
        reason,
        expires_at.isoformat() if expires_at else None,
        created_by_id,
    )
    if not rows:
        raise await _conflict(kind, normalized)
    row = await AuthAccessBlock.prisma().find_unique_or_raise(
        where={"id": rows[0]["id"]}
    )
    return (await _to_access_blocks([row]))[0]


async def delete_access_block(block_id: str) -> None:
    deleted = await AuthAccessBlock.prisma().delete(where={"id": block_id})
    if deleted is None:
        raise NotFoundError(f"Access block {block_id} not found")


def normalize_ip(raw: str) -> str:
    """An IP in the form Better Auth records a session's: IPv4 as is, an
    IPv4-mapped IPv6 address as its IPv4, any other IPv6 address as its /64
    network written out in full."""
    try:
        address = ipaddress.ip_address(raw.strip())
    except ValueError:
        raise ValueError(f"{raw.strip()!r} is not an IP address.") from None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        network = ipaddress.IPv6Network(
            (address, IPV6_SUBNET_PREFIX), strict=False
        ).network_address
        return network.exploded
    return str(address)


def normalize_email_domain(raw: str) -> str:
    """A bare lowercase domain; refuses the big public email providers."""
    domain = raw.strip().lower().lstrip("@").rstrip(".")
    if not _DOMAIN_PATTERN.match(domain):
        raise ValueError(f"{raw.strip()!r} is not an email domain.")
    if domain in PUBLIC_EMAIL_DOMAINS:
        raise ValueError(
            f"{domain} is a public email provider; ban its accounts one by one instead."
        )
    # A public suffix ("co.uk", "nhs.uk", "github.io"): blocking one would refuse
    # every domain registered under it, since sign-up checks each parent domain.
    if _public_suffixes().is_public(domain):
        raise ValueError(f"{domain} is a public suffix, not a domain someone owns.")
    return domain


# The Public Suffix List ships inside the package, so it refreshes with
# dependency bumps; built on first use rather than at every service's import.
@functools.cache
def _public_suffixes() -> PublicSuffixList:
    return PublicSuffixList()


def get_email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


def email_domain_and_parents(domain: str) -> list[str]:
    """Most specific first, never the bare TLD: the domains a sign-up is
    checked against (getDomainAndParents in frontend lib/auth/access-blocks.ts)."""
    labels = domain.split(".")
    return [".".join(labels[i:]) for i in range(len(labels) - 1)]


async def _conflict(kind: AuthAccessBlockKind, value: str) -> ConflictError:
    existing = await AuthAccessBlock.prisma().find_first(
        where={"AND": [{"kind": kind, "value": value}, _active_where()]}
    )
    if existing is None:
        return ConflictError(f"{value} is already blocked.")
    (block,) = await _to_access_blocks([existing])
    by = f", by {block.created_by_email}" if block.created_by_email else ""
    return ConflictError(
        f"{value} is already blocked{by} on {block.created_at.strftime('%-d %b %Y')}."
    )


async def _to_access_blocks(rows: list[AuthAccessBlock]) -> list[AccessBlock]:
    creator_ids = list({row.createdById for row in rows if row.createdById})
    creators = (
        await AuthUser.prisma().find_many(where={"id": {"in": creator_ids}})
        if creator_ids
        else []
    )
    emails = {creator.id: creator.email for creator in creators}
    return [
        AccessBlock(
            id=row.id,
            kind=row.kind,
            value=row.value,
            reason=row.reason,
            created_by_id=row.createdById,
            created_by_email=emails.get(row.createdById or ""),
            created_at=row.createdAt,
            expires_at=row.expiresAt,
        )
        for row in rows
    ]


def _active_where() -> AuthAccessBlockWhereInput:
    return {
        "OR": [
            {"expiresAt": None},
            {"expiresAt": {"gt": datetime.now(timezone.utc)}},
        ]
    }
