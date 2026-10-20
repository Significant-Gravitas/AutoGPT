"""IPs and email domains refused at sign-up (and, for an IP, at sign-in).

Admins write them here; the frontend's Better Auth hooks enforce them
(`frontend/src/lib/auth/access-blocks.ts`), so the two must agree on how a
value is normalised.
"""

import ipaddress
import re
from datetime import datetime, timezone

from prisma.enums import AuthAccessBlockKind
from prisma.models import AuthAccessBlock, AuthUser
from prisma.types import AuthAccessBlockWhereInput
from pydantic import BaseModel

from backend.util.exceptions import ConflictError, NotFoundError

# Better Auth stores an IPv6 client by its /64 (`advanced.ipAddress.ipv6Subnet`
# defaults to 64), so a block has to name the same network to match.
IPV6_SUBNET_PREFIX = 64

MAX_REASON_LENGTH = 500
DEFAULT_PAGE_SIZE = 50

# Blocking one of these would refuse a large share of real sign-ups.
PUBLIC_EMAIL_DOMAINS = frozenset(
    {
        "126.com",
        "163.com",
        "aol.com",
        "fastmail.com",
        "gmail.com",
        "gmx.com",
        "gmx.de",
        "gmx.net",
        "googlemail.com",
        "hey.com",
        "hotmail.com",
        "icloud.com",
        "live.com",
        "mac.com",
        "mail.com",
        "mail.ru",
        "me.com",
        "msn.com",
        "outlook.com",
        "pm.me",
        "proton.me",
        "protonmail.com",
        "qq.com",
        "tutanota.com",
        "web.de",
        "yahoo.com",
        "yandex.com",
        "yandex.ru",
        "ymail.com",
        "zoho.com",
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
    """The blocks in force on any of these IPs or on this exact domain."""
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
                            "value": email_domain,
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

    existing = await AuthAccessBlock.prisma().find_first(
        where={"AND": [{"kind": kind, "value": normalized}, _active_where()]}
    )
    if existing is not None:
        (block,) = await _to_access_blocks([existing])
        by = f", by {block.created_by_email}" if block.created_by_email else ""
        raise ConflictError(
            f"{normalized} is already blocked{by} on "
            f"{block.created_at.strftime('%-d %b %Y')}."
        )

    fields = {
        "reason": reason,
        "expiresAt": expires_at,
        "createdById": created_by_id,
        "createdAt": datetime.now(timezone.utc),
    }
    row = await AuthAccessBlock.prisma().upsert(
        where={"kind_value": {"kind": kind, "value": normalized}},
        data={
            "create": {"kind": kind, "value": normalized, **fields},
            "update": fields,
        },
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
    return domain


def get_email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


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
