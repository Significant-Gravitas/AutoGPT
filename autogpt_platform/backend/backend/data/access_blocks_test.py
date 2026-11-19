import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.enums import AuthAccessBlockKind
from prisma.models import AuthAccessBlock

from backend.data.access_blocks import (
    create_access_block,
    delete_access_block,
    list_active_access_blocks,
    normalize_email_domain,
    normalize_ip,
)
from backend.data.db import query_raw_with_schema
from backend.util.exceptions import ConflictError, NotFoundError
from backend.util.test import SpinTestServer


# The frontend compares a request's IP, as Better Auth's getIp() normalises it,
# with the stored value verbatim, so these must produce that same form.
@pytest.mark.parametrize(
    "raw, expected",
    [
        (" 203.0.113.7 ", "203.0.113.7"),
        ("::ffff:203.0.113.7", "203.0.113.7"),
        ("2001:DB8:1:2:3:4:5:6", "2001:0db8:0001:0002:0000:0000:0000:0000"),
        ("2001:db8::1", "2001:0db8:0000:0000:0000:0000:0000:0000"),
    ],
)
def test_normalize_ip_matches_better_auths_session_ip(raw: str, expected: str):
    assert normalize_ip(raw) == expected


@pytest.mark.parametrize("raw", ["", "203.0.113", "203.0.113.0/24", "example.com"])
def test_normalize_ip_refuses_what_is_not_one_address(raw: str):
    with pytest.raises(ValueError):
        normalize_ip(raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Example.COM", "example.com"),
        ("@example.com", "example.com"),
        ("mail.example.co.uk.", "mail.example.co.uk"),
        ("example.co.uk", "example.co.uk"),
        ("example.uk", "example.uk"),
    ],
)
def test_normalize_email_domain(raw: str, expected: str):
    assert normalize_email_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "gmail.com",
        "@Outlook.com",
        "t-online.de",
        "outlook.co.uk",
        "co.uk",
        "com.au",
        "nhs.uk",
        "police.uk",
        "github.io",
        "user@example.com",
        "example",
        "-x.example",
    ],
)
def test_normalize_email_domain_refuses_public_providers_and_non_domains(raw: str):
    with pytest.raises(ValueError):
        normalize_email_domain(raw)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_block_in_force_conflicts_and_a_lapsed_one_is_replaced(
    server: SpinTestServer,
):
    domain = f"{uuid4().hex[:8]}.example"
    try:
        first = await create_access_block(
            kind=AuthAccessBlockKind.EMAIL_DOMAIN,
            value=domain.upper(),
            reason="first",
            expires_at=None,
            created_by_id="admin-1",
        )
        with pytest.raises(ConflictError, match=f"{domain} is already blocked"):
            await create_access_block(
                kind=AuthAccessBlockKind.EMAIL_DOMAIN,
                value=domain,
                reason="again",
                expires_at=None,
                created_by_id="admin-2",
            )
        await AuthAccessBlock.prisma().update(
            where={"id": first.id},
            data={"expiresAt": datetime.now(timezone.utc) - timedelta(minutes=1)},
        )
        listed = [b.id for b in (await list_active_access_blocks(limit=200)).blocks]
        assert first.id not in listed

        second = await create_access_block(
            kind=AuthAccessBlockKind.EMAIL_DOMAIN,
            value=domain,
            reason="second",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            created_by_id="admin-2",
        )

        assert second.id == first.id
        assert (second.value, second.reason, second.created_by_id) == (
            domain,
            "second",
            "admin-2",
        )
        listed = [b.id for b in (await list_active_access_blocks(limit=200)).blocks]
        assert second.id in listed
    finally:
        await AuthAccessBlock.prisma().delete_many(where={"value": domain})


@pytest.mark.asyncio(loop_scope="session")
async def test_create_refuses_a_blank_reason_and_a_past_expiry(
    server: SpinTestServer,
):
    with pytest.raises(ValueError):
        await create_access_block(
            kind=AuthAccessBlockKind.IP,
            value="198.51.100.78",
            reason="   ",
            expires_at=None,
            created_by_id="admin-1",
        )
    with pytest.raises(ValueError):
        await create_access_block(
            kind=AuthAccessBlockKind.IP,
            value="198.51.100.78",
            reason="past",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            created_by_id="admin-1",
        )
    assert await AuthAccessBlock.prisma().count(where={"value": "198.51.100.78"}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_an_unknown_block_is_not_found(server: SpinTestServer):
    with pytest.raises(NotFoundError):
        await delete_access_block(str(uuid4()))


@pytest.fixture
async def seeded_blocks():
    domain = f"{uuid4().hex[:8]}.example"
    now = datetime.now(timezone.utc)
    rows = [
        ("IP", "198.51.100.81", None),
        ("IP", "198.51.100.82", now - timedelta(minutes=1)),
        ("EMAIL_DOMAIN", domain, now + timedelta(days=1)),
    ]
    for kind, value, expires_at in rows:
        await AuthAccessBlock.prisma().create(
            data={
                "kind": AuthAccessBlockKind(kind),
                "value": value,
                "reason": "test",
                "expiresAt": expires_at,
            }
        )
    yield domain
    await AuthAccessBlock.prisma().delete_many(
        where={"value": {"in": [value for _, value, _ in rows]}}
    )


# The frontend's sign-up and sign-in hooks call this function
# (frontend/src/lib/auth/access-blocks.ts) with the email domain and its parents.
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "ip, domains, expected",
    [
        ("198.51.100.81", [], True),
        ("192.0.2.1", ["{domain}"], True),
        ("192.0.2.1", ["mail.{domain}", "{domain}"], True),
        ("198.51.100.82", [], False),
        ("192.0.2.1", ["not{domain}"], False),
    ],
)
async def test_the_hooks_block_check(
    server: SpinTestServer,
    seeded_blocks: str,
    ip: str,
    domains: list[str],
    expected: bool,
):
    rows = await query_raw_with_schema(
        'SELECT {schema_prefix}"is_auth_access_blocked"($1, $2::text[]) AS blocked',
        ip,
        [d.replace("{domain}", seeded_blocks) for d in domains],
    )

    assert rows[0]["blocked"] is expected


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_blocks_on_one_value_leave_exactly_one(
    server: SpinTestServer,
):
    domain = f"{uuid4().hex[:8]}.example"
    try:
        results = await asyncio.gather(
            *[
                create_access_block(
                    kind=AuthAccessBlockKind.EMAIL_DOMAIN,
                    value=domain,
                    reason=f"admin {i}",
                    expires_at=None,
                    created_by_id=f"admin-{i}",
                )
                for i in range(5)
            ],
            return_exceptions=True,
        )

        won = [r for r in results if not isinstance(r, BaseException)]
        assert len(won) == 1
        assert all(isinstance(r, ConflictError) for r in results if r not in won)
        stored = await AuthAccessBlock.prisma().find_many(where={"value": domain})
        assert [(b.reason, b.createdById) for b in stored] == [
            (won[0].reason, won[0].created_by_id)
        ]
    finally:
        await AuthAccessBlock.prisma().delete_many(where={"value": domain})
