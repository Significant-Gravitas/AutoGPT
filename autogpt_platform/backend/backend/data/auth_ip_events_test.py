from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.enums import AuthAccessBlockKind, AuthIpEventKind
from prisma.models import AuthAccessBlock, AuthIpEvent, AuthUser

from backend.data.auth_ip_events import delete_expired_ip_events, get_account_ip_report
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

RETENTION_DAYS = 90


@pytest.fixture
async def accounts():
    tag = uuid4().hex[:8]
    created: list[str] = []

    async def make(email: str, **fields) -> AuthUser:
        user = await AuthUser.prisma().create(
            data={
                "id": str(uuid4()),
                "name": email.split("@")[0],
                "email": email.replace("{tag}", tag),
                **fields,
            }
        )
        created.append(user.id)
        return user

    yield make, tag
    await AuthUser.prisma().delete_many(where={"id": {"in": created}})


async def _log(user: AuthUser, kind: AuthIpEventKind, ip: str, days_ago: float = 0):
    await AuthIpEvent.prisma().create(
        data={
            "userId": user.id,
            "kind": kind,
            "ipAddress": ip,
            "userAgent": "Mozilla/5.0 (synthetic)",
            "createdAt": datetime.now(timezone.utc) - timedelta(days=days_ago),
        }
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_report_merges_accounts_linked_by_ip_and_by_domain(
    server: SpinTestServer, accounts
):
    make, tag = accounts
    target = await make("first@{tag}-farm.example")
    ip_and_domain = await make("second@{tag}-farm.example")
    ip_only = await make("third@other-{tag}.example")
    domain_only = await make("fourth@{tag}-farm.example")
    unrelated = await make("fifth@unrelated-{tag}.example")
    await _log(target, AuthIpEventKind.SIGN_UP, "203.0.113.10", days_ago=3)
    await _log(target, AuthIpEventKind.SIGN_IN, "198.51.100.20", days_ago=1)
    await _log(target, AuthIpEventKind.SIGN_IN, "198.51.100.20")
    await _log(ip_and_domain, AuthIpEventKind.SIGN_UP, "203.0.113.10")
    await _log(ip_only, AuthIpEventKind.SIGN_IN, "198.51.100.20")
    await _log(unrelated, AuthIpEventKind.SIGN_IN, "192.0.2.30")

    report = await get_account_ip_report(target.id, RETENTION_DAYS)

    # The sign-up address leads although the other was used more recently.
    assert [
        (ip.ip_address, ip.has_sign_up, ip.sign_in_count, ip.other_account_count)
        for ip in report.ips
    ] == [("203.0.113.10", True, 0, 1), ("198.51.100.20", False, 2, 1)]
    assert len(report.ips[1].events) == 2
    assert [(a.id, a.via_ips, a.via_domain) for a in report.related] == [
        (ip_and_domain.id, ["203.0.113.10"], True),
        (ip_only.id, ["198.51.100.20"], False),
        (domain_only.id, [], True),
    ]
    assert report.related_total == 3
    assert report.domain.value == f"{tag}-farm.example"
    assert report.domain.other_account_count == 2
    assert report.retention_days == RETENTION_DAYS


@pytest.mark.asyncio(loop_scope="session")
async def test_domain_match_is_exact_so_an_underscore_is_no_wildcard(
    server: SpinTestServer, accounts
):
    make, tag = accounts
    target = await make("a@x_{tag}.example")
    lookalike = await make("b@xy{tag}.example")

    report = await get_account_ip_report(target.id, RETENTION_DAYS)

    assert lookalike.id not in {a.id for a in report.related}
    assert report.domain.other_account_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_public_mail_domain_links_no_accounts(server: SpinTestServer, accounts):
    make, tag = accounts
    target = await make("a-{tag}@gmail.com")
    mate = await make("b-{tag}@gmail.com")
    await _log(target, AuthIpEventKind.SIGN_IN, "203.0.113.60")
    await _log(mate, AuthIpEventKind.SIGN_IN, "203.0.113.60")
    await make("c-{tag}@gmail.com")

    report = await get_account_ip_report(target.id, RETENTION_DAYS)

    assert report.domain.is_public_provider
    assert report.domain.other_account_count == 0
    assert [(a.id, a.via_domain) for a in report.related] == [(mate.id, False)]


@pytest.mark.asyncio(loop_scope="session")
async def test_report_shows_bans_in_force_and_the_blocks_that_cover_the_account(
    server: SpinTestServer, accounts
):
    make, tag = accounts
    target = await make("a@{tag}.example")
    banned = await make("b@{tag}.example", banned=True)
    lapsed = await make(
        "c@{tag}.example",
        banned=True,
        banExpires=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    for user in (target, banned, lapsed):
        await _log(
            user, AuthIpEventKind.SIGN_IN, "2001:0db8:0000:0001:0000:0000:0000:0000"
        )
    block = await AuthAccessBlock.prisma().create(
        data={
            "kind": AuthAccessBlockKind.IP,
            "value": "2001:0db8:0000:0001:0000:0000:0000:0000",
            "reason": "test",
        }
    )
    try:
        report = await get_account_ip_report(target.id, RETENTION_DAYS)
    finally:
        await AuthAccessBlock.prisma().delete(where={"id": block.id})

    assert {a.id: a.banned for a in report.related} == {
        banned.id: True,
        lapsed.id: False,
    }
    (ip,) = report.ips
    assert (ip.display_address, ip.is_ipv6_prefix) == ("2001:db8:0:1::/64", True)
    assert ip.block is not None and ip.block.id == block.id


@pytest.mark.asyncio(loop_scope="session")
async def test_report_for_an_unknown_account_is_not_found(server: SpinTestServer):
    with pytest.raises(NotFoundError):
        await get_account_ip_report(str(uuid4()), RETENTION_DAYS)


@pytest.mark.asyncio(loop_scope="session")
async def test_retention_deletes_only_records_older_than_the_window(
    server: SpinTestServer, accounts
):
    make, _ = accounts
    user = await make("a@{tag}.example")
    await _log(
        user, AuthIpEventKind.SIGN_UP, "203.0.113.50", days_ago=RETENTION_DAYS + 1
    )
    await _log(
        user, AuthIpEventKind.SIGN_IN, "203.0.113.51", days_ago=RETENTION_DAYS - 1
    )

    await delete_expired_ip_events(RETENTION_DAYS)

    remaining = await AuthIpEvent.prisma().find_many(where={"userId": user.id})
    assert [e.ipAddress for e in remaining] == ["203.0.113.51"]
