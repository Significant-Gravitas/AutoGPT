"""Authentication against real stored keys, Scrypt and Redis."""

import asyncio
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from ipaddress import IPv6Address, IPv6Network
from typing import Annotated
from uuid import uuid4

import fastapi
import httpx
import pytest
import pytest_asyncio
from fastapi import Request, Security
from prisma.enums import APIKeyStatus
from prisma.models import APIKey, User
from prisma.types import APIKeyCreateInput, UserCreateInput

from backend.api.external import auth_rate_limit as rate_limit
from backend.api.external import middleware
from backend.data.auth import api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.redis_client import get_redis_async


@pytest_asyncio.fixture
async def stored_key(server):
    user = await User.prisma().create(
        data=UserCreateInput(id=str(uuid4()), email=f"{uuid4()}@example.invalid")
    )
    info, plaintext = await api_key.create_api_key("Limiter test", user.id, [])
    try:
        yield info, plaintext
    finally:
        await APIKey.prisma().delete_many(where={"userId": user.id})
        await User.prisma().delete(where={"id": user.id})


@pytest.fixture
def app():
    app = fastapi.FastAPI()

    @app.get("/auth")
    async def auth(
        auth: Annotated[APIAuthorizationInfo, Security(middleware.require_auth)]
    ):
        return {"user_id": auth.user_id}

    return app


@pytest_asyncio.fixture
async def client(stored_key, monkeypatch, mocker, app):
    await get_redis_async()
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS", 10)
    mocker.patch.object(rate_limit, "datetime").now.return_value = datetime(
        2026, 1, 1, 0, 0, 17, tzinfo=UTC
    )
    transport = httpx.ASGITransport(app, client=(random_client_ip(), 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(params=["api-key", "bearer"])
def auth_headers(request):
    def headers(plaintext):
        if request.param == "api-key":
            return {"X-API-Key": plaintext}
        return {"Authorization": f"Bearer {plaintext}"}

    return headers


async def test_successful_requests_do_not_exhaust_budget(
    client, stored_key, monkeypatch, auth_headers
):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 3)
    info, plaintext = stored_key
    for _ in range(8):
        response = await client.get("/auth", headers=auth_headers(plaintext))
        assert response.status_code == 200
        assert response.json() == {"user_id": info.user_id}


async def test_all_keys_with_same_head_and_tail_authenticate(stored_key):
    info, plaintext = stored_key
    values = [(info.id, plaintext)]
    for _ in range(5):
        value = plaintext[:8] + secrets.token_urlsafe(24) + plaintext[-8:]
        hashed, salt = api_key.keysmith.hash_key(value)
        collision = await APIKey.prisma().create(
            data=APIKeyCreateInput(
                id=str(uuid4()),
                name="Head and tail collision",
                head=value[:8],
                tail=value[-8:],
                hash=hashed,
                salt=salt,
                userId=info.user_id,
                permissions=[],
            )
        )
        values.append((collision.id, value))
    for key_id, value in values:
        authenticated = await api_key.validate_api_key(value)
        assert authenticated is not None
        assert authenticated.id == key_id


async def test_missing_peer_does_not_create_global_bucket(stored_key, monkeypatch):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 1)
    request = Request({"type": "http", "headers": [], "client": None})
    for _ in range(3):
        budget = await rate_limit.enforce_api_key_validate_rate_limit(
            request, stored_key[1]
        )
        assert budget is None


async def test_deadline_bounds_the_whole_check(monkeypatch):
    async def stalled_connection():
        await asyncio.sleep(10)

    monkeypatch.setattr(rate_limit, "get_redis_async", stalled_connection)
    request = Request({"type": "http", "headers": [], "client": ("192.0.2.1", 1234)})
    budget = await asyncio.wait_for(
        rate_limit.enforce_api_key_validate_rate_limit(
            request, "agpt_abcMIDDLESAMETAIL"
        ),
        timeout=5,
    )
    assert budget is None


async def test_concurrent_invalid_burst_stops_real_scrypt(
    client, stored_key, monkeypatch, mocker, auth_headers
):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 3)
    verify = mocker.spy(api_key.keysmith, "verify_key")
    plaintext = stored_key[1]
    bad = plaintext[:8] + ("X" if plaintext[8] != "X" else "Y") + plaintext[9:]
    responses = await asyncio.gather(
        *(client.get("/auth", headers=auth_headers(bad)) for _ in range(12))
    )
    assert [r.status_code for r in responses].count(401) == 3
    limited = [r for r in responses if r.status_code == 429]
    assert len(limited) == 9
    assert {r.headers["Retry-After"] for r in limited} == {"43"}
    assert verify.call_count == 3


async def test_wrong_tail_and_malformed_keys_skip_scrypt(
    client, stored_key, mocker, auth_headers
):
    verify = mocker.spy(api_key.keysmith, "verify_key")
    redis_access = mocker.spy(rate_limit, "get_redis_async")
    plaintext = stored_key[1]
    different_tail = plaintext[:-1] + ("X" if plaintext[-1] != "X" else "Y")
    for value in ("garbage", "agpt_", "agpt_!abmiddle", different_tail):
        assert (
            await client.get("/auth", headers=auth_headers(value))
        ).status_code == 401
    assert verify.call_count == 0
    redis_access.assert_not_called()


async def test_concurrent_success_refunds_only_its_own_reservation(
    client, stored_key, monkeypatch, mocker, auth_headers
):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 8)
    plaintext = stored_key[1]
    bad = plaintext[:8] + ("X" if plaintext[8] != "X" else "Y") + plaintext[9:]
    ready, release = asyncio.Event(), asyncio.Event()
    entered = 0
    validate = api_key._verify_api_key_candidates

    async def wait_then_validate(value, candidates):
        nonlocal entered
        entered += 1
        if entered == 8:
            ready.set()
        await release.wait()
        return await validate(value, candidates)

    monkeypatch.setattr(api_key, "_verify_api_key_candidates", wait_then_validate)
    reserve = mocker.spy(rate_limit, "_reserve")
    verify = mocker.spy(api_key.keysmith, "verify_key")
    pending = asyncio.gather(
        *(
            client.get("/auth", headers=auth_headers(value))
            for value in [plaintext] * 5 + [bad] * 3
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=15)
        budget = reserve.call_args.args[0]
        redis = await get_redis_async()
        assert await redis.get(budget.head_key) == "8"
        assert (
            await client.get("/auth", headers=auth_headers(plaintext))
        ).status_code == 429
        assert verify.call_count == 0
    finally:
        release.set()
        responses = await pending
    assert [r.status_code for r in responses].count(200) == 5
    assert [r.status_code for r in responses].count(401) == 3
    assert verify.call_count == 8
    # Three failed validations and the denied in-flight request remain charged.
    assert await redis.get(budget.head_key) == "4"
    assert (
        await client.get("/auth", headers=auth_headers(plaintext))
    ).status_code == 200
    assert await redis.get(budget.head_key) == "4"


@pytest.mark.parametrize("state", [APIKeyStatus.REVOKED, APIKeyStatus.SUSPENDED])
async def test_changed_key_status_is_observed_on_next_request(
    client, stored_key, state, mocker, auth_headers
):
    info, plaintext = stored_key
    headers = auth_headers(plaintext)
    assert (await client.get("/auth", headers=headers)).status_code == 200
    await APIKey.prisma().update(where={"id": info.id}, data={"status": state})
    verify = mocker.spy(api_key.keysmith, "verify_key")
    assert (await client.get("/auth", headers=headers)).status_code == 401
    assert verify.call_count == 0


async def test_legacy_key_still_authenticates_and_migrates(
    client, stored_key, auth_headers
):
    info, plaintext = stored_key
    await APIKey.prisma().update(
        where={"id": info.id},
        data={"hash": hashlib.sha256(plaintext.encode()).hexdigest(), "salt": None},
    )
    assert (
        await client.get("/auth", headers=auth_headers(plaintext))
    ).status_code == 200
    migrated = await APIKey.prisma().find_unique_or_raise(where={"id": info.id})
    assert migrated.salt is not None
    assert api_key.keysmith.verify_key(plaintext, migrated.hash, migrated.salt)


async def test_unrelated_prefixes_do_not_block_valid_key_on_shared_address(
    client, stored_key, mocker, auth_headers
):
    verify = mocker.spy(api_key.keysmith, "verify_key")
    redis_access = mocker.spy(rate_limit, "get_redis_async")
    plaintext = stored_key[1]
    heads = [f"agpt_{i:03d}" for i in range(121) if f"agpt_{i:03d}" != plaintext[:8]]
    for head in heads[:120]:
        assert (
            await client.get("/auth", headers=auth_headers(head + "garbage-tail"))
        ).status_code == 401
    assert verify.call_count == 0
    redis_access.assert_not_called()
    assert (
        await client.get("/auth", headers=auth_headers(plaintext))
    ).status_code == 200
    assert verify.call_count == 1


async def test_ipv6_rotation_shares_budget_but_other_subnet_still_works(
    client, app, stored_key, monkeypatch, mocker, auth_headers
):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 3)
    verify = mocker.spy(api_key.keysmith, "verify_key")
    plaintext = stored_key[1]
    bad = plaintext[:8] + ("X" if plaintext[8] != "X" else "Y") + plaintext[9:]
    subnet = IPv6Network(f"{random_client_ip()}/64", strict=False)
    responses = []
    for suffix in range(1, 5):
        transport = httpx.ASGITransport(app, client=(str(subnet[suffix]), 1234))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            responses.append(await c.get("/auth", headers=auth_headers(bad)))
    assert [r.status_code for r in responses] == [401, 401, 401, 429]
    assert verify.call_count == 3
    other_ip = str(IPv6Address(int(subnet.network_address) + (1 << 64)))
    transport = httpx.ASGITransport(app, client=(other_ip, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (
            await c.get("/auth", headers=auth_headers(plaintext))
        ).status_code == 200
    assert verify.call_count == 4


async def test_fixed_window_expires_and_late_refund_does_not_recreate_it(
    stored_key, monkeypatch, mocker
):
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS", 1)
    monkeypatch.setattr(rate_limit, "AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS", 10)
    clock = mocker.patch.object(rate_limit, "datetime")
    clock.now.return_value = datetime(2026, 1, 1, 0, 0, 17, tzinfo=UTC)
    request = Request(
        {"type": "http", "headers": [], "client": (random_client_ip(), 1234)}
    )
    budget = await rate_limit.enforce_api_key_validate_rate_limit(
        request, stored_key[1]
    )
    assert budget is not None
    redis = await get_redis_async()
    assert 0 < await redis.pttl(budget.head_key) <= 60000
    await redis.expire(budget.head_key, 0)
    clock.now.return_value += timedelta(minutes=1)
    new_budget = await rate_limit.enforce_api_key_validate_rate_limit(
        request, stored_key[1]
    )
    assert new_budget is not None
    assert new_budget.head_key != budget.head_key
    await rate_limit.refund_api_key_validate_rate_limit(budget)
    assert not await redis.exists(budget.head_key)
    assert await redis.get(new_budget.head_key) == "1"
    await rate_limit.refund_api_key_validate_rate_limit(new_budget)
    assert await redis.get(new_budget.head_key) == "0"


def random_client_ip() -> str:
    token = uuid4().hex
    return "2001:db8:" + ":".join(token[i : i + 4] for i in range(0, 24, 4))
