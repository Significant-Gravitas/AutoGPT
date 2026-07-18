"""Tests for the catalog per-IP rate limiter and client-IP extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from redis.exceptions import RedisError

from backend.api.features.llm.rate_limit import check_catalog_rate_limit, get_client_ip


def _request(headers: dict | None = None, client_host: str = "10.0.0.9"):
    request = MagicMock(spec=fastapi.Request)
    request.headers = headers or {}
    request.client = MagicMock()
    request.client.host = client_host
    return request


def test_client_ip_no_xff_uses_socket_peer():
    assert get_client_ip(_request()) == "10.0.0.9"


def test_client_ip_takes_lb_appended_xff_entry():
    # Client-forgeable entries come first; the trusted LB appends last.
    request = _request({"x-forwarded-for": "6.6.6.6, 203.0.113.7"})
    assert get_client_ip(request) == "203.0.113.7"


def test_client_ip_single_xff_entry():
    request = _request({"x-forwarded-for": "203.0.113.7"})
    assert get_client_ip(request) == "203.0.113.7"


def test_client_ip_no_client_object():
    request = _request()
    request.client = None
    assert get_client_ip(request) == "unknown"


def _mock_redis(mocker, count: int):
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    pipe.incrby = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[count, True])
    redis = MagicMock()
    redis.pipeline.return_value = pipe
    mocker.patch(
        "backend.api.features.llm.rate_limit.get_redis_async",
        return_value=redis,
    )
    return pipe


@pytest.mark.asyncio
async def test_under_limit_allows(mocker):
    _mock_redis(mocker, count=1)
    assert await check_catalog_rate_limit("1.2.3.4") is True


@pytest.mark.asyncio
async def test_over_limit_blocks(mocker):
    _mock_redis(mocker, count=99999)
    assert await check_catalog_rate_limit("1.2.3.4") is False


@pytest.mark.asyncio
async def test_redis_error_fails_open(mocker):
    mocker.patch(
        "backend.api.features.llm.rate_limit.get_redis_async",
        side_effect=RedisError("cluster down"),
    )
    assert await check_catalog_rate_limit("1.2.3.4") is True
