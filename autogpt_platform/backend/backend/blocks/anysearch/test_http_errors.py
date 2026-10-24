import asyncio
import traceback

import pytest
from aiohttp import web

from backend.blocks.anysearch._api import AnySearchClient
from backend.blocks.anysearch._http import (
    AnySearchHTTPClientError,
    AnySearchHTTPServerError,
)
from backend.blocks.anysearch.domains import AnySearchDomainsBlock
from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.anysearch.parallel_search import AnySearchParallelSearchBlock
from backend.blocks.anysearch.search import AnySearchBlock
from backend.blocks.anysearch.test_helpers import _collect
from backend.blocks.anysearch.test_http_helpers import (
    HEADER_REQUEST_ID,
    QUOTA_RESPONSE,
    REQUEST_ID,
    anysearch_server,
    assert_no_credentials,
)
from backend.copilot.tools.utils import credential_rejection_status
from backend.util.exceptions import BlockExecutionError
from backend.util.request import HTTPClientError, HTTPServerError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_class,input_data",
    [
        (AnySearchBlock, {"query": "quota"}),
        (AnySearchExtractBlock, {"url": "https://example.com"}),
    ],
)
async def test_http_402_retains_status_and_id_without_credentials(
    monkeypatch, block_class, input_data
):
    async def handler(request):
        return web.json_response(QUOTA_RESPONSE, status=402)

    async with anysearch_server(monkeypatch, handler) as calls:
        with pytest.raises(BlockExecutionError) as exc:
            await asyncio.wait_for(_collect(block_class(), input_data), timeout=2)

    assert len(calls) == 1
    assert "HTTP 402" in str(exc.value)
    assert REQUEST_ID in str(exc.value)
    assert isinstance(exc.value.__cause__, HTTPClientError)
    assert exc.value.__cause__.status_code == 402
    assert_no_credentials("".join(traceback.format_exception(exc.value)))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 402, 403, 415, 422, 429, 500, 502])
async def test_parallel_http_failure_retains_id_and_successful_sibling(
    monkeypatch, status
):
    async def handler(request):
        payload = await request.json()
        if payload["query"] == "quota":
            return web.json_response(QUOTA_RESPONSE, status=status)
        return web.json_response(
            {"code": 0, "data": {"results": []}, "request_id": "success-id"}
        )

    async with anysearch_server(monkeypatch, handler) as calls:
        output = await asyncio.wait_for(
            _collect(AnySearchParallelSearchBlock(), {"queries": ["quota", "ok"]}),
            timeout=2,
        )

    assert len(calls) == 2
    failed, succeeded = output["results"]
    assert failed.request_id == REQUEST_ID
    assert f"HTTP {status}" in failed.error
    assert_no_credentials(failed.error)
    assert succeeded.request_id == "success-id"
    assert not succeeded.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [400, 401, 403, 408, 415, 422, 429, 500, 502, 503, 504]
)
@pytest.mark.parametrize("endpoint", ["search", "extract", "domains", "sub-domains"])
async def test_client_http_errors_are_typed_sanitized_and_not_retried(
    monkeypatch, status, endpoint
):
    async def handler(request):
        return web.json_response(QUOTA_RESPONSE, status=status)

    async with anysearch_server(monkeypatch, handler) as calls:
        client = AnySearchClient(None)
        request = {
            "search": lambda: client.search({"query": "q"}),
            "extract": lambda: client.extract("https://example.com"),
            "domains": client.domains,
            "sub-domains": lambda: client.sub_domains(["code"]),
        }[endpoint]
        with pytest.raises((AnySearchHTTPClientError, AnySearchHTTPServerError)) as exc:
            await asyncio.wait_for(request(), timeout=2)

    assert calls == [f"/v1/{endpoint}"]
    assert isinstance(exc.value, HTTPClientError if status < 500 else HTTPServerError)
    assert exc.value.status_code == status
    assert exc.value.request_id == REQUEST_ID
    assert f"HTTP {status}" in str(exc.value)
    assert REQUEST_ID in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert_no_credentials("".join(traceback.format_exception(exc.value)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,header,expected_id",
    [
        ({"request_id": REQUEST_ID}, HEADER_REQUEST_ID, REQUEST_ID),
        (
            {"request_id": REQUEST_ID, "message": {"password": "<PASSWORD>"}},
            None,
            REQUEST_ID,
        ),
        (
            {"request_id": REQUEST_ID, "code": 0, "data": {"results": []}},
            None,
            REQUEST_ID,
        ),
        ({}, HEADER_REQUEST_ID, HEADER_REQUEST_ID),
        ({"request_id": None}, HEADER_REQUEST_ID, HEADER_REQUEST_ID),
        ({"request_id": 123}, HEADER_REQUEST_ID, HEADER_REQUEST_ID),
        (
            {"request_id": {"api_key": "<API_KEY>"}},
            HEADER_REQUEST_ID,
            HEADER_REQUEST_ID,
        ),
        (
            {"request_id": "Authorization: Bearer <API_KEY>"},
            HEADER_REQUEST_ID,
            HEADER_REQUEST_ID,
        ),
        ({"request_id": "not-a-uuid"}, None, ""),
        ({"request_id": "a" * 10000}, None, ""),
        ({"request_id": "00000000-0000-0000-0000-000000000000"}, None, ""),
        ({}, "password=<PASSWORD>", ""),
        ({}, None, ""),
        (None, HEADER_REQUEST_ID, HEADER_REQUEST_ID),
        (["<PASSWORD>"], HEADER_REQUEST_ID, HEADER_REQUEST_ID),
        ("<API_KEY>", HEADER_REQUEST_ID, HEADER_REQUEST_ID),
    ],
)
async def test_error_metadata_validates_id_independently(
    monkeypatch, body, header, expected_id
):
    async def handler(request):
        headers = {"X-Request-ID": header} if header else {}
        return web.json_response(body, status=502, headers=headers)

    async with anysearch_server(monkeypatch, handler):
        with pytest.raises(AnySearchHTTPServerError) as exc:
            await asyncio.wait_for(
                AnySearchClient(None).search({"query": "q"}), timeout=2
            )

    assert exc.value.request_id == expected_id
    assert "HTTP 502" in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert_no_credentials("".join(traceback.format_exception(exc.value)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [b"", b"password=<PASSWORD>", b'{"api_key":"<API_KEY>"', b"\xff<API_KEY>"]
)
@pytest.mark.parametrize("header", [HEADER_REQUEST_ID, None, "password=<PASSWORD>"])
async def test_non_json_http_errors_preserve_status_and_safe_header(
    monkeypatch, body, header
):
    async def handler(request):
        return web.Response(
            body=body,
            status=502,
            headers={"X-Request-ID": header} if header else {},
            reason="password=<PASSWORD>",
        )

    async with anysearch_server(monkeypatch, handler) as calls:
        with pytest.raises(AnySearchHTTPServerError) as exc:
            await asyncio.wait_for(AnySearchClient(None).domains(), timeout=2)

    assert calls == ["/v1/domains"]
    assert exc.value.status_code == 502
    assert exc.value.request_id == (
        HEADER_REQUEST_ID if header == HEADER_REQUEST_ID else ""
    )
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert_no_credentials("".join(traceback.format_exception(exc.value)))


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [None, ["code"]])
async def test_catalog_server_error_keeps_hint_and_id(monkeypatch, domains):
    async def handler(request):
        return web.json_response(QUOTA_RESPONSE, status=502)

    async with anysearch_server(monkeypatch, handler) as calls:
        with pytest.raises(BlockExecutionError) as exc:
            await asyncio.wait_for(
                _collect(AnySearchDomainsBlock(), {"domains": domains}), timeout=2
            )

    assert len(calls) == 1
    assert "HTTP 502" in str(exc.value)
    assert "upstream catalog error" in str(exc.value)
    assert REQUEST_ID in str(exc.value)
    assert isinstance(exc.value.__cause__, HTTPServerError)
    assert_no_credentials("".join(traceback.format_exception(exc.value)))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "block_class,input_data",
    [
        (AnySearchBlock, {"query": "q"}),
        (AnySearchExtractBlock, {"url": "https://example.com"}),
        (AnySearchDomainsBlock, {}),
    ],
)
async def test_http_auth_errors_keep_consumer_classification(
    monkeypatch, status, block_class, input_data
):
    async def handler(request):
        return web.json_response(QUOTA_RESPONSE, status=status)

    async with anysearch_server(monkeypatch, handler):
        with pytest.raises(BlockExecutionError) as exc:
            await asyncio.wait_for(_collect(block_class(), input_data), timeout=2)

    assert isinstance(exc.value.__cause__, HTTPClientError)
    assert exc.value.__cause__.status_code == status
    assert credential_rejection_status(exc.value) == (401 if status == 401 else None)
    assert_no_credentials("".join(traceback.format_exception(exc.value)))
