"""Tests for the SpecStream API client with the HTTP layer replaced."""

from typing import Any
from unittest.mock import patch

import pytest

from backend.blocks.specstream._api import SpecStreamClient, SpecStreamError
from backend.blocks.specstream._config import TEST_CREDENTIALS
from backend.blocks.specstream._test_helpers import (
    QUOTE_BODY,
    ROWS_BODY,
    FakeResponse,
    fake_requests,
)
from backend.blocks.specstream._testdata import (
    TEST_AST_HASH,
    TEST_FIND_ID,
    TEST_QUOTE_HASH,
)


@pytest.mark.asyncio
async def test_client_follows_the_quote_then_authorize_flow():
    requests = fake_requests(
        FakeResponse(200, {"data": {"status": "runnable", "ast_hash": TEST_AST_HASH}}),
        FakeResponse(200, {"data": QUOTE_BODY}),
        FakeResponse(
            202,
            {
                "data": {
                    "reservation_id": "rsv_1",
                    "find_id": TEST_FIND_ID,
                    "maximum_cents": 375,
                    "state": "reserved",
                    "replay": False,
                }
            },
        ),
        FakeResponse(200, {"data": ROWS_BODY}),
    )
    with patch("backend.blocks.specstream._api.Requests", return_value=requests):
        client = SpecStreamClient(TEST_CREDENTIALS)
        reading = await client.compile("fintech startups")
        quote = await client.quote(reading.ast_hash or "", 25)
        execution = await client.authorize(
            quote.quote_id, quote.quote_hash, quote.maximum_cents, "autogpt-key-1"
        )
        page = await client.list_rows(execution.find_id or "", None, 50)

    calls = [call.args[:2] for call in requests.request.await_args_list]
    assert calls == [
        ("POST", "https://api.specstream.ai/v1/parse"),
        ("POST", "https://api.specstream.ai/v1/capacity/quotes"),
        ("POST", "https://api.specstream.ai/v1/capacity/executions"),
        ("GET", f"https://api.specstream.ai/v1/finds/{TEST_FIND_ID}/rows"),
    ]
    quote_call, auth_call = (
        requests.request.await_args_list[1],
        requests.request.await_args_list[2],
    )
    assert quote_call.kwargs["json"] == {
        "ast_hash": TEST_AST_HASH,
        "operation": "find",
        "requested_units": 25,
    }
    assert auth_call.kwargs["json"] == {
        "quote_id": "qte_test_001",
        "quote_hash": TEST_QUOTE_HASH,
        "confirm_max_cents": 375,
        "idempotency_key": "autogpt-key-1",
    }
    assert (
        auth_call.kwargs["headers"]["Authorization"] == "Bearer mock-specstream-api-key"
    )
    assert page.rows[0].cells[0].provenance.source_url == "https://acme.example/about"
    assert all(
        call.kwargs["allow_redirects"] is False
        for call in requests.request.await_args_list
    )


@pytest.mark.parametrize(
    "status_code, error, expected",
    [
        (
            400,
            {"code": "invalid_request", "message": "Bad ast_hash"},
            "SpecStream invalid_request: Bad ast_hash$",
        ),
        (
            401,
            {"code": "invalid_token", "message": "Key not found."},
            r"Key not found\. Check the SpecStream API key\.$",
        ),
        (
            403,
            {"code": "capacity", "message": "Over the spending limit."},
            r"^SpecStream capacity: Over the spending limit\. Check the spending limit",
        ),
        (
            402,
            {
                "code": "unquoted_plan",
                "message": "The quote expired.",
                "details": {"commercial_code": "quote_expired"},
            },
            "SpecStream quote_expired: The quote expired.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_client_raises_the_api_error_message(
    status_code: int, error: dict[str, Any], expected: str
):
    requests = fake_requests(FakeResponse(status_code, {"error": error}))
    with (
        patch("backend.blocks.specstream._api.Requests", return_value=requests),
        pytest.raises(SpecStreamError, match=expected),
    ):
        await SpecStreamClient(TEST_CREDENTIALS).get_run(TEST_FIND_ID)
