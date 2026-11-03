"""Tests for the SpecStream blocks with the HTTP layer replaced."""

import json
from itertools import count
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.specstream import run_status
from backend.blocks.specstream._api import SpecStreamClient, SpecStreamError
from backend.blocks.specstream._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.specstream._models import Execution, Reading, UnsupportedClause
from backend.blocks.specstream._testdata import (
    TEST_AST_HASH,
    TEST_COMPLETED,
    TEST_EXECUTION,
    TEST_FIND_ID,
    TEST_QUOTE,
    TEST_QUOTE_HASH,
    TEST_READING,
    TEST_ROWS_PAGE,
    TEST_RUNNING,
)
from backend.blocks.specstream.approve_and_run import SpecStreamApproveAndRunBlock
from backend.blocks.specstream.dataset_request import (
    SpecStreamCreateDatasetRequestBlock,
)
from backend.blocks.specstream.list_rows import SpecStreamListRowsBlock
from backend.blocks.specstream.run_status import (
    SpecStreamGetRunStatusBlock,
    SpecStreamWaitForRunBlock,
)
from backend.data.model import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError, BlockInputError

QUOTE_BODY = {
    "quote_id": "qte_test_001",
    "quote_hash": TEST_QUOTE_HASH,
    "maximum_cents": 375,
    "expires_at": "2026-11-03T10:00:00Z",
}
ROWS_BODY = {
    "find_id": TEST_FIND_ID,
    "status": "completed",
    "accepted_count": 1,
    "next_cursor": None,
    "rows": [
        {
            "entity_id": "ent_1",
            "ordinal": 0,
            "cells": [
                {
                    "column_id": "identity",
                    "label": "Name",
                    "value": "Acme Pay",
                    "display": "Acme Pay",
                    "provenance": {
                        "source_url": "https://acme.example/about",
                        "observed_at": "2026-11-03T09:30:00Z",
                    },
                }
            ],
        }
    ],
}


class FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status = status_code
        self.ok = 200 <= status_code < 300
        self._text = json.dumps(body)

    def text(self) -> str:
        return self._text


def fake_requests(*responses: FakeResponse) -> MagicMock:
    requests = MagicMock()
    requests.request = AsyncMock(side_effect=list(responses))
    return requests


async def _run(block, **inputs) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
    ):
        outputs.setdefault(name, value)
    return outputs


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


@pytest.mark.asyncio
async def test_create_request_refuses_an_unrunnable_reading_before_pricing():
    needs_edit = Reading(
        status="needs_edit",
        ast_hash=TEST_AST_HASH,
        unsupported=[UnsupportedClause(field_id="x", reason="No source covers this.")],
    )
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=needs_edit)
    block.price_reading = AsyncMock()

    with pytest.raises(BlockInputError, match="No source covers this"):
        await _run(block, request="anything")

    block.price_reading.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_request_says_when_the_service_is_down():
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=Reading(status="compiler_unavailable"))
    block.price_reading = AsyncMock()

    with pytest.raises(BlockExecutionError, match="service is down"):
        await _run(block, request="anything")

    block.price_reading.assert_not_awaited()


@pytest.mark.parametrize(
    "limit, expected",
    [
        ({"max": 100, "inferred": False}, 100),
        ({"max": 25, "inferred": True}, 40),
        ({"max": 50000, "inferred": False}, 10_000),
    ],
)
@pytest.mark.asyncio
async def test_create_request_takes_the_row_count_from_the_request(
    limit: dict[str, Any], expected: int
):
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(
        return_value=Reading.model_validate(
            {"status": "runnable", "ast_hash": TEST_AST_HASH, "ast": {"limit": limit}}
        )
    )
    block.price_reading = AsyncMock(return_value=TEST_QUOTE)

    await _run(block, request="100 fintech startups", row_count=40)

    block.price_reading.assert_awaited_once_with(
        TEST_CREDENTIALS, TEST_AST_HASH, expected
    )


@pytest.mark.asyncio
async def test_create_request_summary_states_the_maximum():
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=TEST_READING)
    block.price_reading = AsyncMock(return_value=TEST_QUOTE)

    outputs = await _run(block, request="fintech startups", row_count=25)

    block.price_reading.assert_awaited_once_with(TEST_CREDENTIALS, TEST_AST_HASH, 25)
    assert outputs["maximum_cents"] == 375
    assert outputs["price_summary"].startswith("Up to 3.75 USD for up to 25 rows")


def _execution(find_id: str | None = TEST_FIND_ID) -> Execution:
    return TEST_EXECUTION.model_copy(update={"find_id": find_id})


def _approve_inputs(**extra) -> dict[str, Any]:
    return {
        "quote_id": "qte_test_001",
        "quote_hash": TEST_QUOTE_HASH,
        "confirm_max_cents": 375,
        **extra,
    }


@pytest.mark.parametrize(
    "key, expected",
    [(None, "autogpt-qte_test_001"), ("my-own-key-1", "my-own-key-1")],
)
@pytest.mark.asyncio
async def test_approve_uses_a_key_that_a_retry_repeats(key: str | None, expected: str):
    block = SpecStreamApproveAndRunBlock()
    block.authorize = AsyncMock(return_value=_execution())

    await _run(block, **_approve_inputs(idempotency_key=key))

    assert block.authorize.await_args.args[2] == expected


def test_approve_is_an_irreversible_action_that_needs_a_stated_maximum():
    block = SpecStreamApproveAndRunBlock()
    assert block.is_irreversible_action
    with pytest.raises(ValueError):
        block.input_schema(
            credentials=TEST_CREDENTIALS_INPUT,
            quote_id="qte_1",
            quote_hash=TEST_QUOTE_HASH,
        )
    with pytest.raises(ValueError):
        block.input_schema(
            credentials=TEST_CREDENTIALS_INPUT,
            **_approve_inputs(idempotency_key="short"),
        )


@pytest.mark.asyncio
async def test_approve_reads_the_run_id_from_the_quote_when_authorize_has_none():
    requests = fake_requests(
        FakeResponse(
            202,
            {
                "data": {
                    "reservation_id": "rsv_1",
                    "maximum_cents": 375,
                    "state": "reserved",
                }
            },
        ),
        FakeResponse(
            200,
            {"data": {**QUOTE_BODY, "existing_execution": {"find_id": "fnd_9"}}},
        ),
    )
    with patch("backend.blocks.specstream._api.Requests", return_value=requests):
        outputs = await _run(SpecStreamApproveAndRunBlock(), **_approve_inputs())

    assert outputs["run_id"] == "fnd_9"


@pytest.mark.asyncio
async def test_approve_names_the_reservation_when_no_run_id_comes_back():
    block = SpecStreamApproveAndRunBlock()
    block.authorize = AsyncMock(return_value=_execution(find_id=None))

    with pytest.raises(BlockExecutionError, match="rsv_test_001"):
        await _run(block, **_approve_inputs())


@pytest.mark.asyncio
async def test_wait_polls_with_a_growing_gap_until_the_run_ends():
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(side_effect=[TEST_RUNNING, TEST_RUNNING, TEST_COMPLETED])
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    with patch.object(run_status, "sleep", record):
        outputs = await _run(block, run_id=TEST_FIND_ID, check_interval=10)

    assert waited == [10.0, 15.0]
    assert outputs["status"] == "completed"
    assert outputs["timed_out"] is False
    assert outputs["charged_cents"] == 375


def test_wait_block_limit_covers_the_longest_timeout():
    limit = SpecStreamWaitForRunBlock().execution_timeout_seconds
    assert limit is not None
    assert limit > run_status.MAX_WAIT_SECONDS


@pytest.mark.asyncio
async def test_wait_gives_up_at_the_timeout_and_leaves_the_run_going():
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(return_value=TEST_RUNNING)
    clock = count(step=100.0)

    async def no_wait(seconds: float) -> None:
        return None

    with (
        patch.object(run_status, "sleep", no_wait),
        patch.object(run_status, "monotonic", lambda: next(clock)),
    ):
        outputs = await _run(block, run_id=TEST_FIND_ID, timeout=250)

    assert outputs["timed_out"] is True
    assert outputs["is_done"] is False
    assert outputs["status"] == "running"


@pytest.mark.asyncio
async def test_list_rows_asks_only_for_the_rows_still_needed_and_keeps_each_source():
    first = TEST_ROWS_PAGE.model_copy(update={"next_cursor": "cur_2"})
    last = TEST_ROWS_PAGE.model_copy(update={"next_cursor": None})
    block = SpecStreamListRowsBlock()
    block.read_page = AsyncMock(side_effect=[first, last])

    outputs = await _run(block, run_id=TEST_FIND_ID, max_rows=2)

    assert [call.args[2:] for call in block.read_page.await_args_list] == [
        (None, 2),
        ("cur_2", 1),
    ]
    assert outputs["row_count"] == 2
    assert outputs["next_cursor"] == ""
    cell = outputs["rows"][0].cells[0]
    assert (cell.source_url, cell.read_at) == (
        "https://acme.example/about",
        "2026-11-03T09:30:00+00:00",
    )
    assert outputs["rows"][0].values["website"] == "https://acme.example"
    sources = {
        cell.column_id: (cell.source_url, cell.read_at)
        for cell in outputs["rows"][0].cells
    }
    witness = ("https://acme.example/team", "2026-11-03T09:31:00+00:00")
    assert sources["detail:w1"] == witness
    assert sources["matches"] == witness


PLATFORM_KEY = APIKeyCredentials(
    id="specstream-default",
    provider="specstream",
    api_key=SecretStr("platform-key"),
    title="SpecStream API Key",
)
BLOCK_INPUTS = [
    (SpecStreamCreateDatasetRequestBlock, {"request": "anything"}),
    (SpecStreamApproveAndRunBlock, _approve_inputs()),
    (SpecStreamGetRunStatusBlock, {"run_id": TEST_FIND_ID}),
    (SpecStreamWaitForRunBlock, {"run_id": TEST_FIND_ID}),
    (SpecStreamListRowsBlock, {"run_id": TEST_FIND_ID}),
]


@pytest.mark.parametrize("block_class, inputs", BLOCK_INPUTS)
@pytest.mark.asyncio
async def test_every_block_refuses_the_platform_key_before_any_call(
    block_class, inputs
):
    block = block_class()
    with (
        patch("backend.blocks.specstream._api.Requests") as requests,
        pytest.raises(BlockExecutionError, match="own SpecStream API key"),
    ):
        async for _ in block.run(
            block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
            credentials=PLATFORM_KEY,
        ):
            pass
    requests.assert_not_called()
