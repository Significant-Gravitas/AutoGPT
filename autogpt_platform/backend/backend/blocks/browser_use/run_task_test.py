"""Unit tests for the Browser Use run-task block and its REST client.

Complements the block's ``test_mock`` harness (run by ``test_available_blocks``)
with the failure paths: failed runs, timeouts and stopped nodes must cancel
the run and still bill what it spent. All network calls are mocked.
"""

import asyncio
from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.browser_use._api import (
    RUNS_URL,
    SESSIONS_URL,
    BrowserUseClient,
    BrowserUseRun,
    BrowserUseRunStatus,
    BrowserUseSessionCost,
    build_run_payload,
)
from backend.blocks.browser_use.run_task import BrowserUseRunTaskBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="browser_use",
    api_key=SecretStr("mock-browser-use-api-key"),
    title="Mock Browser Use API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def _run(status: BrowserUseRunStatus, **fields) -> BrowserUseRun:
    return BrowserUseRun(id="run-1", status=status, session_id="session-1", **fields)


def _block(
    statuses: list[BrowserUseRunStatus], final: BrowserUseRun
) -> BrowserUseRunTaskBlock:
    """A block whose network methods are AsyncMocks replaying ``statuses``."""
    block = BrowserUseRunTaskBlock()
    block._create_run = mock.AsyncMock(return_value="run-1")
    block._get_status = mock.AsyncMock(side_effect=statuses)
    block._get_run = mock.AsyncMock(return_value=final)
    block._get_session_cost = mock.AsyncMock(
        return_value=BrowserUseSessionCost(browser_cost_usd=0.002, proxy_cost_usd=0.003)
    )
    block._cancel_run = mock.AsyncMock(return_value=None)
    return block


async def _collect(block: BrowserUseRunTaskBlock, **inputs) -> dict:
    outputs = {}
    async for name, value in block.execute(
        {"credentials": TEST_CREDENTIALS_INPUT, "task": "do it", **inputs},
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def test_build_run_payload_defaults():
    payload = build_run_payload("do it", "", None, "us", "", None)
    assert payload == {"task": "do it", "browserSettings": {"proxyCountryCode": "us"}}


def test_build_run_payload_all_fields():
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    payload = build_run_payload(
        "do it", " gpt-6-luna ", schema, " DE ", " prof-1 ", 0.5
    )
    assert payload == {
        "task": "do it",
        "model": "gpt-6-luna",
        "outputSchema": schema,
        "maxCostUsd": 0.5,
        "browserSettings": {"proxyCountryCode": "de", "profileId": "prof-1"},
    }


def test_build_run_payload_empty_proxy_disables_it():
    payload = build_run_payload("do it", "", None, "  ", "", None)
    assert payload["browserSettings"] == {"proxyCountryCode": None}


def test_run_summary_parses_string_cost():
    run = BrowserUseRun.model_validate(
        {
            "id": "run-1",
            "status": "completed",
            "sessionId": "session-1",
            "result": "done",
            "output": {"title": "x"},
            "error": None,
            "totalCostUsd": "0.0123",
        }
    )
    assert run.total_cost_usd == pytest.approx(0.0123)
    assert run.output == {"title": "x"}


async def test_client_sends_api_key_header_and_hits_v4_routes():
    client = BrowserUseClient(TEST_CREDENTIALS)
    assert client.requests.extra_headers == {
        "X-Browser-Use-API-Key": "mock-browser-use-api-key"
    }
    created, status, cost = mock.Mock(), mock.Mock(), mock.Mock()
    created.json.return_value = {"id": "run-1", "status": "queued"}
    status.json.return_value = {"status": "running"}
    cost.json.return_value = {"browserCostUsd": "0.004333", "proxyCostUsd": "0.01"}
    client.requests.post = mock.AsyncMock(return_value=created)
    client.requests.get = mock.AsyncMock(side_effect=[status, cost])

    assert await client.create_run({"task": "do it"}) == "run-1"
    assert await client.get_status("run-1") == BrowserUseRunStatus.RUNNING
    session_cost = await client.get_session_cost("session-1")
    await client.cancel_run("run-1")

    assert session_cost.browser_cost_usd == pytest.approx(0.004333)
    assert session_cost.proxy_cost_usd == pytest.approx(0.01)

    assert client.requests.post.call_args_list == [
        mock.call(RUNS_URL, json={"task": "do it"}),
        mock.call(f"{RUNS_URL}/run-1/cancel"),
    ]
    assert client.requests.get.call_args_list == [
        mock.call(f"{RUNS_URL}/run-1/status"),
        mock.call(f"{SESSIONS_URL}/session-1/cost"),
    ]


async def test_completed_run_yields_outputs_and_bills_full_cost():
    final = _run(
        BrowserUseRunStatus.COMPLETED,
        result="done",
        output={"title": "x"},
        total_cost_usd=0.02,
    )
    block = _block([BrowserUseRunStatus.RUNNING, BrowserUseRunStatus.COMPLETED], final)

    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        outputs = await _collect(block, output_schema={"type": "object"})

    assert outputs == {
        "result": "done",
        "output": {"title": "x"},
        "run_id": "run-1",
        "session_id": "session-1",
        "cost_usd": pytest.approx(0.025),
    }
    payload = block._create_run.call_args.args[1]
    assert payload["outputSchema"] == {"type": "object"}
    # Run LLM spend 0.02 + session browser 0.002 + proxy 0.003.
    assert block.execution_stats.provider_cost == pytest.approx(0.025)
    block._get_session_cost.assert_awaited_once_with(TEST_CREDENTIALS, "session-1")
    block._cancel_run.assert_not_awaited()


async def test_unreadable_session_cost_falls_back_to_run_cost():
    final = _run(BrowserUseRunStatus.COMPLETED, result="done", total_cost_usd=0.02)
    block = _block([BrowserUseRunStatus.COMPLETED], final)
    block._get_session_cost = mock.AsyncMock(side_effect=RuntimeError("HTTP 503"))

    outputs = await _collect(block)

    assert outputs["result"] == "done"
    assert outputs["cost_usd"] == pytest.approx(0.02)
    assert block.execution_stats.provider_cost == pytest.approx(0.02)


async def test_failed_run_raises_and_still_bills_cost():
    final = _run(BrowserUseRunStatus.FAILED, error="site blocked", total_cost_usd=0.01)
    block = _block([BrowserUseRunStatus.FAILED], final)

    with pytest.raises(BlockExecutionError, match="failed: site blocked"):
        await _collect(block)

    assert block.execution_stats.provider_cost == pytest.approx(0.015)
    block._cancel_run.assert_not_awaited()


async def test_timeout_cancels_run_and_bills_spend():
    final = _run(BrowserUseRunStatus.CANCELLED, total_cost_usd=0.03)
    block = _block([BrowserUseRunStatus.RUNNING], final)

    # A zero timeout expires on the first non-terminal poll.
    with pytest.raises(BlockExecutionError, match="not finished after 0s"):
        await block._wait_or_cancel(TEST_CREDENTIALS, "run-1", 0)

    block._cancel_run.assert_awaited_once_with(TEST_CREDENTIALS, "run-1")
    assert block.execution_stats.provider_cost == pytest.approx(0.035)


async def test_stopped_node_cancels_run_and_reraises():
    block = _block([asyncio.CancelledError()], _run(BrowserUseRunStatus.CANCELLED))

    with pytest.raises(asyncio.CancelledError):
        await _collect(block)

    block._cancel_run.assert_awaited_once_with(TEST_CREDENTIALS, "run-1")


async def test_cancel_failure_does_not_mask_the_original_error():
    block = _block([RuntimeError("poll broke")], _run(BrowserUseRunStatus.RUNNING))
    block._cancel_run = mock.AsyncMock(side_effect=RuntimeError("cancel broke"))

    with pytest.raises(BlockExecutionError, match="poll broke"):
        await _collect(block)


async def test_create_failure_raises_without_polling():
    block = _block([], _run(BrowserUseRunStatus.COMPLETED))
    block._create_run = mock.AsyncMock(side_effect=RuntimeError("HTTP 402"))

    with pytest.raises(BlockExecutionError, match="Could not start the run: HTTP 402"):
        await _collect(block)

    block._get_status.assert_not_awaited()
    block._cancel_run.assert_not_awaited()
