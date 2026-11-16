"""Behaviour tests for the Exa Agent blocks, mocked at the HTTP boundary."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa import _exa_rest, agent_api
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.agent import ExaAgentRunBlock
from backend.blocks.exa.agent_fixtures import (
    SAMPLE_CANCELLED_RUN,
    SAMPLE_COMPLETED_RUN,
    SAMPLE_FAILED_EVENT_LIST,
    SAMPLE_FAILED_RUN,
    SAMPLE_QUEUED_RUN,
    SAMPLE_RUN_ID,
    SAMPLE_STOPPED_RUN,
)
from backend.blocks.exa.agent_lookup import ExaGetAgentRunBlock, ExaWaitForAgentRunBlock
from backend.blocks.exa.agent_manage import ExaCancelAgentRunBlock, ExaStopAgentRunBlock


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    return response


def _requests(post: list[dict], get: list[dict]) -> MagicMock:
    instance = MagicMock()
    instance.post = AsyncMock(side_effect=[_response(p) for p in post])
    instance.get = AsyncMock(side_effect=[_response(g) for g in get])
    return _bind_request(instance)


def _bind_request(instance: MagicMock) -> MagicMock:
    """Route ``Requests().request(method, url, ...)`` to per-method mocks."""

    async def request(method: str, url: str, **kwargs):
        handler = {
            "GET": instance.get,
            "POST": instance.post,
            "DELETE": instance.delete,
        }
        return await handler[method](url, **kwargs)

    instance.request = request
    return instance


async def _run(block, raw_input: dict) -> list[tuple[str, object]]:
    input_data = block.input_schema(**raw_input)
    return [
        o
        async for o in block.run(
            input_data, credentials=TEST_CREDENTIALS, user_id="user-1"
        )
    ]


def _patched(requests: MagicMock):
    return (
        patch.object(_exa_rest, "Requests", return_value=requests),
        patch.object(agent_api, "sleep", AsyncMock()),
    )


class TestExaAgentRunBlock:
    @pytest.mark.asyncio
    async def test_sends_every_option_waits_and_bills_once(self):
        block = ExaAgentRunBlock()
        requests = _requests(
            post=[SAMPLE_QUEUED_RUN], get=[SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]
        )
        p1, p2 = _patched(requests)
        with p1, p2, patch.object(block, "merge_stats") as merge_stats:
            outputs = dict(
                await _run(
                    block,
                    {
                        "credentials": TEST_CREDENTIALS_INPUT,
                        "query": "Find companies",
                        "effort": "ultra",
                        "output_schema": {"type": "object"},
                        "system_prompt": "Official sites only",
                        "input_data": [{"company": "Acme"}],
                        "exclusions": [{"company": "Globex"}],
                        "previous_run_id": "agent_run_prev",
                        "data_sources": ["fiber", "financial_datasets"],
                        "metadata": {"team": "sales"},
                        "max_cost_dollars": 30,
                        "max_duration_seconds": 3600,
                        "user_identifier": "user-1",
                    },
                )
            )
        body = requests.post.call_args.kwargs["json"]
        assert requests.post.call_args.args[0] == "https://api.exa.ai/agent/runs"
        assert body == {
            "query": "Find companies",
            "effort": "ultra",
            "systemPrompt": "Official sites only",
            "input": {
                "data": [{"company": "Acme"}],
                "exclusion": [{"company": "Globex"}],
            },
            "outputSchema": {"type": "object"},
            "previousRunId": "agent_run_prev",
            "metadata": {"team": "sales"},
            "dataSources": [{"provider": "fiber"}, {"provider": "financial_datasets"}],
            "budget": {"maxCostDollars": 30.0, "maxDurationSeconds": 3600},
            "userIdentifier": "user-1",
        }
        assert outputs["run_id"] == SAMPLE_RUN_ID
        assert outputs["text"] == SAMPLE_COMPLETED_RUN["output"]["text"]
        assert outputs["structured"] == SAMPLE_COMPLETED_RUN["output"]["structured"]
        assert outputs["cost_total"] == 0.1
        assert "error" not in outputs
        merge_stats.assert_called_once()
        assert merge_stats.call_args.args[0].provider_cost == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_invalid_budget_fails_before_calling_exa(self):
        block = ExaAgentRunBlock()
        requests = _requests(post=[], get=[])
        p1, p2 = _patched(requests)
        with p1, p2, pytest.raises(ValueError, match="Max cost only applies"):
            await _run(
                block,
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "query": "q",
                    "effort": "low",
                    "max_cost_dollars": 5,
                },
            )
        requests.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_without_waiting_returns_the_run_id_and_bills_nothing(self):
        block = ExaAgentRunBlock()
        requests = _requests(post=[SAMPLE_QUEUED_RUN], get=[])
        p1, p2 = _patched(requests)
        with p1, p2, patch.object(block, "merge_stats") as merge_stats:
            outputs = await _run(
                block,
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "query": "q",
                    "wait_for_completion": False,
                },
            )
        assert outputs == [
            ("run_id", SAMPLE_RUN_ID),
            ("status", "queued"),
            ("created_at", SAMPLE_QUEUED_RUN["createdAt"]),
        ]
        requests.get.assert_not_called()
        merge_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_run_ends_with_exa_error_message(self):
        block = ExaAgentRunBlock()
        requests = _requests(
            post=[SAMPLE_QUEUED_RUN], get=[SAMPLE_FAILED_RUN, SAMPLE_FAILED_EVENT_LIST]
        )
        p1, p2 = _patched(requests)
        with p1, p2:
            outputs = await _run(
                block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"}
            )
        assert outputs[-1] == (
            "error",
            f"Exa Agent run {SAMPLE_RUN_ID} failed: INTERNAL_ERROR: Upstream timeout",
        )
        assert ("status", "failed") in outputs

    @pytest.mark.asyncio
    async def test_timeout_reports_run_id_then_errors(self):
        block = ExaAgentRunBlock()
        clock = iter([0.0, 20.0])
        requests = _requests(post=[SAMPLE_QUEUED_RUN], get=[SAMPLE_QUEUED_RUN])
        p1, p2 = _patched(requests)
        with (
            p1,
            p2,
            patch.object(agent_api, "monotonic", side_effect=lambda: next(clock)),
        ):
            outputs = await _run(
                block,
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "query": "q",
                    "polling_timeout": 10,
                },
            )
        assert [name for name, _ in outputs] == [
            "run_id",
            "status",
            "created_at",
            "error",
        ]
        assert SAMPLE_RUN_ID in str(outputs[-1][1])
        assert "Wait for Exa Agent Run" in str(outputs[-1][1])


class TestLookupBlocks:
    @pytest.mark.asyncio
    async def test_get_reports_failure_without_failing(self):
        block = ExaGetAgentRunBlock()
        requests = _requests(post=[], get=[SAMPLE_FAILED_RUN, SAMPLE_FAILED_EVENT_LIST])
        p1, p2 = _patched(requests)
        with p1, p2:
            outputs = dict(
                await _run(
                    block,
                    {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
                )
            )
        assert outputs["status"] == "failed"
        assert "Upstream timeout" in str(outputs["failure_reason"])
        assert "error" not in outputs

    @pytest.mark.asyncio
    async def test_a_finished_run_is_billed_once_across_blocks(self):
        """Whichever block sees the finished run first bills it; later reads don't."""
        wait_block, get_block = ExaWaitForAgentRunBlock(), ExaGetAgentRunBlock()
        run_input = {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID}
        requests = _requests(post=[], get=[SAMPLE_COMPLETED_RUN, SAMPLE_COMPLETED_RUN])
        p1, p2 = _patched(requests)
        with (
            p1,
            p2,
            patch.object(wait_block, "merge_stats") as wait_billed,
            patch.object(get_block, "merge_stats") as get_billed,
        ):
            await _run(wait_block, run_input)
            outputs = dict(await _run(get_block, run_input))
        assert outputs["cost_total"] == SAMPLE_COMPLETED_RUN["costDollars"]["total"]
        wait_billed.assert_called_once()
        get_billed.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_bills_a_run_nothing_else_collected(self):
        """A run started without waiting and read with Get is still paid for."""
        block = ExaGetAgentRunBlock()
        requests = _requests(post=[], get=[SAMPLE_COMPLETED_RUN])
        p1, p2 = _patched(requests)
        with p1, p2, patch.object(block, "merge_stats") as merge_stats:
            await _run(
                block, {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID}
            )
        merge_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_returns_result_and_bills_once(self):
        block = ExaWaitForAgentRunBlock()
        requests = _requests(post=[], get=[SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN])
        p1, p2 = _patched(requests)
        with p1, p2, patch.object(block, "merge_stats") as merge_stats:
            outputs = dict(
                await _run(
                    block,
                    {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
                )
            )
        assert outputs["timed_out"] is False
        assert outputs["text"] == SAMPLE_COMPLETED_RUN["output"]["text"]
        merge_stats.assert_called_once()


class TestEndingRuns:
    @pytest.mark.asyncio
    async def test_stop_bills_the_stopped_run_once(self):
        """Stopping twice (or stopping, then reading) charges the run once."""
        first, second = ExaStopAgentRunBlock(), ExaStopAgentRunBlock()
        run_input = {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID}
        with (
            patch.object(first, "stop_run", AsyncMock(return_value=SAMPLE_STOPPED_RUN)),
            patch.object(
                second, "stop_run", AsyncMock(return_value=SAMPLE_STOPPED_RUN)
            ),
            patch.object(first, "merge_stats") as first_billed,
            patch.object(second, "merge_stats") as second_billed,
        ):
            outputs = dict(await _run(first, run_input))
            await _run(second, run_input)
        assert outputs["stop_reason"] == SAMPLE_STOPPED_RUN["stopReason"]
        first_billed.assert_called_once()
        second_billed.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_bills_what_the_run_used(self):
        block = ExaCancelAgentRunBlock()
        with (
            patch.object(
                block, "cancel_run", AsyncMock(return_value=SAMPLE_CANCELLED_RUN)
            ),
            patch.object(block, "merge_stats") as billed,
        ):
            outputs = dict(
                await _run(
                    block,
                    {"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
                )
            )
        assert outputs["status"] == "cancelled"
        billed.assert_called_once()
