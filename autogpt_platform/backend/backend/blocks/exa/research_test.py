"""Backward-compatibility tests: saved Research block inputs now drive /agent/runs.

Each test feeds a block the input a saved workflow would hold (field names and
enum values from before the migration) and checks the Exa Agent request it
sends and the legacy outputs it returns.
"""

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa import _exa_rest, agent_api
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.agent_fixtures import (
    SAMPLE_CANCELLED_RUN,
    SAMPLE_COMPLETED_RUN,
    SAMPLE_EVENT_LIST,
    SAMPLE_FAILED_EVENT_LIST,
    SAMPLE_FAILED_RUN,
    SAMPLE_QUEUED_RUN,
    SAMPLE_RUN_ID,
    SAMPLE_RUN_LIST,
)
from backend.blocks.exa.research import (
    ExaCreateResearchBlock,
    ExaGetResearchBlock,
    ExaListResearchBlock,
    ExaWaitForResearchBlock,
)
from backend.blocks.exa.research_models import iso_to_ms, legacy_status

RUNS_URL = "https://api.exa.ai/agent/runs"
CREATED_MS = iso_to_ms(SAMPLE_COMPLETED_RUN["createdAt"])
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    return response


def _requests(post: Sequence[dict] = (), get: Sequence[dict] = ()) -> MagicMock:
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


async def _run(block, saved_input: dict) -> list[tuple[str, object]]:
    input_data = block.input_schema(
        **{"credentials": TEST_CREDENTIALS_INPUT, **saved_input}
    )
    return [
        o
        async for o in block.run(
            input_data, credentials=TEST_CREDENTIALS, user_id="user-1"
        )
    ]


class TestCreateResearch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, effort",
        [
            ("exa-research-fast", "low"),
            ("exa-research", "medium"),
            ("exa-research-pro", "high"),
        ],
    )
    async def test_saved_input_creates_agent_run(self, model, effort):
        block = ExaCreateResearchBlock()
        requests = _requests(
            post=[SAMPLE_QUEUED_RUN], get=[SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]
        )
        sleep = AsyncMock()
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            patch.object(agent_api, "sleep", sleep),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            outputs = await _run(
                block,
                {
                    "instructions": "Research AI coding assistants",
                    "model": model,
                    "output_schema": SCHEMA,
                    "wait_for_completion": True,
                    "polling_timeout": 600,
                },
            )

        assert requests.post.call_args.args[0] == RUNS_URL
        assert requests.post.call_args.kwargs["json"] == {
            "query": "Research AI coding assistants",
            "effort": effort,
            "outputSchema": SCHEMA,
        }
        assert requests.get.call_args.args[0] == f"{RUNS_URL}/{SAMPLE_RUN_ID}"
        assert [c.args[0] for c in sleep.call_args_list] == [10]
        assert outputs == [
            ("research_id", SAMPLE_RUN_ID),
            ("status", "completed"),
            ("model", model),
            ("instructions", SAMPLE_COMPLETED_RUN["request"]["query"]),
            ("created_at", CREATED_MS),
            ("elapsed_time", outputs[5][1]),
            ("output_content", SAMPLE_COMPLETED_RUN["output"]["text"]),
            ("output_parsed", SAMPLE_COMPLETED_RUN["output"]["structured"]),
            ("cost_total", 0.1),
        ]
        merge_stats.assert_called_once()
        assert merge_stats.call_args.args[0].provider_cost == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_saved_input_without_model_uses_medium(self):
        block = ExaCreateResearchBlock()
        requests = _requests(post=[SAMPLE_QUEUED_RUN])
        with patch.object(_exa_rest, "Requests", return_value=requests):
            outputs = await _run(
                block, {"instructions": "Research X", "wait_for_completion": False}
            )
        assert requests.post.call_args.kwargs["json"] == {
            "query": "Research X",
            "effort": "medium",
        }
        assert outputs == [
            ("research_id", SAMPLE_RUN_ID),
            ("status", "pending"),
            ("model", "exa-research"),
            ("instructions", SAMPLE_QUEUED_RUN["request"]["query"]),
            ("created_at", CREATED_MS),
        ]
        requests.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_still_raises_like_before(self):
        block = ExaCreateResearchBlock()
        clock = iter([0.0, 5.0])
        requests = _requests(post=(SAMPLE_QUEUED_RUN,), get=(SAMPLE_QUEUED_RUN,))
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            patch.object(agent_api, "monotonic", side_effect=lambda: next(clock)),
            pytest.raises(ValueError, match="did not complete within 1 seconds"),
        ):
            await _run(block, {"instructions": "Research X", "polling_timeout": 1})


class TestGetResearch:
    @pytest.mark.asyncio
    async def test_saved_input_reads_run_and_events(self):
        block = ExaGetResearchBlock()
        requests = _requests(get=[SAMPLE_COMPLETED_RUN, SAMPLE_EVENT_LIST])
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            outputs = dict(
                await _run(
                    block, {"research_id": SAMPLE_RUN_ID, "include_events": True}
                )
            )
        urls = [c.args[0] for c in requests.get.call_args_list]
        assert urls == [
            f"{RUNS_URL}/{SAMPLE_RUN_ID}",
            f"{RUNS_URL}/{SAMPLE_RUN_ID}/events",
        ]
        assert outputs["status"] == "completed"
        assert outputs["model"] == "exa-research"
        assert outputs["finished_at"] == iso_to_ms(SAMPLE_COMPLETED_RUN["completedAt"])
        assert outputs["cost_total"] == 0.1
        assert outputs["cost_searches"] == 4
        # Exa Agent reports neither; still emitted as 0 for saved workflows.
        assert outputs["cost_pages"] == 0
        assert outputs["cost_reasoning_tokens"] == 0
        assert outputs["error_message"] is None
        assert [e["event"] for e in outputs["events"]] == [
            "agent_run.created",
            "agent_run.started",
        ]
        merge_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_run_reports_error_message(self):
        block = ExaGetResearchBlock()
        requests = _requests(get=[SAMPLE_FAILED_RUN, SAMPLE_FAILED_EVENT_LIST])
        with patch.object(_exa_rest, "Requests", return_value=requests):
            outputs = dict(await _run(block, {"research_id": SAMPLE_RUN_ID}))
        assert outputs["status"] == "failed"
        assert "output_content" not in outputs
        assert "INTERNAL_ERROR: Upstream timeout" in str(outputs["error_message"])
        assert "events" not in outputs

    @pytest.mark.asyncio
    async def test_cancelled_run_uses_legacy_spelling(self):
        block = ExaGetResearchBlock()
        requests = _requests(get=[SAMPLE_CANCELLED_RUN])
        with patch.object(_exa_rest, "Requests", return_value=requests):
            outputs = dict(await _run(block, {"research_id": SAMPLE_RUN_ID}))
        assert outputs["status"] == "canceled"


class TestWaitForResearch:
    @pytest.mark.asyncio
    async def test_saved_input_polls_at_the_saved_interval(self):
        block = ExaWaitForResearchBlock()
        requests = _requests(
            get=[SAMPLE_QUEUED_RUN, SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]
        )
        sleep = AsyncMock()
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            patch.object(agent_api, "sleep", sleep),
        ):
            outputs = dict(
                await _run(
                    block,
                    {"research_id": SAMPLE_RUN_ID, "timeout": 600, "check_interval": 7},
                )
            )
        assert [c.args[0] for c in sleep.call_args_list] == [7, 7]
        assert outputs["final_status"] == "completed"
        assert outputs["timed_out"] is False
        assert outputs["output_content"] == SAMPLE_COMPLETED_RUN["output"]["text"]
        assert outputs["cost_total"] == 0.1

    @pytest.mark.asyncio
    async def test_timeout_reports_last_status(self):
        block = ExaWaitForResearchBlock()
        clock = iter([0.0, 30.0])
        requests = _requests(get=[SAMPLE_QUEUED_RUN])
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            patch.object(agent_api, "monotonic", side_effect=lambda: next(clock)),
        ):
            outputs = await _run(block, {"research_id": SAMPLE_RUN_ID, "timeout": 10})
        assert outputs == [
            ("research_id", SAMPLE_RUN_ID),
            ("final_status", "pending"),
            ("elapsed_time", 30.0),
            ("timed_out", True),
        ]


class TestListResearch:
    @pytest.mark.asyncio
    async def test_saved_input_lists_agent_runs(self):
        block = ExaListResearchBlock()
        requests = _requests(get=[SAMPLE_RUN_LIST])
        with patch.object(_exa_rest, "Requests", return_value=requests):
            outputs = await _run(block, {"limit": 10, "cursor": "old-cursor"})
        assert requests.get.call_args.args[0] == RUNS_URL
        assert requests.get.call_args.kwargs["params"] == [
            ("limit", "10"),
            ("cursor", "old-cursor"),
        ]
        tasks = dict(outputs)["research_tasks"]
        assert tasks[0].research_id == SAMPLE_RUN_ID
        assert tasks[0].model == "exa-research"
        assert tasks[0].cost_dollars.num_searches == 4
        assert tasks[0].cost_dollars.num_pages == 0
        assert dict(outputs)["has_more"] is True


@pytest.mark.parametrize(
    "status, legacy",
    [
        ("queued", "pending"),
        ("running", "running"),
        ("completed", "completed"),
        ("cancelled", "canceled"),
        ("failed", "failed"),
        ("something_new", "something_new"),
    ],
)
def test_status_mapping(status, legacy):
    assert legacy_status(status) == legacy


def test_iso_to_ms():
    assert iso_to_ms("1970-01-01T00:00:01.500Z") == 1500
    assert iso_to_ms(None) is None
    assert iso_to_ms("not a date") is None
