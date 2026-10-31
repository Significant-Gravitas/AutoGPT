"""Tests for the Exa Agent HTTP client: payloads, validation, URLs, polling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa import _exa_rest, agent_api
from backend.blocks.exa._test import TEST_CREDENTIALS
from backend.blocks.exa.agent_api import AgentRunRequest, build_create_payload
from backend.blocks.exa.agent_fixtures import (
    SAMPLE_COMPLETED_RUN,
    SAMPLE_FAILED_EVENT_LIST,
    SAMPLE_FAILED_RUN,
    SAMPLE_QUEUED_RUN,
    SAMPLE_RUN_ID,
)
from backend.blocks.exa.agent_models import AgentDataSource, AgentEffort, parse_run


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    return response


def _mock_requests(payload: dict) -> MagicMock:
    instance = MagicMock()
    instance.get = AsyncMock(return_value=_response(payload))
    instance.post = AsyncMock(return_value=_response(payload))
    instance.delete = AsyncMock(return_value=_response(payload))
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


class TestBuildCreatePayload:
    def test_minimal_request_sends_query_and_effort_only(self):
        payload = build_create_payload(AgentRunRequest(query="Find X"))
        assert payload == {"query": "Find X", "effort": "auto"}

    def test_every_field_maps_to_its_camel_case_key(self):
        request = AgentRunRequest(
            query="Find companies",
            effort=AgentEffort.ULTRA,
            system_prompt="Official sites only",
            input_data=[{"company": "Acme"}],
            exclusions=[{"company": "Globex"}],
            output_schema={"type": "object"},
            previous_run_id="agent_run_prev",
            metadata={"team": "sales"},
            data_sources=[AgentDataSource.FIBER, AgentDataSource.SIMILARWEB],
            custom_data_sources=["harmonic"],
            max_cost_dollars=25.0,
            max_duration_seconds=1800,
            user_identifier="user-123",
        )
        assert build_create_payload(request) == {
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
            "dataSources": [
                {"provider": "fiber"},
                {"provider": "similarweb"},
                {"provider": "harmonic"},
            ],
            "budget": {"maxCostDollars": 25.0, "maxDurationSeconds": 1800},
            "userIdentifier": "user-123",
        }

    def test_exclusions_alone_send_only_exclusion(self):
        payload = build_create_payload(
            AgentRunRequest(query="q", exclusions=[{"animal": "goat"}])
        )
        assert payload["input"] == {"exclusion": [{"animal": "goat"}]}

    def test_duplicate_and_blank_data_sources_are_dropped(self):
        payload = build_create_payload(
            AgentRunRequest(
                query="q",
                data_sources=[AgentDataSource.FIBER],
                custom_data_sources=["fiber", "  ", "harmonic"],
            )
        )
        assert payload["dataSources"] == [
            {"provider": "fiber"},
            {"provider": "harmonic"},
        ]

    def test_auto_accepts_a_cost_cap(self):
        payload = build_create_payload(
            AgentRunRequest(query="q", effort=AgentEffort.AUTO, max_cost_dollars=3)
        )
        assert payload["budget"] == {"maxCostDollars": 3}


class TestValidation:
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_cost_cap_rejected_for_fixed_efforts(self, effort):
        with pytest.raises(ValueError, match="Max cost only applies"):
            build_create_payload(
                AgentRunRequest(
                    query="q", effort=AgentEffort(effort), max_cost_dollars=5
                )
            )

    @pytest.mark.parametrize("effort", ["auto", "medium"])
    def test_duration_rejected_unless_ultra(self, effort):
        with pytest.raises(ValueError, match="Max duration only applies"):
            build_create_payload(
                AgentRunRequest(
                    query="q", effort=AgentEffort(effort), max_duration_seconds=600
                )
            )

    def test_more_than_five_data_sources_rejected(self):
        with pytest.raises(ValueError, match="at most 5"):
            build_create_payload(
                AgentRunRequest(query="q", data_sources=list(AgentDataSource)[:6])
            )

    def test_blank_query_rejected(self):
        with pytest.raises(ValueError, match="Query must not be empty"):
            build_create_payload(AgentRunRequest(query="   "))


class TestHttpCalls:
    @pytest.mark.asyncio
    async def test_create_posts_json_to_agent_runs(self):
        requests = _mock_requests(SAMPLE_QUEUED_RUN)
        with patch.object(_exa_rest, "Requests", return_value=requests):
            await agent_api.create_agent_run(TEST_CREDENTIALS, {"query": "q"})
        url = requests.post.call_args.args[0]
        kwargs = requests.post.call_args.kwargs
        assert url == "https://api.exa.ai/agent/runs"
        assert kwargs["json"] == {"query": "q"}
        assert kwargs["headers"]["x-api-key"] == "mock-exa-api-key"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call, method, suffix",
        [
            (agent_api.get_agent_run, "get", ""),
            (agent_api.cancel_agent_run, "post", "/cancel"),
            (agent_api.stop_agent_run, "post", "/stop"),
            (agent_api.delete_agent_run, "delete", ""),
        ],
    )
    async def test_run_endpoints(self, call, method, suffix):
        requests = _mock_requests(SAMPLE_COMPLETED_RUN)
        with patch.object(_exa_rest, "Requests", return_value=requests):
            await call(TEST_CREDENTIALS, SAMPLE_RUN_ID)
        url = getattr(requests, method).call_args.args[0]
        assert url == f"https://api.exa.ai/agent/runs/{SAMPLE_RUN_ID}{suffix}"

    @pytest.mark.asyncio
    async def test_run_id_is_escaped_in_the_path(self):
        requests = _mock_requests(SAMPLE_COMPLETED_RUN)
        with patch.object(_exa_rest, "Requests", return_value=requests):
            await agent_api.get_agent_run(TEST_CREDENTIALS, "../other")
        assert requests.get.call_args.args[0] == (
            "https://api.exa.ai/agent/runs/..%2Fother"
        )

    @pytest.mark.asyncio
    async def test_blank_run_id_rejected_before_any_request(self):
        requests = _mock_requests({})
        with (
            patch.object(_exa_rest, "Requests", return_value=requests),
            pytest.raises(ValueError, match="Run ID must not be empty"),
        ):
            await agent_api.get_agent_run(TEST_CREDENTIALS, " ")
        requests.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_sends_limit_and_cursor(self):
        requests = _mock_requests({"data": []})
        with patch.object(_exa_rest, "Requests", return_value=requests):
            await agent_api.list_agent_runs(TEST_CREDENTIALS, 5, "agent_run_cursor")
        assert requests.get.call_args.kwargs["params"] == [
            ("limit", "5"),
            ("cursor", "agent_run_cursor"),
        ]

    @pytest.mark.asyncio
    async def test_events_follow_the_cursor(self):
        pages = [
            {
                "data": [{"id": "1", "event": "agent_run.created"}],
                "hasMore": True,
                "nextCursor": "c2",
            },
            {
                "data": [{"id": "2", "event": "agent_run.started"}],
                "hasMore": False,
                "nextCursor": None,
            },
        ]
        requests = _bind_request(MagicMock())
        requests.get = AsyncMock(side_effect=[_response(p) for p in pages])
        with patch.object(_exa_rest, "Requests", return_value=requests):
            events = await agent_api.list_all_agent_run_events(
                TEST_CREDENTIALS, SAMPLE_RUN_ID
            )
        assert [e.id for e in events] == ["1", "2"]
        assert requests.get.call_args_list[1].kwargs["params"] == [
            ("limit", "100"),
            ("cursor", "c2"),
        ]


class TestWaitForAgentRun:
    @pytest.mark.asyncio
    async def test_polls_until_terminal_with_growing_interval(self):
        get_run = AsyncMock(
            side_effect=[SAMPLE_QUEUED_RUN, SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]
        )
        sleep = AsyncMock()
        with (
            patch.object(agent_api, "get_agent_run", get_run),
            patch.object(agent_api, "sleep", sleep),
            patch.object(agent_api, "monotonic", return_value=0.0),
        ):
            result = await agent_api.wait_for_agent_run(
                TEST_CREDENTIALS, SAMPLE_RUN_ID, timeout_seconds=600, interval_seconds=4
            )
        assert result.timed_out is False
        assert result.run.status == "completed"
        assert [c.args[0] for c in sleep.call_args_list] == [4, 6]

    @pytest.mark.asyncio
    async def test_times_out_with_last_known_state(self):
        clock = iter([0.0, 5.0, 11.0])
        sleep = AsyncMock()
        with (
            patch.object(
                agent_api, "get_agent_run", AsyncMock(return_value=SAMPLE_QUEUED_RUN)
            ),
            patch.object(agent_api, "sleep", sleep),
            patch.object(agent_api, "monotonic", side_effect=lambda: next(clock)),
        ):
            result = await agent_api.wait_for_agent_run(
                TEST_CREDENTIALS, SAMPLE_RUN_ID, timeout_seconds=10, interval_seconds=8
            )
        assert result.timed_out is True
        assert result.run.status == "queued"
        assert result.elapsed_seconds == 11.0
        # Never sleeps past the deadline: 10 - 5 = 5 seconds left, not 8.
        assert [c.args[0] for c in sleep.call_args_list] == [5.0]


class TestDescribeFailure:
    @pytest.mark.asyncio
    async def test_reads_the_failed_event(self):
        with patch.object(
            agent_api,
            "list_agent_run_events",
            AsyncMock(return_value=SAMPLE_FAILED_EVENT_LIST),
        ):
            message = await agent_api.describe_failure(
                TEST_CREDENTIALS, parse_run(SAMPLE_FAILED_RUN)
            )
        assert message == (
            f"Exa Agent run {SAMPLE_RUN_ID} failed: INTERNAL_ERROR: Upstream timeout"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_stop_reason_when_events_are_unavailable(self):
        with patch.object(
            agent_api,
            "list_agent_run_events",
            AsyncMock(
                side_effect=_exa_rest.ExaApiError("Exa API error (404): not found", 404)
            ),
        ):
            message = await agent_api.describe_failure(
                TEST_CREDENTIALS, parse_run(SAMPLE_FAILED_RUN)
            )
        assert message == f"Exa Agent run {SAMPLE_RUN_ID} failed: stop reason 'error'"
