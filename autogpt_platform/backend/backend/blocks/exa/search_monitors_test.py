"""Request-building and response-parsing tests for the Search Monitor blocks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa._exa_rest import ExaApiError, exa_request
from backend.blocks.exa._search_monitor_samples import SAMPLE_MONITOR, SAMPLE_RUN
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.helpers import ContentSettings, SummarySettings
from backend.blocks.exa.search_monitor_runs import (
    ExaGetSearchMonitorRunBlock,
    ExaListSearchMonitorRunsBlock,
    ExaTriggerSearchMonitorBlock,
)
from backend.blocks.exa.search_monitors import (
    ExaCreateSearchMonitorBlock,
    build_create_body,
)
from backend.blocks.exa.search_monitors_batch import (
    ExaBatchSearchMonitorsBlock,
    SearchMonitorBatchAction,
    batch_body,
)
from backend.blocks.exa.search_monitors_manage import (
    ExaListSearchMonitorsBlock,
    list_params,
)
from backend.blocks.exa.search_monitors_models import (
    SearchMonitorModel,
    SearchMonitorRunModel,
    SearchMonitorStatus,
    SearchMonitorWebhookEvent,
    build_trigger,
)
from backend.blocks.exa.search_monitors_update import (
    ExaUpdateSearchMonitorBlock,
    build_update_body,
)

CREDS = {"credentials": TEST_CREDENTIALS_INPUT}


def _create_input(**overrides):
    inputs = {
        **CREDS,
        "query": "battery recycling",
        "webhook_url": " https://example.com/hook ",
        **overrides,
    }
    return ExaCreateSearchMonitorBlock.Input(**inputs)


class TestCreateBody:
    def test_minimal_body_uses_exa_defaults(self):
        assert build_create_body(_create_input()) == {
            "search": {"query": "battery recycling", "numResults": 10},
            "webhook": {"url": "https://example.com/hook"},
            "trigger": {"type": "interval", "period": "1d"},
        }

    def test_every_option_maps_to_exa_fields(self):
        body = build_create_body(
            _create_input(
                name="Batteries",
                period="6H",
                num_results=25,
                include_domains=["a.com"],
                exclude_domains=["b.com"],
                contents=ContentSettings(
                    text=True,
                    summary=SummarySettings(query="facts", output_schema={"x": 1}),
                ),
                output_schema={"type": "text"},
                metadata={"workspace": "w1"},
                webhook_events=[SearchMonitorWebhookEvent.RUN_COMPLETED],
            )
        )
        assert body == {
            "name": "Batteries",
            "search": {
                "query": "battery recycling",
                "numResults": 25,
                "includeDomains": ["a.com"],
                "excludeDomains": ["b.com"],
                # Summary schema is /search-only, so it is dropped for monitors.
                "contents": {"text": True, "summary": {"query": "facts"}},
            },
            "trigger": {"type": "interval", "period": "6h"},
            "outputSchema": {"type": "text"},
            "metadata": {"workspace": "w1"},
            "webhook": {
                "url": "https://example.com/hook",
                "events": ["monitor.run.completed"],
            },
        }

    def test_search_only_content_options_are_dropped(self):
        contents = ContentSettings(text=True, snapshot_as_of="2026-01-01T00:00:00Z")
        body = build_create_body(_create_input(contents=contents))
        assert body["search"]["contents"] == {"text": True}

    def test_monitor_content_options_are_kept(self):
        contents = ContentSettings(text=True, max_age_hours=24)
        body = build_create_body(_create_input(contents=contents))
        assert body["search"]["contents"] == {"text": True, "maxAgeHours": 24}

    def test_empty_period_makes_a_manual_only_monitor(self):
        assert "trigger" not in build_create_body(_create_input(period=" "))

    def test_blank_webhook_url_is_rejected(self):
        with pytest.raises(ValueError, match="webhook URL"):
            build_create_body(_create_input(webhook_url="  "))

    async def test_outputs_the_one_time_webhook_secret(self):
        block = ExaCreateSearchMonitorBlock()
        response = {**SAMPLE_MONITOR, "webhookSecret": "whsec_once"}
        block.call_exa = AsyncMock(return_value=response)
        outputs = {
            name: value
            async for name, value in block.run(
                _create_input(), credentials=TEST_CREDENTIALS
            )
        }
        assert outputs["webhook_secret"] == "whsec_once"
        args, kwargs = block.call_exa.call_args
        assert args[1:] == ("POST", "https://api.exa.ai/monitors")
        assert kwargs["body"]["search"]["query"] == "battery recycling"


class TestTrigger:
    @pytest.mark.parametrize("period", ["1h", "6h", "1d", "7d", "30d"])
    def test_accepts_single_unit_periods(self, period):
        assert build_trigger(period) == {"type": "interval", "period": period}

    @pytest.mark.parametrize("period", ["0h", "30m", "1h30m", "1w", "daily", "-1d"])
    def test_rejects_other_periods(self, period):
        with pytest.raises(ValueError, match="Invalid run interval"):
            build_trigger(period)


class TestUpdateBody:
    def _input(self, **overrides):
        return ExaUpdateSearchMonitorBlock.Input(**CREDS, monitor_id="m1", **overrides)

    def test_only_set_fields_are_sent(self):
        body = build_update_body(
            self._input(status="paused", query="new query", period="12h")
        )
        assert body == {
            "status": "paused",
            "search": {"query": "new query"},
            "trigger": {"type": "interval", "period": "12h"},
        }

    def test_remove_schedule_and_clear_metadata_send_null(self):
        body = build_update_body(
            self._input(
                remove_schedule=True,
                period="1d",
                clear_metadata=True,
                metadata={"a": "b"},
            )
        )
        assert body == {"trigger": None, "metadata": None}

    def test_webhook_changes_are_partial(self):
        body = build_update_body(
            self._input(webhook_events=[SearchMonitorWebhookEvent.RUN_CREATED])
        )
        assert body == {"webhook": {"events": ["monitor.run.created"]}}

    def test_nothing_to_update_is_an_error(self):
        with pytest.raises(ValueError, match="Nothing to update"):
            build_update_body(self._input())


class TestListAndBatch:
    def test_list_params_use_bracket_metadata(self):
        params = list_params(
            ExaListSearchMonitorsBlock.Input(
                **CREDS,
                status=SearchMonitorStatus.PAUSED,
                name="bat",
                metadata={"team": "r&d"},
                cursor="c1",
            )
        )
        assert params == [
            ("status", "paused"),
            ("name", "bat"),
            ("limit", "50"),
            ("cursor", "c1"),
            ("metadata[team]", "r&d"),
        ]

    def test_batch_requires_a_filter(self):
        with pytest.raises(ValueError, match="filter"):
            batch_body(
                ExaBatchSearchMonitorsBlock.Input(
                    **CREDS, action=SearchMonitorBatchAction.DELETE
                )
            )

    def test_batch_defaults_to_a_dry_run(self):
        body = batch_body(
            ExaBatchSearchMonitorsBlock.Input(
                **CREDS, action=SearchMonitorBatchAction.DELETE, name="old"
            )
        )
        assert body == {
            "action": "delete",
            "filter": {"name": "old"},
            "dry_run": True,
            "limit": 50,
        }


class TestModels:
    def test_monitor_flattens_search_trigger_and_webhook(self):
        monitor = SearchMonitorModel.from_api(SAMPLE_MONITOR)
        assert monitor.query.startswith("new battery recycling")
        assert monitor.period == "1d"
        assert monitor.webhook_url == "https://example.com/webhooks/exa"
        assert monitor.webhook_events == ["monitor.run.completed"]

    def test_manual_only_monitor_has_no_period(self):
        data = {**SAMPLE_MONITOR, "trigger": None, "nextRunAt": None, "metadata": None}
        monitor = SearchMonitorModel.from_api(data)
        assert monitor.period is None
        assert monitor.metadata == {}

    def test_pending_run_has_empty_output(self):
        run = SearchMonitorRunModel.from_api(
            {**SAMPLE_RUN, "status": "pending", "output": None}
        )
        assert run.results == []
        assert run.content is None
        assert run.grounding == []

    def test_structured_content_is_kept_as_json(self):
        output = {"content": {"announcements": [{"company": "A"}]}}
        run = SearchMonitorRunModel.from_api({**SAMPLE_RUN, "output": output})
        assert run.content == {"announcements": [{"company": "A"}]}


class TestExaRequest:
    def _response(self, status: int, body: bytes):
        response = MagicMock()
        response.status = status
        response.ok = 200 <= status < 300
        response.content = body
        response.text.return_value = body.decode()
        response.json.return_value = {"id": "m1"}
        return response

    async def test_sends_api_key_header(self):
        requests = MagicMock()
        requests.request = AsyncMock(return_value=self._response(200, b'{"id":"m1"}'))
        with patch("backend.blocks.exa._exa_rest.Requests", return_value=requests):
            data = await exa_request(
                TEST_CREDENTIALS, "GET", "https://api.exa.ai/monitors/m1"
            )
        assert data == {"id": "m1"}
        _, kwargs = requests.request.call_args
        assert kwargs["headers"] == {"x-api-key": "mock-exa-api-key"}

    async def test_error_carries_exas_message_and_tag(self):
        body = b'{"requestId":"r1","error":"Invalid period","tag":"INVALID_REQUEST"}'
        requests = MagicMock()
        requests.request = AsyncMock(return_value=self._response(400, body))
        with patch(
            "backend.blocks.exa._exa_rest.Requests", return_value=requests
        ), pytest.raises(ExaApiError) as exc:
            await exa_request(TEST_CREDENTIALS, "POST", "https://api.exa.ai/monitors")
        assert exc.value.status == 400
        assert "Invalid period [INVALID_REQUEST]" in str(exc.value)


class TestRunBlocks:
    @pytest.mark.asyncio
    async def test_trigger_posts_to_the_escaped_monitor_url(self):
        block = ExaTriggerSearchMonitorBlock()
        call = AsyncMock(return_value={"triggered": True})
        with patch.object(block, "call_exa", call):
            outputs = dict(
                [
                    o
                    async for o in block.run(
                        ExaTriggerSearchMonitorBlock.Input(**CREDS, monitor_id="a/b"),
                        credentials=TEST_CREDENTIALS,
                    )
                ]
            )
        assert outputs == {"monitor_id": "a/b", "triggered": True}
        assert call.call_args.args[1:] == (
            "POST",
            "https://api.exa.ai/monitors/a%2Fb/trigger",
        )

    @pytest.mark.asyncio
    async def test_list_runs_pages_and_yields_each_run(self):
        block = ExaListSearchMonitorRunsBlock()
        page = {"data": [SAMPLE_RUN], "hasMore": True, "nextCursor": "c2"}
        call = AsyncMock(return_value=page)
        with patch.object(block, "call_exa", call):
            outputs = [
                o
                async for o in block.run(
                    ExaListSearchMonitorRunsBlock.Input(
                        **CREDS,
                        monitor_id=SAMPLE_RUN["monitorId"],
                        limit=5,
                        cursor="c1",
                    ),
                    credentials=TEST_CREDENTIALS,
                )
            ]
        named = dict(outputs)
        assert [r.id for r in named["runs"]] == [SAMPLE_RUN["id"]]
        assert [name for name, _ in outputs].count("run") == 1
        assert named["has_more"] is True and named["next_cursor"] == "c2"
        assert call.call_args.kwargs["params"] == [("limit", "5"), ("cursor", "c1")]

    @pytest.mark.asyncio
    async def test_get_run_returns_results_and_failure_reason(self):
        block = ExaGetSearchMonitorRunBlock()
        with patch.object(block, "call_exa", AsyncMock(return_value=SAMPLE_RUN)):
            outputs = dict(
                [
                    o
                    async for o in block.run(
                        ExaGetSearchMonitorRunBlock.Input(
                            **CREDS,
                            monitor_id=SAMPLE_RUN["monitorId"],
                            run_id=SAMPLE_RUN["id"],
                        ),
                        credentials=TEST_CREDENTIALS,
                    )
                ]
            )
        assert outputs["run_id"] == SAMPLE_RUN["id"]
        assert outputs["status"] == SAMPLE_RUN["status"]
        assert outputs["run"] == SearchMonitorRunModel.from_api(SAMPLE_RUN)
