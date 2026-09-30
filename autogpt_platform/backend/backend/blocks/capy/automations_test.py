"""Tests for the Capy automation client and blocks.

Request bodies are checked against the shapes in Capy's OpenAPI reference
(POST /api/v1/automations), since automations can't be exercised through the
standard block harness beyond one canned call each.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.capy import _api
from backend.blocks.capy._automations_api import CapyAutomationsClient, trigger_payload
from backend.blocks.capy._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.capy._testdata import TEST_AUTOMATION
from backend.blocks.capy._types import Automation, AutomationTrigger
from backend.blocks.capy.automations import CapySetAutomationEnabledBlock
from backend.blocks.capy.create_automation import CapyCreateAutomationBlock

# Trimmed from the automation object in Capy's API reference.
LIVE_AUTOMATION = {
    "id": "aut_01",
    "projectId": "be134130-4101-49dd-a5b3-efd3371b7229",
    "name": "Fix new Sentry errors",
    "description": None,
    "prompt": "Root-cause this Sentry issue.",
    "triggers": [{"type": "sentry", "event": "any_issue"}],
    "model": None,
    "repos": [],
    "threadMode": "new",
    "slackImpersonation": False,
    "singleJamId": None,
    "mcpOverrides": None,
    "maxRunsPerDay": 10,
    "machineSize": None,
    "organize": None,
    "enabled": True,
    "deleted": False,
    "disabledReason": None,
    "runAsKind": "human",
    "runAsId": "user_1",
    "runCount": 3,
    "lastTriggeredAt": "2026-09-30T08:00:00.000Z",
    "createdAt": "2026-09-29T08:00:00.000Z",
    "updatedAt": "2026-09-30T08:00:00.000Z",
}


def _client_returning(body: dict[str, Any]) -> CapyAutomationsClient:
    response = MagicMock()
    response.ok = True
    response.status = 200
    response.content = b"x"
    response.json.return_value = body
    client = CapyAutomationsClient(TEST_CREDENTIALS)
    client.requests = MagicMock()
    client.requests.request = AsyncMock(return_value=response)
    return client


class TestTriggerPayload:
    def test_schedule_defaults_to_utc(self):
        assert trigger_payload(AutomationTrigger.SCHEDULE, cron="0 9 * * 1-5") == {
            "type": "schedule",
            "cron": "0 9 * * 1-5",
            "timezone": "UTC",
        }

    def test_event_trigger_carries_its_filters(self):
        payload = trigger_payload(
            AutomationTrigger.GITHUB,
            event="checks",
            conditions={"repositories": ["acme/app"], "conclusions": ["failure"]},
            run_when="Only when the failure is in a test",
        )

        assert payload == {
            "type": "github",
            "event": "checks",
            "conditions": {"repositories": ["acme/app"], "conclusions": ["failure"]},
            "run_when": "Only when the failure is in a test",
        }

    def test_filters_are_left_off_triggers_that_take_none(self):
        assert trigger_payload(
            AutomationTrigger.ON_DEMAND, conditions={"x": ["y"]}, run_when="z"
        ) == {"type": "on_demand"}
        assert "run_when" not in trigger_payload(
            AutomationTrigger.SCHEDULE, cron="0 * * * *", run_when="z"
        )

    def test_incoming_webhook_needs_no_event(self):
        assert trigger_payload(
            AutomationTrigger.INCOMING_WEBHOOK, conditions={"contains": ["deploy"]}
        ) == {"type": "incoming_webhook", "conditions": {"contains": ["deploy"]}}

    @pytest.mark.parametrize(
        "kind,kwargs,message",
        [
            (AutomationTrigger.SCHEDULE, {}, "needs cron"),
            (AutomationTrigger.SENTRY, {}, "any_issue"),
            (AutomationTrigger.LINEAR, {"event": "issue_deleted"}, "issue_created"),
        ],
    )
    def test_missing_or_unknown_fields_name_the_fix(self, kind, kwargs, message):
        with pytest.raises(ValueError, match=message):
            trigger_payload(kind, **kwargs)


class TestClient:
    def test_automation_parses_camel_case_and_ignores_new_fields(self):
        automation = Automation.model_validate({**LIVE_AUTOMATION, "brandNew": 1})

        assert automation.project_id == LIVE_AUTOMATION["projectId"]
        assert automation.max_runs_per_day == 10
        assert automation.run_count == 3
        assert automation.webhook_url is None

    async def test_create_sends_one_trigger_with_the_cap_and_model(self):
        client = _client_returning(
            {**LIVE_AUTOMATION, "webhookUrl": "https://hooks.capy.ai/x"}
        )

        automation = await client.create_automation(
            project_id="p1",
            name="Nightly deps",
            prompt="Bump dependencies",
            trigger={"type": "schedule", "cron": "0 3 * * *", "timezone": "UTC"},
            model_id="meta/muse-spark-1.3",
            max_runs_per_day=1,
            thread_mode="single",
        )

        kwargs = client.requests.request.call_args.kwargs
        body = kwargs["json"]
        assert client.requests.request.call_args.args == (
            "POST",
            f"{_api.API_URL}/automations",
        )
        assert body["requestId"]
        assert body["triggers"] == [
            {"type": "schedule", "cron": "0 3 * * *", "timezone": "UTC"}
        ]
        assert body["model"] == {"modelId": "meta/muse-spark-1.3"}
        assert body["maxRunsPerDay"] == 1
        assert body["threadMode"] == "single"
        assert body["enabled"] is True
        assert "machineSize" not in body and "description" not in body
        assert automation.webhook_url == "https://hooks.capy.ai/x"

    @pytest.mark.parametrize("enabled,action", [(True, "enable"), (False, "disable")])
    async def test_set_enabled_calls_the_matching_endpoint(self, enabled, action):
        client = _client_returning({**LIVE_AUTOMATION, "enabled": enabled})

        automation = await client.set_automation_enabled("aut_01", enabled)

        assert client.requests.request.call_args.args == (
            "POST",
            f"{_api.API_URL}/automations/aut_01/{action}",
        )
        assert automation.enabled is enabled


async def _run(block, **inputs) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
    ):
        collected.setdefault(name, value)
    return collected


class TestCreateAutomationBlock:
    async def test_bad_trigger_fails_before_calling_capy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyCreateAutomationBlock()
        create = AsyncMock()
        monkeypatch.setattr(block, "create_automation", create)

        with pytest.raises(ValueError, match="needs cron"):
            await _run(block, project_id="p1", name="n", prompt="p", trigger="schedule")
        create.assert_not_awaited()

    async def test_webhook_url_is_emitted_only_when_capy_returns_one(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyCreateAutomationBlock()
        hooked = TEST_AUTOMATION.model_copy(
            update={"webhook_url": "https://hooks.capy.ai/x"}
        )
        create = AsyncMock(side_effect=[hooked, TEST_AUTOMATION])
        monkeypatch.setattr(block, "create_automation", create)
        inputs = {"project_id": "p1", "name": "n", "prompt": "p"}

        with_hook = await _run(block, trigger="incoming_webhook", **inputs)
        without = await _run(block, trigger="on_demand", **inputs)

        assert with_hook["webhook_url"] == "https://hooks.capy.ai/x"
        assert "webhook_url" not in without
        assert create.await_args_list[0].args[2] == {"type": "incoming_webhook"}


class TestSetAutomationEnabledBlock:
    async def test_reports_the_state_capy_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapySetAutomationEnabledBlock()
        monkeypatch.setattr(
            block,
            "set_enabled",
            AsyncMock(
                return_value=TEST_AUTOMATION.model_copy(update={"enabled": True})
            ),
        )

        out = await _run(block, automation_id="aut_01", enabled=True)

        assert out["enabled"] is True
