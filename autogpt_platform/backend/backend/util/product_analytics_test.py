"""Tests for the activation event vocabulary and its emitters."""

from unittest.mock import Mock

import pytest

from backend.util import product_analytics
from backend.util.product_analytics import ActivationEvent


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(product_analytics, "get_posthog_client", lambda: client)
    return client.capture


def _only_call(capture: Mock) -> tuple[str, dict]:
    assert capture.call_count == 1
    kwargs = capture.call_args.kwargs
    return kwargs["event"], kwargs["properties"]


def test_track_is_a_noop_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(product_analytics, "get_posthog_client", lambda: None)
    product_analytics.track("user-1", ActivationEvent.RUN_AGENT, {"graph_id": "g"})


def test_track_is_a_noop_without_user(capture: Mock) -> None:
    product_analytics.track(None, ActivationEvent.RUN_AGENT, {"graph_id": "g"})
    capture.assert_not_called()


def test_track_adds_base_properties_and_drops_nulls(capture: Mock) -> None:
    product_analytics.track(
        "user-1", ActivationEvent.RUN_AGENT, {"graph_id": "g", "expert_id": None}
    )

    event, properties = _only_call(capture)
    assert capture.call_args.kwargs["distinct_id"] == "user-1"
    assert event == "run_agent"
    assert properties["source"] == "platform"
    assert "environment" in properties
    assert properties["graph_id"] == "g"
    assert "expert_id" not in properties


def test_track_swallows_client_errors(capture: Mock) -> None:
    capture.side_effect = RuntimeError("posthog down")
    product_analytics.track("user-1", ActivationEvent.RUN_AGENT)


@pytest.mark.parametrize("trigger", ["manual", "api", "copilot"])
def test_human_run_start_is_run_agent(capture: Mock, trigger: str) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger=trigger,
        trigger_ref="library",
    )

    event, properties = _only_call(capture)
    assert event == "run_agent"
    assert properties["trigger"] == trigger
    assert properties["trigger_ref"] == "library"


def test_run_start_accepts_enum_trigger(capture: Mock) -> None:
    from backend.data.execution import ExecutionTrigger

    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger=ExecutionTrigger.MANUAL,
    )

    event, properties = _only_call(capture)
    assert event == "run_agent"
    assert properties["trigger"] == "manual"


def test_expert_workflow_run_start_is_run_expert(capture: Mock) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger="manual",
        expert_id="expert-1",
    )

    event, properties = _only_call(capture)
    assert event == "run_expert"
    assert properties["kind"] == "workflow_run"
    assert properties["expert_id"] == "expert-1"


@pytest.mark.parametrize("trigger", ["schedule", "webhook", "subgraph", "admin", None])
def test_non_human_run_start_emits_nothing(capture: Mock, trigger: str | None) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1", graph_id="graph-1", graph_exec_id="exec-1", trigger=trigger
    )
    capture.assert_not_called()


def test_dry_run_start_emits_nothing(capture: Mock) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger="manual",
        is_dry_run=True,
    )
    capture.assert_not_called()


def test_run_finished_completed(capture: Mock) -> None:
    from backend.data.execution import ExecutionStatus

    product_analytics.track_agent_run_finished(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        status=ExecutionStatus.COMPLETED,
        trigger="schedule",
        cost_cents=12,
        duration_seconds=3.5,
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_completed"
    assert properties["trigger"] == "schedule"
    assert properties["cost_cents"] == 12
    assert properties["duration_seconds"] == 3.5


def test_run_finished_failed_carries_failure_reason(capture: Mock) -> None:
    from backend.data.execution import ExecutionStatus
    from backend.util.exceptions import ExecutionFailureReason

    product_analytics.track_agent_run_finished(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        status=ExecutionStatus.FAILED,
        trigger="manual",
        failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_failed"
    assert properties["failure_reason"] == "insufficient_balance"


def test_run_finished_terminated_or_dry_run_emits_nothing(capture: Mock) -> None:
    from backend.data.execution import ExecutionStatus

    product_analytics.track_agent_run_finished(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        status=ExecutionStatus.TERMINATED,
        trigger="manual",
    )
    product_analytics.track_agent_run_finished(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        status=ExecutionStatus.COMPLETED,
        trigger="manual",
        is_dry_run=True,
    )
    capture.assert_not_called()


def test_chat_turn_autopilot_vs_expert(capture: Mock) -> None:
    product_analytics.track_chat_turn(user_id="user-1", session_id="s1")
    product_analytics.track_chat_turn(
        user_id="user-1", session_id="s2", expert_id="expert-1", surface="slack"
    )

    events = [c.kwargs["event"] for c in capture.call_args_list]
    assert events == ["run_autopilot", "run_expert"]
    autopilot_props = capture.call_args_list[0].kwargs["properties"]
    expert_props = capture.call_args_list[1].kwargs["properties"]
    assert autopilot_props["surface"] == "chat"
    assert autopilot_props["kind"] == "chat_turn"
    assert expert_props["surface"] == "slack"
    assert expert_props["expert_id"] == "expert-1"


def test_automation_chat_turn_emits_nothing(capture: Mock) -> None:
    product_analytics.track_chat_turn(
        user_id="user-1", session_id="s1", origin="automation"
    )
    capture.assert_not_called()


def test_schedule_target() -> None:
    assert (
        product_analytics.schedule_target(expert_id=None, is_copilot_turn=False)
        == "agent"
    )
    assert (
        product_analytics.schedule_target(expert_id=None, is_copilot_turn=True)
        == "autopilot"
    )
    assert (
        product_analytics.schedule_target(expert_id="e", is_copilot_turn=True)
        == "expert"
    )
    assert (
        product_analytics.schedule_target(expert_id="e", is_copilot_turn=False)
        == "expert"
    )


def test_schedule_created_and_fired(capture: Mock) -> None:
    product_analytics.track_schedule_created(
        user_id="user-1",
        schedule_id="sched-1",
        target="autopilot",
        cron=None,
    )
    product_analytics.track_schedule_fired(
        user_id="user-1", schedule_id="sched-1", target="agent", graph_exec_id="exec-1"
    )

    created, fired = capture.call_args_list
    assert created.kwargs["event"] == "schedule_created"
    assert created.kwargs["properties"]["is_recurring"] is False
    assert created.kwargs["properties"]["target"] == "autopilot"
    assert fired.kwargs["event"] == "schedule_fired"
    assert fired.kwargs["properties"]["graph_exec_id"] == "exec-1"


def test_integration_connected(capture: Mock) -> None:
    product_analytics.track_integration_connected(
        user_id="user-1",
        provider="github",
        credential_type="oauth2",
        method="oauth",
    )

    event, properties = _only_call(capture)
    assert event == "integration_connected"
    assert properties == {
        **{k: v for k, v in properties.items() if k in ("environment", "source")},
        "provider": "github",
        "credential_type": "oauth2",
        "method": "oauth",
    }


def test_trigger_fired_and_expert_hired(capture: Mock) -> None:
    product_analytics.track_trigger_fired(
        user_id="user-1",
        webhook_id="wh-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        expert_id="expert-1",
    )
    product_analytics.track_expert_hired(
        user_id="user-1", expert_id="expert-1", template_id="tmpl-1", name="Maria"
    )

    trigger, hired = capture.call_args_list
    assert trigger.kwargs["event"] == "trigger_fired"
    assert trigger.kwargs["properties"]["target"] == "expert"
    assert hired.kwargs["event"] == "expert_hired"
    assert hired.kwargs["properties"]["template_id"] == "tmpl-1"
