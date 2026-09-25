"""Tests for the activation event vocabulary and its emitters."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from backend.data.experiments import ExperimentAssignment
from backend.util import posthog_client, product_analytics
from backend.util.posthog_events import PostHogEvent


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)
    return client.capture


def _only_call(capture: Mock) -> tuple[str, dict]:
    assert capture.call_count == 1
    kwargs = capture.call_args.kwargs
    return kwargs["event"], kwargs["properties"]


def test_track_is_a_noop_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: None)
    product_analytics.track("user-1", PostHogEvent.AGENT_RUN_STARTED, {"graph_id": "g"})


def test_track_is_a_noop_without_user(capture: Mock) -> None:
    product_analytics.track(None, PostHogEvent.AGENT_RUN_STARTED, {"graph_id": "g"})
    capture.assert_not_called()


def test_track_adds_base_properties_and_drops_nulls(capture: Mock) -> None:
    product_analytics.track(
        "user-1", PostHogEvent.AGENT_RUN_STARTED, {"graph_id": "g", "expert_id": None}
    )

    event, properties = _only_call(capture)
    assert capture.call_args.kwargs["distinct_id"] == "user-1"
    assert event == "agent_run_started"
    assert properties["source"] == "platform"
    assert "environment" in properties
    assert properties["graph_id"] == "g"
    assert "expert_id" not in properties


def test_track_swallows_client_errors(capture: Mock) -> None:
    capture.side_effect = RuntimeError("posthog down")
    product_analytics.track("user-1", PostHogEvent.AGENT_RUN_STARTED)


@pytest.mark.parametrize("trigger", ["manual", "api", "copilot"])
def test_human_run_start_is_agent_run_started(capture: Mock, trigger: str) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger=trigger,
        trigger_ref="library",
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_started"
    assert properties["trigger"] == trigger
    assert properties["trigger_ref"] == "library"
    assert "kind" not in properties


def test_run_start_accepts_enum_trigger(capture: Mock) -> None:
    from backend.data.execution import ExecutionTrigger

    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger=ExecutionTrigger.MANUAL,
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_started"
    assert properties["trigger"] == "manual"


def test_expert_workflow_run_start_is_agent_run_started_with_expert(
    capture: Mock,
) -> None:
    product_analytics.track_agent_run_started(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        trigger="manual",
        expert_id="expert-1",
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_started"
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
    assert event == "agent_run_finished"
    assert properties["status"] == "completed"
    assert properties["trigger"] == "schedule"
    assert properties["cost_cents"] == 12
    assert properties["duration_seconds"] == 3.5
    assert properties["is_subgraph_run"] is False


def test_run_finished_is_deduplicated_per_run(capture: Mock) -> None:
    """A requeue after a failed status persist finishes the same run again,
    maybe with another status; both sends must share one event uuid."""
    from backend.data.execution import ExecutionStatus

    for status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
        product_analytics.track_agent_run_finished(
            user_id="user-1",
            graph_id="graph-1",
            graph_exec_id="exec-1",
            status=status,
            trigger="manual",
        )
    product_analytics.track_agent_run_finished(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="exec-2",
        status=ExecutionStatus.COMPLETED,
        trigger="manual",
    )

    first, retry, other = capture.call_args_list
    assert first.kwargs["uuid"] is not None
    assert first.kwargs["uuid"] == retry.kwargs["uuid"]
    assert first.kwargs["properties"]["$insert_id"] == "exec-1"
    assert other.kwargs["uuid"] != first.kwargs["uuid"]


@pytest.mark.parametrize(
    ("parent_execution_id", "is_subgraph_run"), [(None, False), ("parent-1", True)]
)
def test_run_finished_hook_tells_a_subgraph_run_from_a_top_level_one(
    capture: Mock, parent_execution_id: str | None, is_subgraph_run: bool
) -> None:
    """A top-level expert run is ``expert_id`` set and ``is_subgraph_run``
    false: the filter that replaced ``expert_run_completed``."""
    from backend.data.execution import ExecutionStatus

    graph_exec = Mock(user_id="user-1", graph_id="graph-1", graph_exec_id="exec-1")
    graph_exec.execution_context.parent_execution_id = parent_execution_id
    product_analytics.handle_run_finished(
        graph_exec,
        Mock(
            status=ExecutionStatus.FAILED,
            trigger_source="subgraph" if is_subgraph_run else "schedule",
            expert_id="expert-1",
        ),
        Mock(failure_reason=None, cost=5, walltime=1.0, is_dry_run=False),
    )

    event, properties = _only_call(capture)
    assert event == "agent_run_finished"
    assert properties["status"] == "failed"
    assert properties["expert_id"] == "expert-1"
    assert properties["is_subgraph_run"] is is_subgraph_run


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
    assert event == "agent_run_finished"
    assert properties["status"] == "failed"
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
    assert events == ["chat_message_sent", "chat_message_sent"]
    autopilot_props = capture.call_args_list[0].kwargs["properties"]
    expert_props = capture.call_args_list[1].kwargs["properties"]
    assert autopilot_props["surface"] == "chat"
    assert autopilot_props["kind"] == "chat_turn"
    assert "expert_id" not in autopilot_props
    assert "message_length" not in autopilot_props
    assert expert_props["surface"] == "slack"
    assert expert_props["expert_id"] == "expert-1"


def test_chat_turn_carries_the_message_length(capture: Mock) -> None:
    product_analytics.track_chat_turn(
        user_id="user-1", session_id="s1", message_length=42
    )

    event, properties = _only_call(capture)
    assert event == "chat_message_sent"
    assert properties["message_length"] == 42


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


def test_integration_connected(capture: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posthog_client, "_environment", lambda: "local")
    product_analytics.track_integration_connected(
        user_id="user-1",
        provider="github",
        credential_type="oauth2",
        method="oauth",
    )

    event, properties = _only_call(capture)
    assert event == "integration_connected"
    assert properties == {
        "environment": "local",
        "source": "platform",
        "provider": "github",
        "credential_type": "oauth2",
        "method": "oauth",
    }


def test_trigger_fired(capture: Mock) -> None:
    product_analytics.track_trigger_fired(
        user_id="user-1",
        webhook_id="wh-1",
        graph_id="graph-1",
        graph_exec_id="exec-1",
        expert_id="expert-1",
    )

    event, properties = _only_call(capture)
    assert event == "trigger_fired"
    assert properties["target"] == "expert"


def test_signup_completed_carries_the_auth_provider(capture: Mock) -> None:
    product_analytics.track_signup_completed(user_id="user-1", signup_method="google")

    event, properties = _only_call(capture)
    assert capture.call_args.kwargs["distinct_id"] == "user-1"
    assert event == "signup_completed"
    assert properties["signup_method"] == "google"


def test_signup_completed_without_a_known_provider_omits_it(capture: Mock) -> None:
    product_analytics.track_signup_completed(user_id="user-1", signup_method=None)

    _, properties = _only_call(capture)
    assert "signup_method" not in properties


def test_onboarding_completed(capture: Mock) -> None:
    product_analytics.track_onboarding_completed(user_id="user-1")

    event, properties = _only_call(capture)
    assert event == "onboarding_completed"
    assert properties["source"] == "platform"


@pytest.mark.asyncio
async def test_checkout_started_carries_plan_surface_and_experiment_arms(
    capture: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def arms(user_id: str) -> dict[str, str]:
        assert user_id == "user-1"
        return {"$feature/subscription-pricing-page-initial-state": "yearly-pro"}

    monkeypatch.setattr(product_analytics, "_experiment_arm_properties", arms)

    await product_analytics.track_checkout_started(
        user_id="user-1",
        checkout_kind="subscription",
        surface="onboarding",
        subscription_tier="PRO",
        billing_cycle="yearly",
    )

    event, properties = _only_call(capture)
    assert event == "checkout_started"
    assert properties["checkout_kind"] == "subscription"
    assert properties["surface"] == "onboarding"
    assert properties["subscription_tier"] == "PRO"
    assert properties["billing_cycle"] == "yearly"
    assert (
        properties["$feature/subscription-pricing-page-initial-state"] == "yearly-pro"
    )


@pytest.mark.asyncio
async def test_checkout_started_still_sends_when_arms_cannot_be_read(
    capture: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(user_id: str) -> dict[str, str]:
        raise RuntimeError("db down")

    monkeypatch.setattr(product_analytics, "_experiment_arm_properties", broken)

    await product_analytics.track_checkout_started(
        user_id="user-1", checkout_kind="top_up", surface="billing"
    )

    event, properties = _only_call(capture)
    assert event == "checkout_started"
    assert properties["checkout_kind"] == "top_up"
    assert "subscription_tier" not in properties
    assert not any(key.startswith("$feature/") for key in properties)


@pytest.mark.asyncio
async def test_experiment_arms_become_feature_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_assignments(user_id: str) -> list[ExperimentAssignment]:
        return [
            ExperimentAssignment(
                experiment_key="subscription-pricing-page-initial-state",
                variant="monthly-max",
                source="posthog",
                assigned_at=datetime.now(timezone.utc),
            )
        ]

    monkeypatch.setattr(product_analytics, "list_assignments", list_assignments)

    assert await product_analytics._experiment_arm_properties("user-1") == {
        "$feature/subscription-pricing-page-initial-state": "monthly-max"
    }


def test_subscription_ended(capture: Mock) -> None:
    product_analytics.track_subscription_ended(
        user_id="user-1",
        subscription_tier="MAX",
        billing_cycle="monthly",
        reason="payment_failed",
    )

    event, properties = _only_call(capture)
    assert event == "subscription_ended"
    assert properties["subscription_tier"] == "MAX"
    assert properties["billing_cycle"] == "monthly"
    assert properties["reason"] == "payment_failed"


def test_listing_added_to_library_and_downloaded(capture: Mock) -> None:
    product_analytics.track_listing_added_to_library(
        user_id="user-1",
        store_listing_version_id="slv-1",
        graph_id="graph-1",
        library_agent_id="lib-1",
    )
    product_analytics.track_listing_downloaded(
        user_id="user-1", store_listing_version_id="slv-1", graph_id="graph-1"
    )

    added, downloaded = capture.call_args_list
    assert added.kwargs["event"] == "listing_added_to_library"
    assert added.kwargs["properties"]["library_agent_id"] == "lib-1"
    assert downloaded.kwargs["event"] == "listing_downloaded"
    assert downloaded.kwargs["properties"]["store_listing_version_id"] == "slv-1"


def test_signed_out_download_is_not_tracked(capture: Mock) -> None:
    product_analytics.track_listing_downloaded(
        user_id=None, store_listing_version_id="slv-1", graph_id="graph-1"
    )

    capture.assert_not_called()
