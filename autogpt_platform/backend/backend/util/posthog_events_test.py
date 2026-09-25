"""Pins every PostHog event name the backend sends.

A failure here means an event name changed. PostHog cannot backfill a rename,
so every insight built on the old name silently goes empty. Add new members
freely; never edit an existing string. The names follow the product analytics
plan; the pre-plan names are reserved below.
"""

from backend.util.posthog_events import PlannedPostHogEvent, PostHogEvent

LIVE_EVENT_NAMES = {
    "agent_run_started",
    "agent_run_finished",
    "chat_message_sent",
    "schedule_created",
    "schedule_fired",
    "trigger_fired",
    "expert_hired",
    "integration_connected",
    "hire_failed",
    "writing_style_added",
    "workflow_installed_on_expert",
    "expert_fired",
    "briefing_generated",
    "briefing_delivered",
    "chat_tool_called",
    "chat_outcome",
    "chat_library_check_outcome",
    "topup_completed",
    "subscription_cancellation_scheduled",
    "subscription_changed",
    "payment_succeeded",
    "subscription_tier_reconciled",
    "trial_started",
    "trial_ending",
    "trial_canceled",
    "trial_resumed",
    "trial_ended",
    "trial_converted",
    "payment_failed",
}

PLANNED_EVENT_NAMES = {
    "signup_completed",
    "onboarding_completed",
    "checkout_started",
    "subscription_ended",
    "listing_added_to_library",
    "listing_downloaded",
}

# No longer sent, or renamed (SECRT-2722). The names stay reserved: reusing one would
# splice a different action onto the history PostHog already holds for it.
RETIRED_EVENT_NAMES = {
    "hire_completed",
    "expert_run_completed",
    "copilot_message_sent",
    "copilot_agent_run_success",
    "copilot_agent_scheduled",
    "copilot_followup_scheduled",
    "copilot_trigger_setup",
    # Renamed to the analytics plan's names (SECRT-2722).
    "run_agent",
    "run_autopilot",
    "run_expert",
    "agent_run_completed",
    "agent_run_failed",
    "copilot_tool_called",
    "copilot_library_check_outcome",
    "credit_topup_success",
    "subscription_upgraded",
    "subscription_payment_success",
    "subscription_tier_reconciliation_discrepancy",
    "subscription_trial_started",
    "subscription_trial_ending",
    "subscription_trial_canceled",
    "subscription_trial_resumed",
    "subscription_trial_ended",
    "subscription_trial_converted",
    "subscription_trial_payment_failed",
}


def test_live_event_names_are_pinned():
    assert {event.value for event in PostHogEvent} == LIVE_EVENT_NAMES


def test_planned_event_names_are_pinned():
    assert {event.value for event in PlannedPostHogEvent} == PLANNED_EVENT_NAMES


def test_a_planned_event_is_not_also_live():
    assert not LIVE_EVENT_NAMES & PLANNED_EVENT_NAMES


def test_a_retired_event_name_is_never_reused():
    assert not RETIRED_EVENT_NAMES & (LIVE_EVENT_NAMES | PLANNED_EVENT_NAMES)
