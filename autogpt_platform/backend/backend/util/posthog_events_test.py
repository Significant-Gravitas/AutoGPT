"""Pins every PostHog event name the backend sends.

A failure here means an event name changed. PostHog cannot backfill a rename,
so every insight built on the old name silently goes empty. Add new members
freely; never edit an existing string.
"""

from backend.util.posthog_events import PlannedPostHogEvent, PostHogEvent

LIVE_EVENT_NAMES = {
    "run_agent",
    "run_autopilot",
    "run_expert",
    "agent_run_completed",
    "agent_run_failed",
    "schedule_created",
    "schedule_fired",
    "trigger_fired",
    "expert_hired",
    "integration_connected",
    "hire_failed",
    "hire_completed",
    "writing_style_added",
    "workflow_installed_on_expert",
    "expert_fired",
    "briefing_generated",
    "briefing_delivered",
    "expert_run_completed",
    "copilot_message_sent",
    "copilot_tool_called",
    "copilot_agent_run_success",
    "copilot_agent_scheduled",
    "copilot_followup_scheduled",
    "copilot_library_check_outcome",
    "copilot_trigger_setup",
    "credit_topup_success",
    "subscription_cancellation_scheduled",
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

PLANNED_EVENT_NAMES = {
    "signup_completed",
    "onboarding_completed",
    "checkout_started",
    "subscription_ended",
    "listing_added_to_library",
    "listing_downloaded",
}


def test_live_event_names_are_pinned():
    assert {event.value for event in PostHogEvent} == LIVE_EVENT_NAMES


def test_planned_event_names_are_pinned():
    assert {event.value for event in PlannedPostHogEvent} == PLANNED_EVENT_NAMES


def test_a_planned_event_is_not_also_live():
    assert not LIVE_EVENT_NAMES & PLANNED_EVENT_NAMES
