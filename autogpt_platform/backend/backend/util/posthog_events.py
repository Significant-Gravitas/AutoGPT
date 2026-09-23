"""Every PostHog event name the backend sends, in one place.

``docs/platform/tracking-plan.md`` says what each event means, which side
sends it and which properties it carries. This module is the code half of
that plan: emitters take the name from here instead of spelling it out.

Never change the value of a member of ``PostHogEvent``. PostHog stores the
raw string, so a rename orphans every insight, funnel and cohort built on the
old name, and past events cannot be backfilled. ``posthog_events_test.py``
pins the values so an accidental rename fails CI.
"""

from enum import StrEnum


class PostHogEvent(StrEnum):
    """Events the backend emits today."""

    # Activation: backend/util/product_analytics.py
    RUN_AGENT = "run_agent"
    RUN_AUTOPILOT = "run_autopilot"
    RUN_EXPERT = "run_expert"
    AGENT_RUN_COMPLETED = "agent_run_completed"
    AGENT_RUN_FAILED = "agent_run_failed"
    SCHEDULE_CREATED = "schedule_created"
    SCHEDULE_FIRED = "schedule_fired"
    TRIGGER_FIRED = "trigger_fired"
    INTEGRATION_CONNECTED = "integration_connected"

    # Experts loop: backend/util/funnel_analytics.py
    EXPERT_HIRED = "expert_hired"
    HIRE_FAILED = "hire_failed"
    WRITING_STYLE_ADDED = "writing_style_added"
    WORKFLOW_INSTALLED_ON_EXPERT = "workflow_installed_on_expert"
    EXPERT_FIRED = "expert_fired"
    BRIEFING_GENERATED = "briefing_generated"
    BRIEFING_DELIVERED = "briefing_delivered"

    # Copilot: backend/copilot/tracking.py
    COPILOT_TOOL_CALLED = "copilot_tool_called"
    COPILOT_LIBRARY_CHECK_OUTCOME = "copilot_library_check_outcome"

    # Billing: backend/data/credit.py
    CREDIT_TOPUP_SUCCESS = "credit_topup_success"
    SUBSCRIPTION_CANCELLATION_SCHEDULED = "subscription_cancellation_scheduled"
    SUBSCRIPTION_UPGRADED = "subscription_upgraded"
    SUBSCRIPTION_PAYMENT_SUCCESS = "subscription_payment_success"
    SUBSCRIPTION_TIER_RECONCILIATION_DISCREPANCY = (
        "subscription_tier_reconciliation_discrepancy"
    )

    # Trial lifecycle: backend/notifications/trial.py
    SUBSCRIPTION_TRIAL_STARTED = "subscription_trial_started"
    SUBSCRIPTION_TRIAL_ENDING = "subscription_trial_ending"
    SUBSCRIPTION_TRIAL_CANCELED = "subscription_trial_canceled"
    SUBSCRIPTION_TRIAL_RESUMED = "subscription_trial_resumed"
    SUBSCRIPTION_TRIAL_ENDED = "subscription_trial_ended"
    SUBSCRIPTION_TRIAL_CONVERTED = "subscription_trial_converted"
    SUBSCRIPTION_TRIAL_PAYMENT_FAILED = "subscription_trial_payment_failed"


class PlannedPostHogEvent(StrEnum):
    """Planned in the tracking plan and NOT emitted yet (SECRT-2723).

    Move a member into ``PostHogEvent`` in the change that starts sending it.
    """

    SIGNUP_COMPLETED = "signup_completed"
    ONBOARDING_COMPLETED = "onboarding_completed"
    CHECKOUT_STARTED = "checkout_started"
    SUBSCRIPTION_ENDED = "subscription_ended"
    LISTING_ADDED_TO_LIBRARY = "listing_added_to_library"
    LISTING_DOWNLOADED = "listing_downloaded"
