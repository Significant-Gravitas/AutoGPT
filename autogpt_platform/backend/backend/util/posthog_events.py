"""Every PostHog event name the backend sends, in one place.

``docs/platform/tracking-plan.md`` says what each event means, which side
sends it and which properties it carries. This module is the code half of
that plan: emitters take the name from here instead of spelling it out.
The names follow the product analytics plan ("Every Second Counts"). The
funnel ``data_index`` keys (``briefing_generated:<id>``, ...) embed the name
as a string, so a rename has to update those too.

Never change the value of a member of ``PostHogEvent``. PostHog stores the
raw string, so a rename orphans every insight, funnel and cohort built on the
old name, and past events cannot be backfilled. The one-time move to the
analytics plan's names (SECRT-2722) is the exception, and the old names are
reserved in ``posthog_events_test.py``, which pins the values so an
accidental rename fails CI.
"""

from enum import StrEnum


class PostHogEvent(StrEnum):
    """Events the backend emits today."""

    # Activation: backend/util/product_analytics.py
    AGENT_RUN_STARTED = "agent_run_started"
    AGENT_RUN_FINISHED = "agent_run_finished"
    CHAT_MESSAGE_SENT = "chat_message_sent"
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

    # Chat (Autopilot): backend/copilot/tracking.py
    CHAT_TOOL_CALLED = "chat_tool_called"
    CHAT_OUTCOME = "chat_outcome"
    CHAT_LIBRARY_CHECK_OUTCOME = "chat_library_check_outcome"

    # Billing: backend/data/credit.py
    TOPUP_COMPLETED = "topup_completed"
    SUBSCRIPTION_CANCELLATION_SCHEDULED = "subscription_cancellation_scheduled"
    SUBSCRIPTION_CHANGED = "subscription_changed"
    PAYMENT_SUCCEEDED = "payment_succeeded"
    SUBSCRIPTION_TIER_RECONCILED = "subscription_tier_reconciled"

    # Trial lifecycle: backend/notifications/trial.py
    TRIAL_STARTED = "trial_started"
    TRIAL_ENDING = "trial_ending"
    TRIAL_CANCELED = "trial_canceled"
    TRIAL_RESUMED = "trial_resumed"
    TRIAL_ENDED = "trial_ended"
    TRIAL_CONVERTED = "trial_converted"
    PAYMENT_FAILED = "payment_failed"

    # Key moments (SECRT-2723): backend/util/product_analytics.py
    SIGNUP_COMPLETED = "signup_completed"
    ONBOARDING_COMPLETED = "onboarding_completed"
    CHECKOUT_STARTED = "checkout_started"
    SUBSCRIPTION_ENDED = "subscription_ended"
    LISTING_ADDED_TO_LIBRARY = "listing_added_to_library"
    LISTING_DOWNLOADED = "listing_downloaded"


class PlannedPostHogEvent(StrEnum):
    """Planned in the tracking plan and NOT emitted yet.

    Move a member into ``PostHogEvent`` in the change that starts sending it.
    """
