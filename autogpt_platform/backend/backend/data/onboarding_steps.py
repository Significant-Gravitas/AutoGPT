from enum import StrEnum
from typing import Literal


class OnboardingStep(StrEnum):
    """Application-level onboarding step identifiers.

    Stored as plain strings in UserOnboarding.{completedSteps,notified,rewardedFor}
    so adds/renames/retires are code-only. Boundary validation lives on the
    completion endpoint via ``FrontendOnboardingStep`` (a Pydantic ``Literal``).
    Legacy values that no longer appear here remain readable from existing rows
    as inert strings.

    Lives in its own module (rather than ``onboarding.py``) so the response
    model in ``backend.data.model`` can type its fields with this enum without
    importing the onboarding logic layer, which would create an import cycle.
    """

    # Introductory onboarding (Library)
    WELCOME = "WELCOME"
    USAGE_REASON = "USAGE_REASON"
    INTEGRATIONS = "INTEGRATIONS"
    AGENT_CHOICE = "AGENT_CHOICE"
    AGENT_NEW_RUN = "AGENT_NEW_RUN"
    AGENT_INPUT = "AGENT_INPUT"
    CONGRATS = "CONGRATS"
    # First Wins
    # ONBOARDING_COMPLETE is the wizard-completion signal that backs the
    # wallet's "Complete onboarding $3" tile. Renamed from VISIT_COPILOT in
    # SECRT-2355; existing rows are migrated in-place.
    ONBOARDING_COMPLETE = "ONBOARDING_COMPLETE"
    GET_RESULTS = "GET_RESULTS"
    MARKETPLACE_VISIT = "MARKETPLACE_VISIT"
    MARKETPLACE_ADD_AGENT = "MARKETPLACE_ADD_AGENT"
    MARKETPLACE_RUN_AGENT = "MARKETPLACE_RUN_AGENT"
    BUILDER_SAVE_AGENT = "BUILDER_SAVE_AGENT"
    # Consistency Challenge
    RE_RUN_AGENT = "RE_RUN_AGENT"
    SCHEDULE_AGENT = "SCHEDULE_AGENT"
    RUN_AGENTS = "RUN_AGENTS"
    RUN_3_DAYS = "RUN_3_DAYS"
    # The Pro Playground
    TRIGGER_WEBHOOK = "TRIGGER_WEBHOOK"
    RUN_14_DAYS = "RUN_14_DAYS"
    RUN_AGENTS_100 = "RUN_AGENTS_100"
    # No longer rewarded but exist for analytical purposes
    BUILDER_OPEN = "BUILDER_OPEN"
    BUILDER_RUN_AGENT = "BUILDER_RUN_AGENT"


FrontendOnboardingStep = Literal[
    OnboardingStep.WELCOME,
    OnboardingStep.USAGE_REASON,
    OnboardingStep.INTEGRATIONS,
    OnboardingStep.AGENT_CHOICE,
    OnboardingStep.AGENT_NEW_RUN,
    OnboardingStep.AGENT_INPUT,
    OnboardingStep.CONGRATS,
    OnboardingStep.ONBOARDING_COMPLETE,
    OnboardingStep.BUILDER_OPEN,
]
