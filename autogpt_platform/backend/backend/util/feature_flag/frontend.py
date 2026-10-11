"""The flags the browser reads, evaluated here so it gets the backend's answer."""

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from backend.util.cache import cached

from . import (
    Flag,
    _evaluate_boolean_flag,
    _evaluate_flag_value,
    _flag_backend_configured,
)

# The key `create_feature_flag_dependency` evaluates signed-out callers under.
ANONYMOUS_USER = "anonymous"


class FrontendFlag(str, Enum):
    """Every flag the frontend reads; the frontend's `Flag` is generated from this.

    A key the backend reads too comes from the backend's `Flag`, so it is written once.
    """

    MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms"
    ENABLE_PLATFORM_PAYMENT = Flag.ENABLE_PLATFORM_PAYMENT.value
    ARTIFACTS_PAGE = "artifacts-page"
    CHAT_MODE_OPTION = Flag.CHAT_MODE_OPTION.value
    GENERIC_TRIGGER_AGENTS = Flag.GENERIC_TRIGGER_AGENTS.value
    CHAT_SEARCH = "chat-search"
    AUTOGPT_NEW_LAYOUT = "autogpt-new-layout"
    CHAT_WORKSPACE_FILES = "chat-workspace-files"
    CHAT_PINNING = "chat-pinning"
    TASK_PROGRESS_BAR = "task-progress-bar"
    HIRE_EXPERTS = Flag.HIRE_EXPERTS.value
    SKILLS_HUB = Flag.SKILLS_HUB.value
    # Until this is on for everyone, /profile/settings stays un-redirected
    # (`useNewSettingsRedirect`) so the toggles stay reachable: flip both together.
    SETTINGS_NOTIFICATIONS = "settings-notifications"
    ONBOARDING_BRAIN_DUMP = Flag.ONBOARDING_BRAIN_DUMP.value
    ONBOARDING_EXPERT_TEAM = Flag.ONBOARDING_EXPERT_TEAM.value
    GRAPHITI_MEMORY = Flag.GRAPHITI_MEMORY.value
    GRAPHITI_COMMUNITIES_ENABLED = Flag.GRAPHITI_COMMUNITIES_ENABLED.value
    DREAM_PASS_ENABLED = Flag.DREAM_PASS_ENABLED.value
    DREAM_PASS_INVALIDATE_ENTITY = Flag.DREAM_PASS_INVALIDATE_ENTITY.value
    # Platform key -> visible on the Bots settings page; only an explicit false hides one.
    COPILOT_BOT_PLATFORMS = "copilot-bot-platforms"
    COPILOT_VOICE_MODE = Flag.COPILOT_VOICE_MODE.value
    COPILOT_AUTO_MODE = Flag.COPILOT_AUTO_MODE.value
    # The share of chat sessions whose stream is shadowed and checked, 0 to 1.
    COPILOT_STREAM_SHADOW = "copilot-stream-shadow"
    COPILOT_STREAM_RUNTIME = "copilot-stream-runtime"

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Orval names the generated const's members from this, so `Flag.X` keeps its names.
        schema = handler(core_schema)
        handler.resolve_ref_schema(schema)["x-enum-varnames"] = [m.name for m in cls]
        return schema


# Served as payloads, so an env override (a bare true/false) never stands in for them.
PAYLOAD_FLAGS = frozenset(
    {
        FrontendFlag.MARKETPLACE_SEARCH_TERMS,
        FrontendFlag.COPILOT_BOT_PLATFORMS,
        FrontendFlag.COPILOT_STREAM_SHADOW,
    }
)


class FrontendFlagValues(BaseModel):
    flags: dict[FrontendFlag, Any] = Field(
        description="Each flag the vendor answered; one left out takes the frontend's default."
    )
    configured: bool = Field(
        description="False when no flag vendor is configured, so every flag is its default."
    )


async def evaluate_frontend_flags(user_id: str | None) -> FrontendFlagValues:
    """The frontend's flags for *user_id*, read exactly as the backend reads them."""
    flags = await _anonymous_flags() if user_id is None else await _evaluate(user_id)
    return FrontendFlagValues(flags=flags, configured=_flag_backend_configured())


@cached(maxsize=1, ttl_seconds=30)
async def _anonymous_flags() -> dict[FrontendFlag, Any]:
    # Every signed-out visitor gets the same answer, so one read serves them all.
    return await _evaluate(ANONYMOUS_USER)


async def _evaluate(user_id: str) -> dict[FrontendFlag, Any]:
    results = await asyncio.gather(*(_read(flag, user_id) for flag in FrontendFlag))
    return {
        flag: value
        for flag, (value, answered) in zip(FrontendFlag, results)
        if answered and value is not None
    }


async def _read(flag: FrontendFlag, user_id: str) -> tuple[Any, bool]:
    if flag in PAYLOAD_FLAGS:
        return await _evaluate_flag_value(flag.value, user_id)
    return await _evaluate_boolean_flag(flag.value, user_id)
