"""The flags the browser reads, evaluated here so it gets the backend's answer."""

import asyncio
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from backend.util.cache import cached

from . import (
    ANONYMOUS_USER,
    Flag,
    _env_flag_override,
    _evaluate_boolean_flag,
    _evaluate_flag_value,
    _flag_backend_configured,
    _flag_backend_gave_up,
    defined_flag_keys,
    evaluate_flags_in_memory,
)

# A consenting visitor's analytics id (a UUID); anything else counts as no id.
VISITOR_ID = re.compile(r"[A-Za-z0-9-]{8,64}")
# A visitor's flag key; no user id has a colon, so it can never target a user.
VISITOR_KEY = "visitor:{}"
# How long a signed-out answer missing a flag is served before the vendor is asked again.
PARTIAL_TTL_SECONDS = 5


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
        description="Each flag the vendor answered, or null for one it has no definition "
        "of, which takes the frontend's default; one left out has no answer yet."
    )
    configured: bool = Field(
        description="False when no flag vendor is configured or it gave up starting, "
        "so every flag is its default."
    )


async def evaluate_frontend_flags(
    user_id: str | None, visitor_id: str | None = None
) -> FrontendFlagValues:
    """The frontend's flags for the caller, read exactly as the backend reads them.

    A signed-out caller is bucketed by *visitor_id*, the analytics id its page sends
    only with consent, wherever the vendor answers in memory; otherwise it reads as
    `ANONYMOUS_USER`. A signed-in caller ignores it.
    """
    if user_id is not None:
        flags = await _evaluate(user_id)
    elif visitor_id is not None and VISITOR_ID.fullmatch(visitor_id):
        flags = await _visitor_flags(visitor_id)
    else:
        flags = await _anonymous_flags()
    # A vendor that gave up for good serves every default, as the route gates do;
    # one still connecting leaves its flags out, so the page asks again.
    configured = _flag_backend_configured() and not _flag_backend_gave_up()
    return FrontendFlagValues(flags=flags, configured=configured)


@cached(maxsize=1, ttl_seconds=30)
async def _anonymous_flags() -> dict[FrontendFlag, Any]:
    # Every signed-out visitor gets the same answer, so one read serves them all.
    flags = await _evaluate(ANONYMOUS_USER)
    if len(flags) < len(FrontendFlag):
        # A vendor that is down or starting is asked again soon; one read per
        # window still answers the whole flood of visitors meanwhile.
        asyncio.get_running_loop().call_later(
            PARTIAL_TTL_SECONDS, _anonymous_flags.cache_clear
        )
    return flags


# Listing LaunchDarkly's flags evaluates all of them; "cannot tell yet" is retried.
@cached(maxsize=1, ttl_seconds=30, cache_none=False)
def _defined_flags() -> frozenset[str] | None:
    return defined_flag_keys()


async def _visitor_flags(visitor_id: str) -> dict[FrontendFlag, Any]:
    # The caller picks the id, so its own bucket is only what the vendor answers in
    # memory; everything else is the one shared signed-out read.
    own = evaluate_flags_in_memory(
        [flag.value for flag in FrontendFlag], VISITOR_KEY.format(visitor_id)
    )
    flags = dict(await _anonymous_flags())
    for flag in FrontendFlag:
        if _takes_own_answer(flag, own.get(flag.value)):
            flags[flag] = own[flag.value]
    return flags


async def _evaluate(user_id: str) -> dict[FrontendFlag, Any]:
    defined = _defined_flags()
    # A flag the vendor has no definition of is never read (on PostHog that is a
    # /flags call each) and is answered None, so the page stops waiting for it.
    readable = [f for f in FrontendFlag if defined is None or f.value in defined]
    results = await asyncio.gather(*(_read(flag, user_id) for flag in readable))
    undefined = {flag: _forced(flag) for flag in FrontendFlag if flag not in readable}
    return undefined | {
        flag: value
        for flag, (value, answered) in zip(readable, results)
        if answered and value is not None
    }


async def _read(flag: FrontendFlag, user_id: str) -> tuple[Any, bool]:
    if flag in PAYLOAD_FLAGS:
        return await _evaluate_flag_value(flag.value, user_id)
    return await _evaluate_boolean_flag(flag.value, user_id)


def _forced(flag: FrontendFlag) -> bool | None:
    return None if flag in PAYLOAD_FLAGS else _env_flag_override(flag.value)


def _takes_own_answer(flag: FrontendFlag, value: Any) -> bool:
    # A switch forced by env stays forced, as in the shared read; a payload never is.
    return value is not None and (
        flag in PAYLOAD_FLAGS or _env_flag_override(flag.value) is None
    )
