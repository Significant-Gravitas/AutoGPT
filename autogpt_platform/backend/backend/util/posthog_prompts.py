"""The PostHog Prompts client our managed prompts are read with.

Two prompts are managed in PostHog (LLM analytics > Prompts), so they can be
edited without a deploy: the AutoPilot system prompt (``copilot/service.py``)
and the onboarding greeting (``api/features/onboarding_dump/intro.py``). Each
has a copy in code, served whenever PostHog is not configured or cannot be
reached, so neither a chat turn nor onboarding depends on PostHog.

Reading prompts takes a personal API key (``POSTHOG_PROMPTS_API_KEY``), not the
project key events are sent with. The project token picks the project the
prompts are read from (``POSTHOG_PROMPTS_PROJECT_API_KEY``, else
``POSTHOG_API_KEY``), so every environment can read one project's prompts and
pick its version by label (:func:`prompt_label`).

A prompt may include others by reference (``@@@prompt:name=...@@@``). PostHog
resolves them before it answers, so what a fetch returns is the assembled text.
"""

import asyncio
import functools
from typing import Any, Literal

from posthog.ai.prompts import PromptResult, Prompts, PromptVariables
from pydantic import BaseModel, ConfigDict

from backend.util.settings import AppEnvironment, Secrets, Settings

PRODUCTION_LABEL = "production"
# PostHog reserves this label for the newest version and refuses it as a query
# parameter; the newest version is what a fetch without a label returns.
LATEST_LABEL = "latest"


class PromptRef(BaseModel):
    """Which copy of a managed prompt a model call ran on, for LLM analytics.

    ``version`` is the PostHog version, or None for the copy in code. A
    version counts edits to the prompt itself: a prompt it references can
    change under the same version.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: int | None = None

    @property
    def source(self) -> Literal["posthog", "code"]:
        return "code" if self.version is None else "posthog"

    def event_properties(self) -> dict[str, Any]:
        """The properties PostHog links an LLM analytics event to a prompt by."""
        properties: dict[str, Any] = {
            "$ai_prompt_name": self.name,
            "prompt_source": self.source,
        }
        if self.version is not None:
            properties["$ai_prompt_version"] = self.version
        return properties

    def broadcast_trace(self) -> dict[str, Any]:
        """Keys for an OpenRouter request's ``trace``.

        Broadcast forwards them as ``metadata_prompt_name``,
        ``metadata_prompt_version`` and ``metadata_prompt_source``: it maps no
        request field to ``$ai_prompt_name``.
        """
        trace: dict[str, Any] = {"prompt_name": self.name, "prompt_source": self.source}
        if self.version is not None:
            trace["prompt_version"] = self.version
        return trace


def posthog_prompts_configured() -> bool:
    """Whether a key to read prompts with is set."""
    secrets = Settings().secrets
    return bool(secrets.posthog_prompts_api_key and _project_api_key(secrets))


@functools.cache
def get_posthog_prompts_client() -> Prompts:
    """The process's PostHog Prompts client."""
    secrets = Settings().secrets
    return Prompts(
        personal_api_key=secrets.posthog_prompts_api_key,
        project_api_key=_project_api_key(secrets),
        host=secrets.posthog_prompts_host,
    )


def prompt_label() -> str | None:
    """The label prompts are read at, or None for the newest version."""
    config = Settings().config
    label = config.posthog_prompts_label.strip()
    if not label:
        is_production = config.app_env == AppEnvironment.PRODUCTION
        label = PRODUCTION_LABEL if is_production else LATEST_LABEL
    return None if label == LATEST_LABEL else label


async def fetch_prompt(
    name: str, *, cache_ttl_seconds: int, timeout_seconds: float
) -> PromptResult:
    """Read ``name`` at this environment's label.

    The SDK serves its own copy within ``cache_ttl_seconds`` and an expired copy
    when a refetch fails (``result.source`` says which). Raises when there is no
    copy to serve, or when the read takes longer than ``timeout_seconds``.
    """
    client = get_posthog_prompts_client()
    label = prompt_label()

    def get() -> PromptResult:
        return client.get(
            name,
            with_metadata=True,
            cache_ttl_seconds=cache_ttl_seconds,
            label=label,
        )

    return await asyncio.wait_for(asyncio.to_thread(get), timeout=timeout_seconds)


def compile_prompt(template: str, variables: PromptVariables) -> str:
    """Fill ``{{name}}`` placeholders; the ones not in ``variables`` stay."""
    return get_posthog_prompts_client().compile(template, variables)


def _project_api_key(secrets: Secrets) -> str:
    return secrets.posthog_prompts_project_api_key or secrets.posthog_api_key
