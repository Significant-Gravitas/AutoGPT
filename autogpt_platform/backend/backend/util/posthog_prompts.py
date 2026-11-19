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
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from posthog.ai.prompts import PromptResult, Prompts, PromptVariables
from pydantic import BaseModel, ConfigDict

from backend.util.settings import AppEnvironment, Secrets, Settings

PRODUCTION_LABEL = "production"
# PostHog reserves this label for the newest version and refuses it as a query
# parameter; the newest version is what a fetch without a label returns.
LATEST_LABEL = "latest"

# The SDK's reads block, and one that stalls keeps retrying past our timeout.
# They get threads of their own, so a PostHog stall cannot take the default
# executor's from every other ``asyncio.to_thread`` caller.
_FETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="posthog-prompts"
)
# ``{{ name }}``, which Langfuse templates allowed; the SDK fills ``{{name}}`` only.
_SPACED_PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")


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
    copy to serve, when the read takes longer than ``timeout_seconds``, or when
    the copy is not at the label asked for.
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

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(_FETCH_EXECUTOR, get), timeout=timeout_seconds
    )
    # A server that ignores the label answers with the newest version and no
    # label; that is not the version this environment should run.
    if label is not None and result.label != label:
        raise RuntimeError(
            f"PostHog served prompt {name!r} at label {result.label!r}, not {label!r}"
        )
    return result


def compile_prompt(template: str, variables: PromptVariables) -> str:
    """Fill ``{{name}}`` placeholders, spaced (``{{ name }}``) or not; the ones
    not in ``variables`` stay as written."""

    def unspaced(match: re.Match[str]) -> str:
        name = match.group(1)
        return f"{{{{{name}}}}}" if name in variables else match.group(0)

    return get_posthog_prompts_client().compile(
        _SPACED_PLACEHOLDER.sub(unspaced, template), variables
    )


def _project_api_key(secrets: Secrets) -> str:
    return secrets.posthog_prompts_project_api_key or secrets.posthog_api_key
