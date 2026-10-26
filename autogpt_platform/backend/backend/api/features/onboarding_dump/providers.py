"""The live integration registry, as prompt context.

Both LLM jobs need to know what this platform can actually connect to:
the recommender picks from the list, and the greeting needs it so the
automations it proposes are ones we can really run. Shared here so the
two can never drift onto different views of the registry.
"""

import logging
from typing import TypeVar

from backend.api.features.integrations import support
from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
)
from backend.api.features.onboarding_dump.models import SuggestedPrompt
from backend.blocks import load_all_blocks

logger = logging.getLogger(__name__)

PromptT = TypeVar("PromptT", bound=SuggestedPrompt)


def known_providers() -> dict[str, str | None]:
    """The live provider registry as ``{id: description}``.

    Mirrors the ``/providers`` endpoint: block modules must be imported
    before AutoRegistry knows about SDK-registered providers.
    ``load_all_blocks`` is ``@cached``, so the two jobs that share this
    helper also share one import pass rather than each paying for it.
    """
    try:
        load_all_blocks()
    except Exception as e:  # static providers still work
        logger.warning("Brain dump: block load failed: %s", e)
    return {name: get_provider_description(name) for name in get_all_provider_names()}


def provider_lines(providers: dict[str, str | None]) -> str:
    """``providers`` as one ``- id: description`` line each."""
    return "\n".join(
        f"- {name}: {description}" if description else f"- {name}"
        for name, description in providers.items()
    )


def connectable_lines() -> str:
    """Every service a user can connect here, one ``- id: description`` each.

    What the greeting may promise: a provider whose sign-in is not set up in
    this deployment is left out, and the official MCP servers Otto connects
    are in.
    """
    return provider_lines(
        {service.key: service.description for service in support.connectable_services()}
    )


def label_support(prompts: list[PromptT]) -> list[PromptT]:
    """Each prompt with ``unsupported`` worked out against the live registry,
    from what it declares and what its text names alike."""
    services = support.connectable_services()
    for prompt in prompts:
        prompt.unsupported = support.unsupported_in_task(
            prompt.integrations, f"{prompt.title}\n{prompt.prompt}", services
        )
    return prompts
