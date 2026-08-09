"""The live integration registry, as prompt context.

Both LLM jobs need to know what this platform can actually connect to:
the recommender picks from the list, and the greeting needs it so the
automations it proposes are ones we can really run. Shared here so the
two can never drift onto different views of the registry.
"""

import logging

from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
)
from backend.blocks import load_all_blocks

logger = logging.getLogger(__name__)


def known_providers() -> dict[str, str | None]:
    """The live provider registry as ``{id: description}``.

    Mirrors the ``/providers`` endpoint: block modules must be imported
    before AutoRegistry knows about SDK-registered providers.
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
