"""Managed credential providers.

Call :func:`register_all` at application startup (e.g. in ``rest_api.py``)
to populate the provider registry before any requests are processed.
"""

from backend.integrations.managed_credentials import (
    get_managed_provider,
    register_managed_provider,
)
from backend.integrations.managed_providers.agentmail import AgentMailManagedProvider


def register_all() -> None:
    """Register every built-in managed credential provider (idempotent)."""
    if get_managed_provider(AgentMailManagedProvider.provider_name) is None:
        register_managed_provider(AgentMailManagedProvider())
