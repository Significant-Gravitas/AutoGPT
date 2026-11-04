"""A managed provider registers no ``<provider>-default`` block credential.

Each provider here uses an org-level key to provision a per-user key, so its
block ``ProviderBuilder`` declares it with ``with_managed_api_key()``.
"""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.blocks
from backend.integrations.managed_credentials import get_managed_providers
from backend.integrations.managed_providers import register_all
from backend.sdk import builder
from backend.sdk.registry import AutoRegistry

register_all()
MANAGED_PROVIDER_NAMES = sorted(get_managed_providers())
BLOCKS_DIR = Path(backend.blocks.__file__).parent


def test_managed_providers_are_discovered():
    assert {"agent_mail", "ayrshare"} <= set(MANAGED_PROVIDER_NAMES)


@pytest.mark.parametrize("provider_name", MANAGED_PROVIDER_NAMES)
def test_block_provider_registers_no_default_even_with_env_set(
    provider_name: str, monkeypatch: pytest.MonkeyPatch
):
    config_path = BLOCKS_DIR / provider_name / "_config.py"
    assert config_path.exists(), f"No block provider config for {provider_name}"

    # Every env var the builder reads is set, whatever its name.
    monkeypatch.setattr(
        builder, "os", SimpleNamespace(getenv=lambda *_args, **_kw: "org-key")
    )
    monkeypatch.setattr(AutoRegistry, "_providers", {})
    monkeypatch.setattr(AutoRegistry, "_default_credentials", [])
    monkeypatch.setattr(AutoRegistry, "_api_key_mappings", {})

    runpy.run_path(str(config_path))

    provider = AutoRegistry.get_provider(provider_name)
    assert provider is not None
    assert provider.default_credentials == []
    assert AutoRegistry.get_all_credentials() == []
    assert provider_name not in AutoRegistry._api_key_mappings
