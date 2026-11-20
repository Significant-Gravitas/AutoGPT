import runpy
from pathlib import Path

import pytest

from backend.sdk.registry import AutoRegistry


def test_provider_registers_no_default_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AGENTMAIL_API_KEY", "org-key")
    monkeypatch.setattr(AutoRegistry, "_providers", {})
    monkeypatch.setattr(AutoRegistry, "_default_credentials", [])
    monkeypatch.setattr(AutoRegistry, "_api_key_mappings", {})

    runpy.run_path(str(Path(__file__).with_name("_config.py")))

    provider = AutoRegistry.get_provider("agent_mail")
    assert provider is not None
    assert provider.supported_auth_types == {"api_key"}
    assert provider.default_credentials == []
    assert AutoRegistry.get_all_credentials() == []
    assert "agent_mail" not in AutoRegistry._api_key_mappings
