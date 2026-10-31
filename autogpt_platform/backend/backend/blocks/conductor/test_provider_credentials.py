import runpy
from pathlib import Path

import pytest

from backend.sdk.registry import AutoRegistry


def test_provider_ignores_server_key_and_accepts_user_api_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CONDUCTOR_API_KEY", "test-server-key-must-not-be-shared")
    monkeypatch.setattr(AutoRegistry, "_providers", {})
    monkeypatch.setattr(AutoRegistry, "_default_credentials", [])
    monkeypatch.setattr(AutoRegistry, "_api_key_mappings", {})

    runpy.run_path(str(Path(__file__).with_name("_config.py")))

    provider = AutoRegistry.get_provider("conductor")
    assert provider is not None
    assert provider.supported_auth_types == {"api_key"}
    assert provider.default_credentials == []
    assert AutoRegistry.get_all_credentials() == []
    assert "conductor" not in AutoRegistry._api_key_mappings
