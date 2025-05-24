import os

import pytest

from backend.data.model import UserIntegrations
from backend.integrations import credentials_store as cs


@pytest.mark.skip(reason="Settings caching makes environment-based testing unreliable")
def test_discover_default_credentials_env():
    # Use os.environ directly since monkeypatch doesn't work with Settings
    original_value = os.environ.get("FAL_API_KEY")
    try:
        os.environ["FAL_API_KEY"] = "test-key"
        cs.discover_default_credentials.cache_clear()
        creds = cs.discover_default_credentials()
        assert any(
            c.provider == "fal" and c.api_key.get_secret_value() == "test-key"
            for c in creds
        )
    finally:
        if original_value is None:
            os.environ.pop("FAL_API_KEY", None)
        else:
            os.environ["FAL_API_KEY"] = original_value


class DummyStore(cs.IntegrationCredentialsStore):
    def __init__(self):
        pass

    def _get_user_integrations(self, user_id: str) -> UserIntegrations:
        return UserIntegrations()


@pytest.mark.skip(reason="Settings caching makes environment-based testing unreliable")
def test_get_all_creds_includes_discovered():
    # Use os.environ directly since monkeypatch doesn't work with Settings
    original_value = os.environ.get("FAL_API_KEY")
    try:
        os.environ["FAL_API_KEY"] = "test-key"
        cs.discover_default_credentials.cache_clear()

        # Reload the module to pick up the new environment variable
        import importlib

        importlib.reload(cs)

        store = DummyStore()
        creds = store.get_all_creds("user")
        assert any(c.provider == "fal" for c in creds)
    finally:
        if original_value is None:
            os.environ.pop("FAL_API_KEY", None)
        else:
            os.environ["FAL_API_KEY"] = original_value
