from backend.data.model import UserIntegrations
from backend.integrations import credentials_store as cs


def test_discover_default_credentials_env(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    cs.discover_default_credentials.cache_clear()
    creds = cs.discover_default_credentials()
    assert any(
        c.provider == "fal" and c.api_key.get_secret_value() == "test-key"
        for c in creds
    )


class DummyStore(cs.IntegrationCredentialsStore):
    def __init__(self):
        pass

    def _get_user_integrations(self, user_id: str) -> UserIntegrations:
        return UserIntegrations()


def test_get_all_creds_includes_discovered(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    cs.discover_default_credentials.cache_clear()
    store = DummyStore()
    creds = store.get_all_creds("user")
    assert any(c.provider == "fal" for c in creds)
