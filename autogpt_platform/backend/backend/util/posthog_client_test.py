from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from unittest.mock import Mock

import pytest

from backend.util import posthog_client


def test_concurrent_initialization_reuses_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = Event()
    release_first = Event()
    client = Mock()
    extra_client = Mock()

    def create_client(*args: object, **kwargs: object) -> Mock:
        if first_started.is_set():
            return extra_client
        first_started.set()
        assert release_first.wait(timeout=5)
        return client

    settings = Mock()
    settings.secrets.posthog_api_key = "test-key"
    settings.secrets.posthog_host = "https://example.com"
    factory = Mock(side_effect=create_client)
    monkeypatch.setattr(posthog_client, "_client", None)
    monkeypatch.setattr(posthog_client, "Settings", Mock(return_value=settings))
    monkeypatch.setattr(posthog_client, "Posthog", factory)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(posthog_client.get_posthog_client)
        assert first_started.wait(timeout=5)
        second = executor.submit(posthog_client.get_posthog_client)
        try:
            second.result(timeout=0.2)
        except TimeoutError:
            pass
        finally:
            release_first.set()

        assert first.result(timeout=5) is client
        assert second.result(timeout=5) is client

    factory.assert_called_once_with("test-key", host="https://example.com")


def test_disabled_analytics_does_not_create_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Mock()
    settings.secrets.posthog_api_key = ""
    factory = Mock()
    monkeypatch.setattr(posthog_client, "_client", None)
    monkeypatch.setattr(posthog_client, "Settings", Mock(return_value=settings))
    monkeypatch.setattr(posthog_client, "Posthog", factory)

    assert posthog_client.get_posthog_client() is None
    factory.assert_not_called()
