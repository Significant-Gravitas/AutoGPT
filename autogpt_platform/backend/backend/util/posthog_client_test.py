import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from unittest.mock import Mock

import pytest

from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent


@pytest.fixture(autouse=True)
def reset_client_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posthog_client, "_client", None)
    monkeypatch.setattr(posthog_client, "_initialized", False, raising=False)


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


def test_disabled_analytics_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Mock()
    settings.secrets.posthog_api_key = ""
    settings_factory = Mock(return_value=settings)
    monkeypatch.setattr(posthog_client, "Settings", settings_factory)

    assert posthog_client.get_posthog_client() is None
    assert posthog_client.get_posthog_client() is None

    settings_factory.assert_called_once_with()


@pytest.mark.parametrize("failure", ["settings", "client"])
def test_initialization_failure_is_safe_and_retryable(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    settings = Mock()
    settings.secrets.posthog_api_key = "test-key"
    settings.secrets.posthog_host = "https://example.com"
    client = Mock()
    settings_factory = Mock(return_value=settings)
    client_factory = Mock(return_value=client)
    if failure == "settings":
        settings_factory.side_effect = [
            RuntimeError("configuration unavailable"),
            settings,
        ]
    else:
        client_factory.side_effect = [RuntimeError("client unavailable"), client]
    monkeypatch.setattr(posthog_client, "Settings", settings_factory)
    monkeypatch.setattr(posthog_client, "Posthog", client_factory)

    assert posthog_client.get_posthog_client() is None
    assert posthog_client.get_posthog_client() is client
    assert posthog_client.get_posthog_client() is client
    assert settings_factory.call_count == 2


def test_capture_adds_the_base_properties_last(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)
    monkeypatch.setattr(posthog_client, "_environment", lambda: "prod")

    posthog_client.capture(
        "user-1",
        PostHogEvent.SUBSCRIPTION_CHANGED,
        {"subscription_tier": "PRO", "source": "caller"},
        source="chat_copilot",
    )

    client.capture.assert_called_once_with(
        distinct_id="user-1",
        event="subscription_changed",
        properties={
            "subscription_tier": "PRO",
            "environment": "prod",
            "source": "chat_copilot",
        },
        uuid=None,
    )


def test_capture_defaults_the_source_to_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    posthog_client.capture("user-1", PostHogEvent.TOPUP_COMPLETED)

    properties = client.capture.call_args.kwargs["properties"]
    assert properties["source"] == "platform"
    assert properties["environment"] in {"local", "dev", "prod"}


@pytest.mark.parametrize("distinct_id", [None, ""])
def test_capture_without_a_user_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, distinct_id: str | None
) -> None:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    posthog_client.capture(distinct_id, PostHogEvent.CHAT_TOOL_CALLED)

    client.capture.assert_not_called()


def test_capture_is_a_noop_when_analytics_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: None)

    posthog_client.capture("user-1", PostHogEvent.CHAT_TOOL_CALLED)


def test_capture_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.capture.side_effect = RuntimeError("posthog down")
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    posthog_client.capture("user-1", PostHogEvent.CHAT_TOOL_CALLED)


def test_capture_dedup_key_is_insert_id_and_a_stable_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    posthog_client.capture(
        "user-1", PostHogEvent.BRIEFING_DELIVERED, {}, dedup_key="briefing:b-1"
    )

    kwargs = client.capture.call_args.kwargs
    assert kwargs["properties"]["$insert_id"] == "briefing:b-1"
    assert kwargs["uuid"] == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "user-1:briefing_delivered:briefing:b-1")
    )
