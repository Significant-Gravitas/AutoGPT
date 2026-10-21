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


def test_capture_never_raises_on_a_bad_event(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    posthog_client.capture("user-1", "chat_tool_called")  # type: ignore[arg-type]

    client.capture.assert_not_called()


@pytest.mark.parametrize(
    "returned, side_effect, queued",
    [
        ("event-uuid", None, True),
        # The client returns None for an event it dropped, such as on a full
        # queue.
        (None, None, False),
        (None, RuntimeError("posthog down"), False),
    ],
    ids=["queued", "dropped", "raised"],
)
def test_capture_says_whether_the_client_queued_the_event(
    monkeypatch: pytest.MonkeyPatch, returned, side_effect, queued: bool
) -> None:
    client = Mock()
    client.capture.return_value = returned
    client.capture.side_effect = side_effect
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)

    assert posthog_client.capture("user-1", PostHogEvent.CHAT_TOOL_CALLED) is queued


def test_capture_queues_nothing_without_a_user_or_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: None)

    assert posthog_client.capture("user-1", PostHogEvent.CHAT_TOOL_CALLED) is False
    assert posthog_client.capture(None, PostHogEvent.CHAT_TOOL_CALLED) is False


def test_capture_on_a_client_of_its_own_leaves_the_shared_one_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = Mock()
    own = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: shared)

    assert posthog_client.capture("user-1", PostHogEvent.CHAT_TOOL_CALLED, client=own)

    own.capture.assert_called_once()
    shared.capture.assert_not_called()


def test_a_client_of_its_own_gets_the_options_the_shared_one_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Mock()
    settings.secrets.posthog_api_key = "test-key"
    settings.secrets.posthog_host = "https://example.com"
    factory = Mock()
    on_error = Mock()
    monkeypatch.setattr(posthog_client, "Settings", Mock(return_value=settings))
    monkeypatch.setattr(posthog_client, "Posthog", factory)

    assert posthog_client.new_posthog_client(on_error=on_error) is factory.return_value
    factory.assert_called_once_with(
        "test-key", host="https://example.com", on_error=on_error
    )
    # The shared client is not created or replaced on the way.
    assert posthog_client._client is None


def test_no_client_of_its_own_when_analytics_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Mock()
    settings.secrets.posthog_api_key = ""
    factory = Mock()
    monkeypatch.setattr(posthog_client, "Settings", Mock(return_value=settings))
    monkeypatch.setattr(posthog_client, "Posthog", factory)

    assert posthog_client.new_posthog_client(on_error=Mock()) is None
    factory.assert_not_called()


@pytest.fixture
def own_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The run's own client. `own_client.on_error` is the callback the run
    gave it for uploads that failed for good."""
    client = Mock()

    def new_client(**options: object) -> Mock:
        client.on_error = options["on_error"]
        return client

    monkeypatch.setattr(posthog_client, "new_posthog_client", new_client)
    return client


def _sends(
    count: int, refused: frozenset[str] = frozenset()
) -> list[posthog_client.Send]:
    """One send per distinct id "0".."count-1"; those in `refused` report the
    client didn't take them."""
    return [
        (str(i), lambda client, i=str(i): client.capture(i) and i not in refused)
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_send_each_sends_all_flushing_as_it_goes_then_shuts_down(
    own_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(posthog_client, "FLUSH_EVERY", 2)

    assert await posthog_client.send_each(_sends(5)) == 0

    assert [c.args[0] for c in own_client.capture.call_args_list] == list("01234")
    # After the second and the fourth; shutting down flushes the rest.
    assert own_client.flush.call_count == 2
    own_client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_send_each_counts_each_distinct_id_posthog_did_not_take_once(
    own_client: Mock,
) -> None:
    """Refused when queued, or in an upload that still failed after the
    client's retries, which reaches `on_error` by the end of the shutdown."""
    own_client.shutdown.side_effect = lambda: own_client.on_error(
        RuntimeError("503"), [{"distinct_id": "1"}, {"distinct_id": "3"}]
    )

    assert await posthog_client.send_each(_sends(4, frozenset({"1", "2"}))) == 3


@pytest.mark.asyncio
async def test_send_each_shuts_the_client_down_even_when_a_send_raises(
    own_client: Mock,
) -> None:
    def broken(_client: object) -> bool:
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        await posthog_client.send_each([("0", broken)])
    own_client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_send_each_without_posthog_raises_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(posthog_client, "new_posthog_client", lambda **_: None)
    send = Mock()

    with pytest.raises(RuntimeError, match="PostHog is not configured"):
        await posthog_client.send_each([("0", send)])
    send.assert_not_called()


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
