import pytest
import urllib3.connectionpool
from posthog import Posthog


@pytest.fixture(autouse=True)
def no_real_http(request, monkeypatch):
    """Fails a test that reaches for the network, naming what it would have sent.

    A real PostHog client posts a `$feature_flag_called` event for every flag
    read; refusing it at `capture` also stops the uploads its background thread
    would make after the test ends. Yields the record, which the guard's own
    tests clear.
    """
    sent: list[str] = []

    def refuse_request(pool, method, url, *args, **kwargs):
        sent.append(f"{method} {pool.scheme}://{pool.host}{url.split('?')[0]}")
        raise urllib3.exceptions.ProtocolError("refused by the no_real_http fixture")

    def refuse_event(client, event, **kwargs):
        sent.append(f"a {event} event")

    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "urlopen", refuse_request
    )
    monkeypatch.setattr(Posthog, "capture", refuse_event)
    yield sent
    if sent:
        pytest.fail(
            f"{request.node.nodeid} would have sent {sent} to the network. Mock "
            "the request, or use posthog_events_dropped for a flag read's events."
        )


@pytest.fixture
def posthog_events_dropped(mocker):
    """For tests whose real PostHog client reads flags, and so reports each read."""
    mocker.patch.object(Posthog, "capture")
