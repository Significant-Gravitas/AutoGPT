from backend.util.link_checkout.network import NetworkDrain, NetworkMetadata


def test_drain_requires_request_completion_and_a_quiet_period(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "backend.util.link_checkout.network.time.monotonic", lambda: now[0]
    )
    drain = NetworkDrain()
    drain.observe(
        "Network.requestWillBeSent",
        "page",
        NetworkMetadata(requestId="payment", type="Fetch"),
    )
    now[0] = 10
    assert not drain.settled()
    drain.observe(
        "Network.loadingFinished", "page", NetworkMetadata(requestId="payment")
    )
    assert not drain.settled()
    now[0] = 11
    assert drain.settled()


def test_long_lived_websocket_is_not_an_unfinished_http_request(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "backend.util.link_checkout.network.time.monotonic", lambda: now[0]
    )
    drain = NetworkDrain()
    drain.observe(
        "Network.requestWillBeSent",
        "frame",
        NetworkMetadata(requestId="chat", type="WebSocket"),
    )
    now[0] = 3
    assert drain.settled()
