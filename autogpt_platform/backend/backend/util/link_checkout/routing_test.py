import json
import os
from unittest.mock import AsyncMock

import pytest

from backend.copilot import integration_creds
from backend.util.link_checkout.broker_routing import route_for, routed


@pytest.mark.asyncio
async def test_code_execution_does_not_even_load_wallet_tokens(monkeypatch):
    lookup = AsyncMock(return_value="synthetic-provider-token")
    monkeypatch.setattr(integration_creds, "get_provider_token", lookup)
    result = await integration_creds.get_integration_env_vars("owner")
    assert "GH_TOKEN" in result
    assert all(call.args[1] != "stripe_link" for call in lookup.await_args_list)


def test_routes_are_selected_only_from_operator_registry(monkeypatch, tmp_path):
    registry = tmp_path / "routes.json"
    common = dict(ca="ca", client_cert="cert", client_key="key", secret_file="secret")
    routes = [
        {**common, "user_id": "alice", "url": "https://alice-broker:8443"},
        {**common, "user_id": "bob", "url": "https://bob-broker:8443"},
    ]
    registry.write_text(json.dumps(routes))
    monkeypatch.setenv("CHECKOUT_BROKER_ROUTES_FILE", str(registry))
    assert route_for("alice").url == "https://alice-broker:8443"
    assert route_for("bob").url == "https://bob-broker:8443"
    with pytest.raises(ValueError):
        route_for("mallory")
    assert routed("alice") and not routed("mallory")
    registry.write_text(json.dumps([routes[0], routes[0]]))
    with pytest.raises(ValueError):
        route_for("alice")
    assert not routed("alice")


def test_a_single_broker_is_only_its_provisioned_users(monkeypatch):
    monkeypatch.delenv("CHECKOUT_BROKER_ROUTES_FILE", raising=False)
    monkeypatch.setenv("CHECKOUT_BROKER_URL", "https://broker:8443")
    for name in ("CA", "CLIENT_CERT", "CLIENT_KEY", "SECRET_FILE"):
        monkeypatch.setenv(f"CHECKOUT_BROKER_{name}", "configured")
    monkeypatch.delenv("CHECKOUT_BROKER_USER_ID", raising=False)
    assert not routed("alice")
    monkeypatch.setenv("CHECKOUT_BROKER_USER_ID", "alice")
    assert route_for("alice").url == "https://broker:8443"
    with pytest.raises(ValueError):
        route_for("mallory")


@pytest.mark.parametrize(
    "url",
    [
        "http://broker:8443",
        "https://broker:8443/path",
        "https://user:pass@broker:8443",
        "https://broker:8443?x=1",
    ],
)
def test_a_route_is_a_bare_https_origin(monkeypatch, tmp_path, url):
    registry = tmp_path / "routes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "user_id": "alice",
                    "url": url,
                    "ca": "ca",
                    "client_cert": "cert",
                    "client_key": "key",
                    "secret_file": "secret",
                }
            ]
        )
    )
    monkeypatch.setenv("CHECKOUT_BROKER_ROUTES_FILE", str(registry))
    with pytest.raises(ValueError):
        route_for("alice")


def test_routes_are_read_again_only_when_the_file_changes(monkeypatch, tmp_path):
    registry = tmp_path / "routes.json"
    common = dict(ca="ca", client_cert="cert", client_key="key", secret_file="secret")
    alice = {**common, "user_id": "alice", "url": "https://alice-broker:8443"}
    registry.write_text(json.dumps([alice]))
    monkeypatch.setenv("CHECKOUT_BROKER_ROUTES_FILE", str(registry))

    first = route_for("alice")
    assert route_for("alice") is first

    bob = {**common, "user_id": "bob", "url": "https://bob-broker:8443"}
    registry.write_text(json.dumps([bob]))
    stat = registry.stat()
    os.utime(registry, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError):
        route_for("alice")
    assert route_for("bob").url == "https://bob-broker:8443"
