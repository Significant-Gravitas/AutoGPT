import base64
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.util.link_checkout import broker_checkout, engine
from backend.util.link_checkout.broker_protocol import AuthorizedCheckout, BrowserOutput

VIEW = {
    "checkout_id": "a" * 32,
    "credentials_id": "wallet",
    "merchant_name": "Test store",
    "merchant_url": "https://shop.example/checkout",
    "amount": 100,
    "currency": "usd",
    "test_mode": True,
    "status": "approved",
    "message": "ok",
}


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setenv("CHECKOUT_BROKER_URL", "https://broker.internal:8443")
    request = AsyncMock(return_value=VIEW)
    monkeypatch.setattr(engine.broker_client, "request", request)
    return request


def completion() -> AuthorizedCheckout:
    return AuthorizedCheckout(
        user_id="owner",
        session_id="chat",
        checkout_id="a" * 32,
        access_token=SecretStr("liwltoken_test"),
    )


@pytest.mark.parametrize(
    "requested,runtime_ready,expected",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_in_process_private_browsing_needs_a_ready_runtime(
    monkeypatch, requested, runtime_ready, expected
):
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true" if requested else "")
    for name in ("CHECKOUT_BROKER_URL", "CHECKOUT_BROKER_ROUTES_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engine, "local_runtime_ready", lambda: runtime_ready)
    assert engine.enabled() is expected
    assert engine.active_for("owner") is expected


def test_a_broker_serves_only_the_user_it_was_provisioned_for(monkeypatch):
    """Everyone else keeps ordinary browsing, rather than having every browser
    command refused by a broker that is not theirs."""
    monkeypatch.setenv("CHECKOUT_BROKER_URL", "https://broker.internal:8443")
    monkeypatch.setenv("CHECKOUT_BROKER_USER_ID", "owner")
    for name in ("CA", "CLIENT_CERT", "CLIENT_KEY", "SECRET_FILE"):
        monkeypatch.setenv(f"CHECKOUT_BROKER_{name}", "configured")
    monkeypatch.delenv("CHECKOUT_BROKER_ROUTES_FILE", raising=False)

    assert engine.enabled()
    assert engine.active_for("owner")
    assert not engine.active_for("someone-else")
    assert not engine.active_for(None)


def test_commands_reach_the_private_browser_only_inside_its_callers_context():
    assert not engine.serves("chat")
    with engine.caller("owner", "chat", "owner"):
        assert engine.serves("chat")
        assert not engine.serves("another-chat")
    assert not engine.serves("chat")


@pytest.mark.asyncio
async def test_a_remote_broker_receives_the_operation_and_the_token(remote):
    view = await engine.complete(completion())

    operation, payload = remote.await_args.args
    assert operation == "checkout/complete"
    assert payload.access_token.get_secret_value() == "liwltoken_test"
    assert view.status == "approved"


@pytest.mark.asyncio
async def test_without_a_remote_broker_the_checkout_runs_in_process(monkeypatch):
    monkeypatch.delenv("CHECKOUT_BROKER_URL", raising=False)
    monkeypatch.delenv("CHECKOUT_BROKER_ROUTES_FILE", raising=False)
    local = AsyncMock(return_value=VIEW)
    monkeypatch.setattr(broker_checkout, "complete_checkout", local)
    await engine.complete(completion())
    local.assert_awaited_once()


def test_the_caller_must_own_the_chat():
    with pytest.raises(ValueError):
        with engine.caller("mallory", "chat", "owner"):
            pytest.fail("Entered another user's browser context")


@pytest.mark.asyncio
async def test_browser_commands_need_the_callers_own_session(remote):
    with pytest.raises(ValueError):
        await engine.run_browser_command("chat", ("get", "url"))
    with engine.caller("owner", "chat", "owner"):
        with pytest.raises(ValueError):
            await engine.run_browser_command("another-chat", ("get", "url"))


@pytest.mark.asyncio
async def test_a_screenshot_comes_back_as_the_local_file(remote, tmp_path):
    image = b"\x89PNG synthetic"
    remote.return_value = BrowserOutput(
        code=0, image=base64.b64encode(image).decode()
    ).model_dump()
    target = tmp_path / "shot.png"

    with engine.caller("owner", "chat", "owner"):
        code, _, _ = await engine.run_browser_command(
            "chat", ("screenshot", "--annotate", str(target))
        )

    assert code == 0
    assert target.read_bytes() == image
    operation, command = remote.await_args.args
    assert operation == "browser"
    assert command.args == ["screenshot", "--annotate"]
