import httpx
import pytest

from backend.util.link_checkout import preflight
from backend.util.link_checkout.preflight import purchase_blocker


@pytest.fixture
def user_info(monkeypatch):
    reply: dict = {"status": 200, "json": {"email": "jane@example.com"}}
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(reply["status"], json=reply["json"])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        preflight.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    return reply, seen


def limits(per_purchase=None, daily=None, thirty_day=None) -> dict:
    return {
        "agent_wallet_spend_limits": {
            "per_transaction": {"limit": per_purchase},
            "daily": {"limit": None, "used": 0, "remaining": daily},
            "thirty_day": {"limit": None, "used": 0, "remaining": thirty_day},
        }
    }


@pytest.mark.asyncio
async def test_nothing_in_the_way_lets_the_purchase_go_ahead(user_info):
    reply, seen = user_info
    reply["json"] = limits(per_purchase=5000, daily=10_000, thirty_day=50_000)

    assert await purchase_blocker("liwltoken_test", 2500, "usd") is None
    assert str(seen[0].url) == "https://api.link.com/userinfo"
    assert seen[0].headers["Authorization"] == "Bearer liwltoken_test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spend_limits,words",
    [
        (limits(per_purchase=2000), "per-purchase limit"),
        (limits(daily=1000), "remaining daily limit"),
        (limits(thirty_day=500), "remaining 30-day limit"),
    ],
)
async def test_a_purchase_over_an_agent_limit_is_stopped_before_approval(
    user_info, spend_limits, words
):
    reply, _ = user_info
    reply["json"] = spend_limits

    blocker = await purchase_blocker("liwltoken_test", 2500, "usd")

    assert blocker is not None
    assert words in blocker.message
    assert "$25.00" in blocker.message


@pytest.mark.asyncio
async def test_unlimited_or_missing_limits_never_block(user_info):
    reply, _ = user_info
    reply["json"] = limits()
    assert await purchase_blocker("liwltoken_test", 50_000, "usd") is None
    # Limits carry no currency, so they are only compared for US dollars.
    reply["json"] = limits(per_purchase=100)
    assert await purchase_blocker("liwltoken_test", 2500, "cad") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key", ["agent_wallet_verification_requirement", "agent_wallet_step_up"]
)
async def test_a_verification_link_needs_is_named_with_its_link(user_info, key):
    reply, _ = user_info
    reply["json"] = {
        key: {
            "status": "identity_verification",
            "action_url": "https://app.link.com/finish_setup?verify=identity",
        }
    }

    blocker = await purchase_blocker("liwltoken_test", 100, "usd")

    assert blocker is not None
    assert "verify their identity" in blocker.message
    assert blocker.action_url == "https://app.link.com/finish_setup?verify=identity"


@pytest.mark.asyncio
async def test_a_verification_link_elsewhere_is_not_passed_on(user_info):
    reply, _ = user_info
    reply["json"] = {
        "agent_wallet_verification_requirement": {
            "status": "ssn_verification",
            "action_url": "https://evil.example/collect",
        }
    }

    blocker = await purchase_blocker("liwltoken_test", 100, "usd")

    assert blocker is not None
    assert blocker.action_url == ""
    assert "evil" not in blocker.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,json",
    [(401, {"error": "expired"}), (500, {}), (200, {"agent_wallet_spend_limits": 7})],
)
async def test_an_unreadable_answer_lets_the_purchase_go_ahead(user_info, status, json):
    reply, _ = user_info
    reply["status"], reply["json"] = status, json
    assert await purchase_blocker("liwltoken_test", 100, "usd") is None
