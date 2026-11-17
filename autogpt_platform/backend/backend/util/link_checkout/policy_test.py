import httpx
import pytest

from backend.util.link_checkout import policy
from backend.util.link_checkout.policy import in_app_approval_allowed


@pytest.fixture
def link_policy(monkeypatch):
    reply: dict = {"status": 200, "json": {"rules": []}}
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(reply["status"], json=reply["json"])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        policy.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    return reply, seen


def rule(amount=5000, currency="usd", methods=None, action="spend_request_create"):
    return {
        "action": action,
        "limits": {"per_purchase": {"amount": amount, "currency": currency}},
        "allowed_payment_methods": methods,
    }


@pytest.mark.asyncio
async def test_a_purchase_within_the_customers_policy_is_approved_in_chat(
    plan, link_policy
):
    reply, seen = link_policy
    reply["json"] = {"rules": [rule(methods=["csmrpd_test"])]}

    assert await in_app_approval_allowed("liwltoken_test", plan)
    assert str(seen[0].url) == "https://api.link.com/approval-policy"
    assert seen[0].headers["Authorization"] == "Bearer liwltoken_test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rules",
    [
        [],
        [rule(amount=99)],
        [rule(currency="eur")],
        [rule(methods=["csmrpd_other"])],
        [rule(action="something_else")],
    ],
)
async def test_anything_outside_the_policy_is_approved_in_link(
    plan, link_policy, rules
):
    reply, _ = link_policy
    reply["json"] = {"rules": rules}
    assert not await in_app_approval_allowed("liwltoken_test", plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,json",
    [(403, {"error": "not enabled"}), (200, {"unexpected": True}), (500, {})],
)
async def test_an_unreadable_policy_falls_back_to_link(plan, link_policy, status, json):
    reply, _ = link_policy
    reply["status"], reply["json"] = status, json
    assert not await in_app_approval_allowed("liwltoken_test", plan)
