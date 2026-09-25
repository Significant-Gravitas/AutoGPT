"""Route tests for approving or declining a Link purchase from the chat card.

Redis is faked at the boundary; the approval store's own rules are covered in
``backend/util/link_checkout/approval_test.py``.
"""

import time
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.chat import link_checkout as routes
from backend.util.link_checkout import approval
from backend.util.link_checkout.approval import PendingApproval, open_approval

app = fastapi.FastAPI()
app.include_router(routes.router)
client = fastapi.testclient.TestClient(app)

SESSION_ID = "8d2f5c1e-2b7a-4c55-9d0e-3f6a1b2c4d5e"
CONTEXT = "One paperback from the signed-in cart, shipped to the saved address."
CHECKOUT_ID = "d" * 32


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int = 0) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def purchase(monkeypatch, test_user_id):
    monkeypatch.setattr(
        approval, "get_redis_async", AsyncMock(return_value=FakeRedis())
    )

    async def open_for(user_id: str = test_user_id, expires_in: float = 60):
        await open_approval(
            PendingApproval(
                checkout_id=CHECKOUT_ID,
                user_id=user_id,
                session_id=SESSION_ID,
                merchant_name="Test store",
                merchant_url="https://shop.example/checkout",
                context=CONTEXT,
                amount=1250,
                currency="usd",
                test_mode=True,
                expires_at=time.time() + expires_in,
            )
        )

    return open_for


def url(action: str = "") -> str:
    suffix = f"/{action}" if action else ""
    return f"/sessions/{SESSION_ID}/link-checkouts/{CHECKOUT_ID}{suffix}"


@pytest.mark.asyncio
async def test_the_card_reads_the_purchase_as_recorded(purchase):
    await purchase()

    response = client.get(url())

    assert response.status_code == 200
    assert response.json() == {
        "checkout_id": CHECKOUT_ID,
        "state": "awaiting",
        "merchant_name": "Test store",
        "merchant_url": "https://shop.example/checkout",
        "context": CONTEXT,
        "amount": 1250,
        "currency": "usd",
        "test_mode": True,
        "expires_at": response.json()["expires_at"],
    }


@pytest.mark.asyncio
async def test_approving_records_the_click_and_is_idempotent(purchase):
    await purchase()

    first = client.post(url("approve"), headers={"User-Agent": "Mozilla/5.0"})
    again = client.post(url("approve"))

    assert first.status_code == again.status_code == 200
    assert first.json()["state"] == "approved"
    view = await approval.read_approval(CHECKOUT_ID)
    assert view is not None and view.decision is not None
    assert view.decision.user_agent == "Mozilla/5.0"


@pytest.mark.asyncio
async def test_declined_cannot_be_approved_later(purchase):
    await purchase()

    assert client.post(url("decline")).json()["state"] == "declined"
    assert client.post(url("approve")).status_code == 409


@pytest.mark.asyncio
async def test_an_expired_purchase_cannot_be_approved(purchase):
    await purchase(expires_in=-1)

    assert client.get(url()).json()["state"] == "expired"
    assert client.post(url("approve")).status_code == 409


@pytest.mark.asyncio
async def test_another_users_purchase_is_not_found(purchase):
    await purchase(user_id="someone-else")

    assert client.get(url()).status_code == 404
    assert client.post(url("approve")).status_code == 404
    view = await approval.read_approval(CHECKOUT_ID)
    assert view is not None and view.state == "awaiting"


def test_a_malformed_checkout_id_is_rejected():
    response = client.post(
        f"/sessions/{SESSION_ID}/link-checkouts/not-a-checkout/approve"
    )
    assert response.status_code == 422
