"""The refund route against Postgres and the app's own error handlers."""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fastapi
import httpx
import pytest
import pytest_asyncio
import pytest_mock
from autogpt_libs.auth import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from prisma.enums import CreditTransactionType
from prisma.models import CreditRefundRequest, CreditTransaction, User

from backend.api.features.billing.credits.routes import router
from backend.api.rest_api import app as rest_app
from backend.data.credit import UserCredit
from backend.util.json import SafeJson

pytestmark = pytest.mark.asyncio(loop_scope="session")

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(router)


async def test_refunding_someone_elses_top_up_is_the_same_404_as_an_unknown_key(
    client: httpx.AsyncClient,
    users: tuple[str, str],
    refund_side_effects: list[AsyncMock],
):
    _, owner_id = users
    key = f"pi_{uuid.uuid4().hex}"

    unknown = await client.post(f"/credits/{key}/refund", json={"reason": "x"})
    await CreditTransaction.prisma().create(
        data={
            "userId": owner_id,
            "amount": 1000,
            "type": CreditTransactionType.TOP_UP,
            "transactionKey": key,
            "runningBalance": 1000,
            "isActive": True,
            "metadata": SafeJson({}),
        }
    )
    someone_elses = await client.post(f"/credits/{key}/refund", json={"reason": "x"})

    assert unknown.status_code == 404
    assert someone_elses.status_code == 404
    assert someone_elses.json() == unknown.json()
    assert await CreditRefundRequest.prisma().count(where={"transactionKey": key}) == 0
    for side_effect in refund_side_effects:
        side_effect.assert_not_awaited()


@pytest_asyncio.fixture(loop_scope="session")
async def client(
    users: tuple[str, str], mocker: pytest_mock.MockFixture
) -> AsyncIterator[httpx.AsyncClient]:
    caller_id, _ = users

    def jwt_payload() -> dict[str, str]:
        return {"sub": caller_id, "role": "user", "email": f"{caller_id}@example.com"}

    def request_context() -> RequestContext:
        return RequestContext(
            user_id=caller_id,
            org_id="personal-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    mocker.patch(
        "backend.api.features.billing.credits.routes.get_credit_model",
        AsyncMock(return_value=UserCredit()),
    )
    app.dependency_overrides[get_jwt_payload] = jwt_payload
    app.dependency_overrides[get_request_context] = request_context
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(loop_scope="session")
async def users(server) -> AsyncIterator[tuple[str, str]]:
    """The caller, and the owner of the top-up the caller asks to refund."""
    ids = [f"refund-route-{uuid.uuid4()}" for _ in range(2)]
    for user_id in ids:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com"}
        )
    yield ids[0], ids[1]
    await CreditRefundRequest.prisma().delete_many(where={"userId": {"in": ids}})
    await CreditTransaction.prisma().delete_many(where={"userId": {"in": ids}})
    await User.prisma().delete_many(where={"id": {"in": ids}})


@pytest.fixture
def refund_side_effects(mocker: pytest_mock.MockFixture) -> list[AsyncMock]:
    """A lookup that matched another user's top-up would go on to refund it in
    Stripe, or mail the refunds team."""
    return [
        mocker.patch("backend.data.credit.stripe_call", new_callable=AsyncMock),
        mocker.patch(
            "backend.data.credit.queue_notification_async", new_callable=AsyncMock
        ),
    ]
