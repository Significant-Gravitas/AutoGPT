import httpx
import pytest
from fastapi import FastAPI

from backend.data import trial_notifications as outbox
from backend.data import trial_notifications_integration_test as fixtures
from backend.data.db_manager import DatabaseManager

enrollment = fixtures.enrollment
payload = fixtures.payload
pytestmark = fixtures.pytestmark


def make_rpc_app():
    manager = DatabaseManager()
    app = FastAPI()
    for operation in (
        outbox.enqueue_trial_notification,
        outbox.claim_trial_notification,
        outbox.finish_trial_notification,
    ):
        app.add_api_route(
            f"/{operation.__name__}",
            manager._create_fastapi_endpoint(operation),
            methods=["POST"],
        )
    return app


async def post_json(client, operation, body, expected_status=200):
    response = await client.post(f"/{operation}", json=body)
    assert response.status_code == expected_status
    return response.json()


async def claim(client, delivery_id):
    result = await post_json(
        client, "claim_trial_notification", {"delivery_id": delivery_id}
    )
    return outbox.ClaimedTrialDelivery.model_validate(result)


async def finish(client, delivery, status, expected_status=200):
    return await post_json(
        client,
        "finish_trial_notification",
        {
            "delivery_id": delivery.id,
            "lease_token": delivery.lease_token,
            "status": status,
        },
        expected_status,
    )


@pytest.mark.asyncio
async def test_reactivation_and_obsolete_status_cross_actual_rpc_endpoint_models(
    enrollment, payload
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_rpc_app()), base_url="http://trial-rpc"
    ) as client:
        request = {
            "user_id": enrollment.user_id,
            "trial_id": enrollment.id,
            "idempotency_key": f"trial:{enrollment.id}:started",
            "data": payload.model_dump(mode="json"),
        }
        receipt = outbox.TrialNotificationReceipt.model_validate(
            await post_json(client, "enqueue_trial_notification", request)
        )
        first = await claim(client, receipt.id)
        assert first.payload == payload and first.attempts == 1
        await finish(client, first, "unknown", expected_status=422)
        assert await finish(client, first, "suppressed") is True
        assert await post_json(client, "enqueue_trial_notification", request) == {
            "id": receipt.id,
            "created": False,
        }
        second = await claim(client, receipt.id)
        assert second.payload == payload and second.attempts == 2
        assert await finish(client, second, "obsolete") is True
        assert (
            await post_json(
                client, "claim_trial_notification", {"delivery_id": receipt.id}
            )
            is None
        )
