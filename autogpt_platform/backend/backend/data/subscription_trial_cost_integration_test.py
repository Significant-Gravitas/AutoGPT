"""Late trial costs remain user-scoped after access changes."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma.models import SubscriptionTrial

from backend.data import trial_notifications_integration_test as fixtures
from backend.data.db_manager import DatabaseManager
from backend.data.subscription_trial import (
    get_subscription_trial,
    record_subscription_trial_cost,
)

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "canceled", "past_due", "trialing"])
async def test_attributed_cost_survives_status_and_card_changes(enrollment, status):
    await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={"status": status, "consumedAt": datetime.now(UTC)},
    )
    await asyncio.gather(
        *[
            record_subscription_trial_cost(
                enrollment.user_id, 100, trial_id=enrollment.id
            )
            for _ in range(12)
        ]
    )
    trial = await get_subscription_trial(enrollment.user_id)
    assert trial is not None and trial.cost_microdollars == 1200


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_field", ["user", "trial", "unconsumed"])
async def test_invalid_cost_attribution_is_rejected(enrollment, wrong_field):
    if wrong_field != "unconsumed":
        await SubscriptionTrial.prisma().update(
            where={"id": enrollment.id}, data={"consumedAt": datetime.now(UTC)}
        )
    with pytest.raises(ValueError, match="Trial cost attribution"):
        await record_subscription_trial_cost(
            str(uuid4()) if wrong_field == "user" else enrollment.user_id,
            100,
            trial_id=str(uuid4()) if wrong_field == "trial" else enrollment.id,
        )
    trial = await get_subscription_trial(enrollment.user_id)
    assert trial is not None and trial.cost_microdollars == 0


@pytest.mark.asyncio
async def test_attribution_crosses_actual_database_rpc_models(enrollment):
    manager = DatabaseManager()
    app = FastAPI()
    manager._register_exception_handlers(app)
    app.add_api_route(
        "/record_subscription_trial_cost",
        manager._create_fastapi_endpoint(record_subscription_trial_cost),
        methods=["POST"],
    )
    await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={"status": "active", "consumedAt": datetime.now(UTC)},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://trial-rpc"
    ) as client:
        request = {
            "user_id": enrollment.user_id,
            "trial_id": enrollment.id,
            "cost_microdollars": 200,
        }
        response = await client.post("/record_subscription_trial_cost", json=request)
        assert response.status_code == 200 and response.json() is None
        response = await client.post(
            "/record_subscription_trial_cost", json={**request, "user_id": str(uuid4())}
        )
        assert response.status_code == 400
        response = await client.post(
            "/record_subscription_trial_cost", json={**request, "trial_id": 123}
        )
        assert response.status_code == 422
    trial = await get_subscription_trial(enrollment.user_id)
    assert trial is not None and trial.cost_microdollars == 200


@pytest.mark.asyncio
async def test_unscoped_calls_do_not_charge_a_previous_trial(enrollment):
    await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={"status": "active", "consumedAt": datetime.now(UTC)},
    )
    await record_subscription_trial_cost(enrollment.user_id, 100)
    await record_subscription_trial_cost(enrollment.user_id, 0, trial_id=enrollment.id)
    await record_subscription_trial_cost(enrollment.user_id, -1, trial_id=enrollment.id)
    trial = await get_subscription_trial(enrollment.user_id)
    assert trial is not None and trial.cost_microdollars == 0
