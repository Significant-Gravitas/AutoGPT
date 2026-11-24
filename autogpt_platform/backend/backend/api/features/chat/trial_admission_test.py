"""A trial user's message through the real chat stream route.

The paywall, the tier and trial lookups, the trial's spend limits and the
rate-limit check all run as in production. Only what they read is stubbed:
the user's tier, the trial row and the Redis spend counters. The turn is
scheduled no further than the RabbitMQ boundary.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.chat import routes as chat_routes
from backend.copilot import rate_limit, usage_activation
from backend.copilot.billing_e2e.trial_account import active_trial, usage_state_of
from backend.copilot.briefing import scheduling as briefing_scheduling
from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import SubscriptionTier, UserPaywalledError
from backend.data.subscription_trial import TrialState

# Kept before the API suite's autouse fixture swaps in a paywall that lets
# every user through.
_REAL_IS_USER_PAYWALLED = rate_limit.is_user_paywalled

app = fastapi.FastAPI()
app.include_router(chat_routes.router)


@app.exception_handler(UserPaywalledError)
async def _paywalled_handler(
    request: fastapi.Request, exc: UserPaywalledError
) -> fastapi.responses.JSONResponse:
    """Mirror the production UserPaywalledError → 402 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=402, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)


class RouteTrialAccount:
    """What the route reads for the user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.tier = SubscriptionTier.TRIAL
        self.trial = active_trial(user_id, daily_limit=250_000, weekly_limit=1_000_000)
        self.spent_today = 0
        self.read_error: Exception | None = None

    async def read_trial(self, user_id: str) -> TrialState | None:
        assert user_id == self.user_id
        if self.read_error is not None:
            raise self.read_error
        return self.trial

    async def read_tier(self, user_id: str) -> SubscriptionTier:
        assert user_id == self.user_id
        return self.tier

    async def read_usage_state(self, user_id: str):
        assert user_id == self.user_id
        return usage_state_of(user_id, self.tier, self.trial)

    async def read_counter(self, key: str) -> int | None:
        # Any day's daily counter: the check computes its own date.
        daily = rate_limit._daily_key(self.user_id).rsplit(":", 1)[0]
        if key.startswith(f"{daily}:"):
            return self.spent_today
        return None

    def end_trial(self) -> None:
        self.trial = self.trial.model_copy(
            update={"ends_at": datetime.now(UTC) - timedelta(minutes=1)}
        )


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def account(mocker: pytest_mock.MockerFixture, test_user_id: str) -> RouteTrialAccount:
    account = RouteTrialAccount(test_user_id)
    mocker.patch.object(rate_limit, "is_user_paywalled", _REAL_IS_USER_PAYWALLED)
    mocker.patch.object(
        rate_limit, "_fetch_user_tier", AsyncMock(side_effect=account.read_tier)
    )
    mocker.patch.object(
        usage_activation,
        "pro_activation_db",
        return_value=MagicMock(
            get_usage_activation_state=AsyncMock(side_effect=account.read_usage_state)
        ),
    )
    mocker.patch.object(
        rate_limit,
        "credit_db",
        return_value=MagicMock(
            get_subscription_trial=AsyncMock(side_effect=account.read_trial)
        ),
    )
    mocker.patch.object(
        rate_limit,
        "get_redis_async",
        AsyncMock(
            return_value=MagicMock(get=AsyncMock(side_effect=account.read_counter))
        ),
    )
    # The flag service's overrides of the default limits; none are set.
    mocker.patch.object(
        rate_limit, "_fetch_cost_limits_flag", AsyncMock(return_value=None)
    )
    mocker.patch.object(
        rate_limit, "_fetch_tier_multipliers_flag", AsyncMock(return_value=None)
    )
    return account


@pytest.fixture
def scheduled(mocker: pytest_mock.MockerFixture, test_user_id: str) -> AsyncMock:
    """The turns the route handed to the executor."""
    session = ChatSession.new(test_user_id, dry_run=False)
    mocker.patch.object(
        chat_routes, "_validate_and_get_session", AsyncMock(return_value=session)
    )
    mocker.patch.object(chat_routes, "is_turn_in_flight", AsyncMock(return_value=False))
    mocker.patch.object(
        briefing_scheduling, "ensure_morning_briefing_scheduled", AsyncMock()
    )
    mocker.patch.object(chat_routes.stream_registry, "publish_chunk", AsyncMock())
    mocker.patch.object(
        chat_routes.stream_registry,
        "subscribe_to_session",
        AsyncMock(return_value=None),
    )
    return mocker.patch.object(
        chat_routes, "schedule_chat_turn", AsyncMock(return_value="turn-1")
    )


def test_a_trial_within_its_limits_starts_the_turn(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    response = _send_message()

    assert response.status_code == 200
    scheduled.assert_awaited_once()


def test_a_trial_that_spent_its_allowance_waits_for_its_end(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    account.trial = account.trial.model_copy(
        update={"cost_microdollars": account.trial.offer.total_cost_limit}
    )

    response = _send_message()

    assert response.status_code == 429
    failure = response.json()["detail"]
    assert failure["kind"] == "usage_limit"
    assert "trial usage limit" in failure["message"]
    assert account.trial.ends_at is not None
    assert failure["resetsAt"] == int(account.trial.ends_at.timestamp())
    scheduled.assert_not_awaited()


@pytest.mark.parametrize(
    "spent_today, admitted",
    [(249_999, True), (250_000, False)],
    ids=["just under", "at the limit"],
)
def test_a_trial_is_held_to_its_own_daily_limit(
    account: RouteTrialAccount, scheduled: AsyncMock, spent_today: int, admitted: bool
) -> None:
    # The platform's default daily limit is far above the trial's $0.25.
    assert account.trial.offer.daily_cost_limit == 250_000
    account.spent_today = spent_today

    response = _send_message()

    if admitted:
        assert response.status_code == 200
        scheduled.assert_awaited_once()
        return
    assert response.status_code == 429
    assert "daily usage limit" in response.json()["detail"]["message"]
    scheduled.assert_not_awaited()


def test_an_ended_trial_meets_the_paywall(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    account.end_trial()

    response = _send_message()

    assert response.status_code == 402
    scheduled.assert_not_awaited()


def test_a_trial_that_ends_between_messages_stops_the_next_one(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    assert _send_message().status_code == 200
    account.end_trial()

    assert _send_message().status_code == 402
    scheduled.assert_awaited_once()


def test_a_failed_trial_read_refuses_the_message_and_is_not_kept(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    account.read_error = ConnectionError("database unreachable")

    refused = _send_message()

    assert refused.status_code == 503
    scheduled.assert_not_awaited()
    account.read_error = None
    assert _send_message().status_code == 200
    scheduled.assert_awaited_once()


def test_a_paying_user_is_not_held_to_a_trial_that_ended(
    account: RouteTrialAccount, scheduled: AsyncMock
) -> None:
    account.tier = SubscriptionTier.PRO
    account.end_trial()

    response = _send_message()

    assert response.status_code == 200
    scheduled.assert_awaited_once()


def _send_message():
    return client.post("/sessions/sess-1/stream", json={"message": "hello"})
