"""The daily report, driven against seeded accounts, request counts written by
the real counter and the paywall's own check of who has a plan."""

import random
from datetime import UTC, date, datetime, time, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from prisma.enums import AuthIPEventKind, SubscriptionTier
from prisma.models import AuthIPEvent, AuthUser, User

from backend.copilot import rate_limit
from backend.data import api_activity
from backend.data.redis_client import HOST, PORT
from backend.monitoring import no_plan_activity_monitor as monitor
from backend.util.feature_flag import Flag
from backend.util.test import SpinTestServer
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(HOST, PORT), reason="needs a live Redis"
)

NO_PLAN = SubscriptionTier.NO_TIER
PAYING = SubscriptionTier.PRO
SHARED_IP = "203.0.113.10"
OTHER_IP = "203.0.113.20"


class _Clock:
    """The counter's and the report's clock, on a day no other run writes to."""

    def __init__(self):
        self.day = date(2001, 1, 1) + timedelta(days=random.randrange(5000))
        self.now = self.at(12)

    def at(self, hour: int, day_offset: int = 0) -> datetime:
        self.now = datetime.combine(
            self.day + timedelta(days=day_offset), time(hour), tzinfo=UTC
        )
        return self.now


@pytest.fixture
def clock(mocker) -> _Clock:
    pinned = _Clock()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return pinned.now

    mocker.patch.object(api_activity, "datetime", Clock)
    mocker.patch.object(monitor, "datetime", Clock)
    return pinned


@pytest.fixture
def alerts(mocker):
    client = MagicMock()
    mocker.patch.object(monitor, "get_notification_manager_client", return_value=client)
    return client.discord_system_alert


@pytest.fixture(autouse=True)
def payments_on(mocker):
    async def flag(flag: Flag, *args, **kwargs) -> bool:
        return flag == Flag.ENABLE_PLATFORM_PAYMENT

    mocker.patch.object(rate_limit, "is_feature_enabled", new=flag)


@pytest.fixture
async def account(server: SpinTestServer, clock):
    """`account(tier) -> id`: an account two days old on the reported day."""
    created: list[str] = []

    async def make(tier: SubscriptionTier) -> str:
        user_id = str(uuid4())
        email = f"{user_id}@no-plan-report.example"
        signed_up = datetime.combine(clock.day, time(12), tzinfo=UTC)
        await AuthUser.prisma().create(
            data={
                "id": user_id,
                "name": "Reported",
                "email": email,
                "createdAt": signed_up - timedelta(days=2),
            }
        )
        await User.prisma().create(
            data={"id": user_id, "email": email, "subscriptionTier": tier}
        )
        created.append(user_id)
        return user_id

    yield make
    await User.prisma().delete_many(where={"id": {"in": created}})
    await AuthUser.prisma().delete_many(where={"id": {"in": created}})


@pytest.mark.asyncio(loop_scope="session")
async def test_reports_a_busy_account_without_a_plan_but_not_a_paying_one(
    account, clock, alerts
):
    split = await account(NO_PLAN)
    paying = await account(PAYING)
    quiet = await account(NO_PLAN)
    # Neither surface alone reaches the threshold of 500.
    await _requests(split, internal=300, external=300)
    await _requests(paying, internal=600)
    await _requests(quiet, internal=499)

    clock.at(0, day_offset=1)
    await monitor.report_no_plan_api_activity()

    alerts.assert_called_once()
    message = alerts.call_args.args[0]
    assert (
        f"`{split}`: 600 requests (300 internal, 300 external), 2d old, "
        f"<{monitor._admin_users_url(split)}>"
    ) in message
    assert paying not in message
    assert quiet not in message
    assert "@" not in message


@pytest.mark.asyncio(loop_scope="session")
async def test_reports_three_accounts_without_a_plan_on_one_ip_as_a_group(
    account, clock, alerts
):
    trio = [await account(NO_PLAN) for _ in range(3)]
    paying = await account(PAYING)
    pair = [await account(NO_PLAN) for _ in range(2)]
    pair_payer = await account(PAYING)
    for user_id in [*trio, paying]:
        await _ip_event(user_id, SHARED_IP, clock.at(9))
    for user_id in [*pair, pair_payer]:
        await _ip_event(user_id, OTHER_IP, clock.at(10))
    await _requests(trio[0], internal=4, external=1)

    clock.at(0, day_offset=1)
    await monitor.report_no_plan_api_activity()

    alerts.assert_called_once()
    message = alerts.call_args.args[0]
    assert "- 3 accounts:" in message
    assert all(f"`{user_id}`" in message for user_id in trio)
    assert f"`{trio[0]}`: 5 requests (4 internal, 1 external), 2d old" in message
    # The second IP has three accounts, but only two without a plan.
    assert not any(user_id in message for user_id in [paying, *pair, pair_payer])
    assert SHARED_IP not in message and OTHER_IP not in message


@pytest.mark.asyncio(loop_scope="session")
async def test_sends_nothing_when_nothing_crosses_a_threshold(account, clock, alerts):
    below = await account(NO_PLAN)
    pair = [await account(NO_PLAN) for _ in range(2)]
    for user_id in pair:
        await _ip_event(user_id, SHARED_IP, clock.at(9))
    await _requests(below, internal=499)

    clock.at(0, day_offset=1)
    result = await monitor.report_no_plan_api_activity()

    alerts.assert_not_called()
    assert "No account without a plan crossed a threshold" in result


async def _requests(user_id: str, internal: int = 0, external: int = 0) -> None:
    for _ in range(internal):
        await api_activity.record_api_request(user_id, "internal")
    for _ in range(external):
        await api_activity.record_api_request(user_id, "external")


async def _ip_event(user_id: str, ip: str, at: datetime) -> None:
    await AuthIPEvent.prisma().create(
        data={
            "userId": user_id,
            "kind": AuthIPEventKind.SIGN_IN,
            "ipAddress": ip,
            "createdAt": at,
        }
    )
