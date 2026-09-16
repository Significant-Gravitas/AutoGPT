from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import prisma.errors
import pytest
import pytest_mock

from backend.data.attribution import (
    SIGNUP_ATTRIBUTION_WINDOW,
    UserAttributionInput,
    record_user_attribution,
)

_COLUMNS = (
    "anonymousId",
    "posthogDistinctId",
    "datafastVisitorId",
    "datafastSessionId",
    "landingPath",
    "referrer",
    "utmSource",
    "utmMedium",
    "utmCampaign",
    "signupMethod",
)


def _row(**values: str) -> MagicMock:
    row = MagicMock()
    row.userId = "user-1"
    row.createdAt = datetime(2026, 9, 4, tzinfo=timezone.utc)
    row.updatedAt = row.createdAt
    for column in _COLUMNS:
        setattr(row, column, values.get(column))
    return row


def _unique_violation() -> prisma.errors.UniqueViolationError:
    return prisma.errors.UniqueViolationError(
        {
            "user_facing_error": {
                "error_code": "P2002",
                "message": "Unique constraint failed on the fields: (`userId`)",
            }
        }
    )


@pytest.fixture(autouse=True)
def account(mocker: pytest_mock.MockerFixture) -> MagicMock:
    """A brand-new account, so acquisition fields are accepted by default."""
    row = MagicMock()
    row.createdAt = datetime.now(timezone.utc)
    user_client = MagicMock()
    user_client.find_unique = AsyncMock(return_value=row)
    mocker.patch("prisma.models.User.prisma", return_value=user_client)
    return row


@pytest.fixture
def client(mocker: pytest_mock.MockerFixture) -> MagicMock:
    client = MagicMock()
    client.find_unique = AsyncMock(return_value=None)
    client.create = AsyncMock()
    client.update = AsyncMock()
    mocker.patch("prisma.models.UserAttribution.prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_first_report_creates_the_row(client: MagicMock) -> None:
    client.create.return_value = _row(anonymousId="anon-1", landingPath="/pricing")

    result = await record_user_attribution(
        "user-1", UserAttributionInput(anonymous_id="anon-1", landing_path="/pricing")
    )

    client.create.assert_awaited_once_with(
        data={"userId": "user-1", "anonymousId": "anon-1", "landingPath": "/pricing"}
    )
    client.update.assert_not_awaited()
    assert result.anonymous_id == "anon-1"
    assert result.landing_path == "/pricing"


@pytest.mark.asyncio
async def test_losing_a_concurrent_create_fills_the_winners_empty_fields(
    client: MagicMock,
) -> None:
    winner = _row(anonymousId="anon-from-other-tab")
    client.find_unique.side_effect = [None, winner]
    client.create.side_effect = _unique_violation()
    client.update.return_value = _row(
        anonymousId="anon-from-other-tab", landingPath="/pricing"
    )

    result = await record_user_attribution(
        "user-1",
        UserAttributionInput(anonymous_id="anon-mine", landing_path="/pricing"),
    )

    client.create.assert_awaited_once()
    # The winner's anonymous id stands; only the field it left empty is filled.
    client.update.assert_awaited_once_with(
        where={"userId": "user-1"}, data={"landingPath": "/pricing"}
    )
    assert result.anonymous_id == "anon-from-other-tab"
    assert result.landing_path == "/pricing"


@pytest.mark.asyncio
async def test_a_second_report_never_overwrites_filled_fields(
    client: MagicMock,
) -> None:
    client.find_unique.return_value = _row(anonymousId="anon-1", landingPath="/")

    result = await record_user_attribution(
        "user-1", UserAttributionInput(anonymous_id="anon-2", landing_path="/other")
    )

    client.create.assert_not_awaited()
    client.update.assert_not_awaited()
    assert result.anonymous_id == "anon-1"
    assert result.landing_path == "/"


@pytest.mark.asyncio
async def test_an_account_older_than_the_window_records_no_signup_channel(
    client: MagicMock, account: MagicMock
) -> None:
    account.createdAt = (
        datetime.now(timezone.utc) - SIGNUP_ATTRIBUTION_WINDOW - timedelta(minutes=1)
    )
    client.create.return_value = _row(anonymousId="anon-1")

    await record_user_attribution(
        "user-1",
        UserAttributionInput(
            anonymous_id="anon-1",
            landing_path="/pricing",
            utm_source="newsletter",
            signup_method="google",
        ),
    )

    # The join key still lands; where they "came from" this session does not.
    client.create.assert_awaited_once_with(
        data={"userId": "user-1", "anonymousId": "anon-1"}
    )


@pytest.mark.asyncio
async def test_an_account_inside_the_window_records_the_channel(
    client: MagicMock, account: MagicMock
) -> None:
    account.createdAt = (
        datetime.now(timezone.utc) - SIGNUP_ATTRIBUTION_WINDOW + timedelta(minutes=1)
    )
    client.create.return_value = _row(anonymousId="anon-1", utmSource="newsletter")

    await record_user_attribution(
        "user-1",
        UserAttributionInput(anonymous_id="anon-1", utm_source="newsletter"),
    )

    client.create.assert_awaited_once_with(
        data={"userId": "user-1", "anonymousId": "anon-1", "utmSource": "newsletter"}
    )


@pytest.mark.asyncio
async def test_a_missing_account_row_records_no_signup_channel(
    client: MagicMock, mocker: pytest_mock.MockerFixture
) -> None:
    user_client = MagicMock()
    user_client.find_unique = AsyncMock(return_value=None)
    mocker.patch("prisma.models.User.prisma", return_value=user_client)
    client.create.return_value = _row(anonymousId="anon-1")

    await record_user_attribution(
        "user-1",
        UserAttributionInput(anonymous_id="anon-1", landing_path="/pricing"),
    )

    client.create.assert_awaited_once_with(
        data={"userId": "user-1", "anonymousId": "anon-1"}
    )


@pytest.mark.asyncio
async def test_identity_only_reports_never_read_the_account(
    client: MagicMock, mocker: pytest_mock.MockerFixture
) -> None:
    user_client = MagicMock()
    user_client.find_unique = AsyncMock(return_value=None)
    mocker.patch("prisma.models.User.prisma", return_value=user_client)
    client.create.return_value = _row(anonymousId="anon-1")

    await record_user_attribution("user-1", UserAttributionInput(anonymous_id="anon-1"))

    user_client.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_update_path_also_withholds_the_signup_channel(
    client: MagicMock, account: MagicMock
) -> None:
    """The filter runs before the create/update branch; pin it on both, so moving
    it into the create branch alone cannot pass."""
    account.createdAt = (
        datetime.now(timezone.utc) - SIGNUP_ATTRIBUTION_WINDOW - timedelta(minutes=1)
    )
    client.find_unique.return_value = _row()
    client.update.return_value = _row(anonymousId="anon-1")

    await record_user_attribution(
        "user-1",
        UserAttributionInput(
            anonymous_id="anon-1", landing_path="/pricing", utm_source="newsletter"
        ),
    )

    client.create.assert_not_awaited()
    client.update.assert_awaited_once_with(
        where={"userId": "user-1"}, data={"anonymousId": "anon-1"}
    )
