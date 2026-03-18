from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import prisma.enums
import prisma.models
import pytest
import pytest_mock

from backend.util.exceptions import NotAuthorizedError, PreconditionFailed

from .invited_user import (
    InvitedUserRecord,
    _compute_invited_user_tally_seed_inner,
    bulk_create_invited_users_from_file,
    create_invited_user,
    get_or_activate_user,
    retry_invited_user_tally,
)
from .tally import TallyExtractionTimeoutError


def _invited_user_db_record(
    *,
    status: prisma.enums.InvitedUserStatus = prisma.enums.InvitedUserStatus.INVITED,
    tally_understanding: dict | None = None,
):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id="invite-1",
        email="invited@example.com",
        status=status,
        authUserId=None,
        name="Invited User",
        tallyUnderstanding=tally_understanding,
        tallyStatus=prisma.enums.TallyComputationStatus.PENDING,
        tallyComputedAt=None,
        tallyError=None,
        createdAt=now,
        updatedAt=now,
    )


def _invited_user_record(
    *,
    status: prisma.enums.InvitedUserStatus = prisma.enums.InvitedUserStatus.INVITED,
    tally_understanding: dict | None = None,
):
    return InvitedUserRecord.from_db(
        cast(
            prisma.models.InvitedUser,
            _invited_user_db_record(
                status=status,
                tally_understanding=tally_understanding,
            ),
        )
    )


def _user_db_record():
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id="auth-user-1",
        email="invited@example.com",
        emailVerified=True,
        name="Invited User",
        createdAt=now,
        updatedAt=now,
        metadata={},
        integrations="",
        stripeCustomerId=None,
        topUpConfig=None,
        maxEmailsPerDay=3,
        notifyOnAgentRun=True,
        notifyOnZeroBalance=True,
        notifyOnLowBalance=True,
        notifyOnBlockExecutionFailed=True,
        notifyOnContinuousAgentError=True,
        notifyOnDailySummary=True,
        notifyOnWeeklySummary=True,
        notifyOnMonthlySummary=True,
        notifyOnAgentApproved=True,
        notifyOnAgentRejected=True,
        timezone="not-set",
    )


@pytest.mark.asyncio
async def test_create_invited_user_rejects_existing_active_user(
    mocker: pytest_mock.MockerFixture,
) -> None:
    user_repo = Mock()
    user_repo.find_unique = AsyncMock(return_value=_user_db_record())
    invited_user_repo = Mock()
    invited_user_repo.find_unique = AsyncMock()

    mocker.patch(
        "backend.data.invited_user.prisma.models.User.prisma", return_value=user_repo
    )
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )

    with pytest.raises(PreconditionFailed):
        await create_invited_user("Invited@example.com")


@pytest.mark.asyncio
async def test_create_invited_user_schedules_tally_seed(
    mocker: pytest_mock.MockerFixture,
) -> None:
    user_repo = Mock()
    user_repo.find_unique = AsyncMock(return_value=None)
    invited_user_repo = Mock()
    invited_user_repo.find_unique = AsyncMock(return_value=None)
    invited_user_repo.create = AsyncMock(return_value=_invited_user_db_record())
    schedule = mocker.patch(
        "backend.data.invited_user.schedule_invited_user_tally_precompute"
    )

    mocker.patch(
        "backend.data.invited_user.prisma.models.User.prisma", return_value=user_repo
    )
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )

    invited_user = await create_invited_user("Invited@example.com", "Invited User")

    assert invited_user.email == "invited@example.com"
    invited_user_repo.create.assert_awaited_once()
    schedule.assert_called_once_with("invite-1")


@pytest.mark.asyncio
async def test_retry_invited_user_tally_resets_state_and_schedules(
    mocker: pytest_mock.MockerFixture,
) -> None:
    invited_user_repo = Mock()
    invited_user_repo.find_unique = AsyncMock(return_value=_invited_user_db_record())
    invited_user_repo.update = AsyncMock(return_value=_invited_user_db_record())
    schedule = mocker.patch(
        "backend.data.invited_user.schedule_invited_user_tally_precompute"
    )

    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )

    invited_user = await retry_invited_user_tally("invite-1")

    assert invited_user.id == "invite-1"
    invited_user_repo.update.assert_awaited_once()
    schedule.assert_called_once_with("invite-1")


@pytest.mark.asyncio
async def test_get_or_activate_user_requires_invite(
    mocker: pytest_mock.MockerFixture,
) -> None:
    invited_user_repo = Mock()
    invited_user_repo.find_unique = AsyncMock(return_value=None)

    mock_get_user_by_id = AsyncMock(side_effect=ValueError("User not found"))
    mock_get_user_by_id.cache_delete = Mock()
    mocker.patch(
        "backend.data.invited_user.get_user_by_id",
        mock_get_user_by_id,
    )
    mocker.patch(
        "backend.data.invited_user._settings.config.enable_invite_gate",
        True,
    )
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )

    with pytest.raises(NotAuthorizedError):
        await get_or_activate_user(
            {"sub": "auth-user-1", "email": "invited@example.com"}
        )


@pytest.mark.asyncio
async def test_get_or_activate_user_creates_user_from_invite(
    mocker: pytest_mock.MockerFixture,
) -> None:
    tx = object()
    invited_user = _invited_user_db_record(
        tally_understanding={"user_name": "Invited User", "industry": "Automation"}
    )
    created_user = _user_db_record()

    outside_user_repo = Mock()
    # Only called once at post-transaction verification (line 741);
    # get_user_by_id (line 657) uses prisma.user.find_unique, not this mock.
    outside_user_repo.find_unique = AsyncMock(return_value=created_user)

    inside_user_repo = Mock()
    inside_user_repo.find_unique = AsyncMock(return_value=None)
    inside_user_repo.create = AsyncMock(return_value=created_user)

    outside_invited_repo = Mock()
    outside_invited_repo.find_unique = AsyncMock(return_value=invited_user)

    inside_invited_repo = Mock()
    inside_invited_repo.find_unique = AsyncMock(return_value=invited_user)
    inside_invited_repo.update = AsyncMock(return_value=invited_user)

    def user_prisma(client=None):
        return inside_user_repo if client is tx else outside_user_repo

    def invited_user_prisma(client=None):
        return inside_invited_repo if client is tx else outside_invited_repo

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    # Mock get_user_by_id since it uses prisma.user.find_unique (global client),
    # not prisma.models.User.prisma().find_unique which we mock above.
    mock_get_user_by_id = AsyncMock(side_effect=ValueError("User not found"))
    mock_get_user_by_id.cache_delete = Mock()
    mocker.patch(
        "backend.data.invited_user.get_user_by_id",
        mock_get_user_by_id,
    )
    mock_get_user_by_email = AsyncMock()
    mock_get_user_by_email.cache_delete = Mock()
    mocker.patch(
        "backend.data.invited_user.get_user_by_email",
        mock_get_user_by_email,
    )
    ensure_profile = mocker.patch(
        "backend.data.invited_user._ensure_default_profile",
        AsyncMock(),
    )
    ensure_onboarding = mocker.patch(
        "backend.data.invited_user._ensure_default_onboarding",
        AsyncMock(),
    )
    apply_tally = mocker.patch(
        "backend.data.invited_user._apply_tally_understanding",
        AsyncMock(),
    )
    mocker.patch("backend.data.invited_user.transaction", fake_transaction)
    mocker.patch(
        "backend.data.invited_user.prisma.models.User.prisma", side_effect=user_prisma
    )
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        side_effect=invited_user_prisma,
    )

    user = await get_or_activate_user(
        {
            "sub": "auth-user-1",
            "email": "Invited@example.com",
            "user_metadata": {"name": "Invited User"},
        }
    )

    assert user.id == "auth-user-1"
    inside_user_repo.create.assert_awaited_once()
    inside_invited_repo.update.assert_awaited_once()
    ensure_profile.assert_awaited_once()
    ensure_onboarding.assert_awaited_once_with("auth-user-1", tx)
    apply_tally.assert_awaited_once_with("auth-user-1", invited_user, tx)


@pytest.mark.asyncio
async def test_bulk_create_invited_users_from_text_file(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.data.invited_user._fetch_existing_emails",
        AsyncMock(return_value=(set(), set())),
    )
    create_invited = mocker.patch(
        "backend.data.invited_user.create_invited_user",
        AsyncMock(
            side_effect=[
                _invited_user_record(),
                _invited_user_record(),
            ]
        ),
    )

    result = await bulk_create_invited_users_from_file(
        "invites.txt",
        b"Invited@example.com\nsecond@example.com\n",
    )

    assert result.created_count == 2
    assert result.skipped_count == 0
    assert result.error_count == 0
    assert [row.status for row in result.results] == ["CREATED", "CREATED"]
    assert create_invited.await_count == 2


@pytest.mark.asyncio
async def test_bulk_create_invited_users_handles_csv_duplicates_and_invalid_rows(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.data.invited_user._fetch_existing_emails",
        AsyncMock(return_value=(set(), set())),
    )
    create_invited = mocker.patch(
        "backend.data.invited_user.create_invited_user",
        AsyncMock(
            side_effect=[
                _invited_user_record(),
                PreconditionFailed("An invited user with this email already exists"),
            ]
        ),
    )

    result = await bulk_create_invited_users_from_file(
        "invites.csv",
        (
            "email,name\n"
            "valid@example.com,Valid User\n"
            "not-an-email,Bad Row\n"
            "valid@example.com,Duplicate In File\n"
            "existing@example.com,Existing User\n"
        ).encode("utf-8"),
    )

    assert result.created_count == 1
    assert result.skipped_count == 2
    assert result.error_count == 1
    # Results are returned in upload row order after processing.
    assert [row.status for row in result.results] == [
        "CREATED",
        "ERROR",
        "SKIPPED",
        "SKIPPED",
    ]
    assert create_invited.await_count == 2


@pytest.mark.asyncio
async def test_bulk_create_skips_already_invited_emails(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.data.invited_user._fetch_existing_emails",
        AsyncMock(
            return_value=(
                {"active@example.com"},
                {"invited@example.com"},
            )
        ),
    )
    create_invited = mocker.patch(
        "backend.data.invited_user.create_invited_user",
        AsyncMock(return_value=_invited_user_record()),
    )

    result = await bulk_create_invited_users_from_file(
        "invites.csv",
        (
            "email,name\n"
            "active@example.com,Active User\n"
            "invited@example.com,Already Invited\n"
            "new@example.com,New User\n"
        ).encode("utf-8"),
    )

    assert result.created_count == 1
    assert result.skipped_count == 2
    assert result.error_count == 0
    assert [row.status for row in result.results] == [
        "SKIPPED",
        "SKIPPED",
        "CREATED",
    ]
    assert result.results[0].message == "An active user with this email already exists"
    assert result.results[1].message == "An invited user with this email already exists"
    assert create_invited.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (
            "invites.txt",
            "\n".join(f"user{i}@example.com" for i in range(5001)).encode("utf-8"),
        ),
        (
            "invites.csv",
            (
                "email,name\n"
                + "\n".join(f"user{i}@example.com,User {i}" for i in range(5001))
            ).encode("utf-8"),
        ),
    ],
)
async def test_bulk_create_rejects_files_over_5000_rows(
    filename: str,
    content: bytes,
    mocker: pytest_mock.MockerFixture,
) -> None:
    fetch_existing = mocker.patch(
        "backend.data.invited_user._fetch_existing_emails",
        AsyncMock(return_value=(set(), set())),
    )
    create_invited = mocker.patch(
        "backend.data.invited_user.create_invited_user",
        AsyncMock(return_value=_invited_user_record()),
    )

    with pytest.raises(ValueError, match="maximum of 5000 users"):
        await bulk_create_invited_users_from_file(filename, content)

    fetch_existing.assert_not_awaited()
    create_invited.assert_not_awaited()


@pytest.mark.asyncio
async def test_compute_invited_user_tally_seed_handles_timeout_without_traceback(
    mocker: pytest_mock.MockerFixture,
) -> None:
    invited_user_repo = Mock()
    invited_user_repo.find_unique = AsyncMock(return_value=_invited_user_db_record())
    invited_user_repo.update = AsyncMock(return_value=_invited_user_db_record())

    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )
    mocker.patch(
        "backend.data.invited_user.get_redis_async",
        AsyncMock(return_value=None),
    )
    mocker.patch(
        "backend.data.invited_user.get_business_understanding_input_from_tally",
        AsyncMock(
            side_effect=TallyExtractionTimeoutError(
                attempts=3,
                timeout_seconds=30,
            )
        ),
    )
    warning = mocker.patch("backend.data.invited_user.logger.warning")
    exception = mocker.patch("backend.data.invited_user.logger.exception")

    await _compute_invited_user_tally_seed_inner("invite-1")

    assert invited_user_repo.update.await_count == 2
    assert warning.call_count == 1
    exception.assert_not_called()
    failure_update = invited_user_repo.update.await_args_list[-1].kwargs["data"]
    assert failure_update["tallyStatus"] == prisma.enums.TallyComputationStatus.FAILED
    assert "timed out after 3 attempts" in failure_update["tallyError"]
