import asyncio
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
    _compute_invited_user_tally_seed,
    _fetch_existing_emails,
    _get_invited_user_tally_understanding,
    bulk_create_invited_users_from_file,
    check_invite_eligibility,
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
    schedule.assert_called_once_with("invite-1", tally_mode="default")


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
    schedule.assert_called_once_with("invite-1", tally_mode="default")


@pytest.mark.asyncio
async def test_fetch_existing_emails_batches_queries(
    mocker: pytest_mock.MockerFixture,
) -> None:
    emails = [f"user{i}@example.com" for i in range(1200)]
    active_targets = {emails[0], emails[500], emails[1199]}
    invited_targets = {emails[1], emails[501], emails[1188]}

    user_chunk_sizes: list[int] = []
    invited_chunk_sizes: list[int] = []

    async def find_many_users(*, where):
        email_chunk = where["email"]["in"]
        user_chunk_sizes.append(len(email_chunk))
        return [
            SimpleNamespace(email=email)
            for email in email_chunk
            if email in active_targets
        ]

    async def find_many_invited(*, where):
        email_chunk = where["email"]["in"]
        invited_chunk_sizes.append(len(email_chunk))
        return [
            SimpleNamespace(email=email)
            for email in email_chunk
            if email in invited_targets
        ]

    user_repo = Mock()
    user_repo.find_many = AsyncMock(side_effect=find_many_users)
    invited_user_repo = Mock()
    invited_user_repo.find_many = AsyncMock(side_effect=find_many_invited)

    mocker.patch(
        "backend.data.invited_user.prisma.models.User.prisma",
        return_value=user_repo,
    )
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=invited_user_repo,
    )

    active_emails, invited_emails = await _fetch_existing_emails(emails)

    assert active_emails == active_targets
    assert invited_emails == invited_targets
    assert user_chunk_sizes == [500, 500, 200]
    assert invited_chunk_sizes == [500, 500, 200]


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
    mocker.patch(
        "backend.data.invited_user._settings.config.enable_invite_gate",
        True,
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
    assert [call.kwargs["tally_mode"] for call in create_invited.await_args_list] == [
        "bulk",
        "bulk",
    ]


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
    assert [call.kwargs["tally_mode"] for call in create_invited.await_args_list] == [
        "bulk",
        "bulk",
    ]


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
    assert create_invited.await_args_list[0].kwargs["tally_mode"] == "bulk"


@pytest.mark.asyncio
async def test_bulk_create_invited_users_limits_create_concurrency(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.data.invited_user._fetch_existing_emails",
        AsyncMock(return_value=(set(), set())),
    )

    current_in_flight = 0
    peak_in_flight = 0
    release_creates = asyncio.Event()
    fifty_started = asyncio.Event()

    async def create_invited_side_effect(
        email: str, name: str | None, *, tally_mode: str
    ):
        nonlocal current_in_flight, peak_in_flight
        assert tally_mode == "bulk"
        current_in_flight += 1
        peak_in_flight = max(peak_in_flight, current_in_flight)
        if peak_in_flight >= 50:
            fifty_started.set()
        await release_creates.wait()
        current_in_flight -= 1
        return _invited_user_record()

    create_invited = mocker.patch(
        "backend.data.invited_user.create_invited_user",
        AsyncMock(side_effect=create_invited_side_effect),
    )

    async def run_bulk_create() -> None:
        result = await bulk_create_invited_users_from_file(
            "invites.csv",
            (
                "email,name\n"
                + "\n".join(f"user{i}@example.com,User {i}" for i in range(120))
            ).encode("utf-8"),
        )
        assert result.created_count == 120
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert [row.status for row in result.results] == ["CREATED"] * 120

    bulk_task = asyncio.create_task(run_bulk_create())
    await asyncio.wait_for(fifty_started.wait(), timeout=2)
    assert peak_in_flight == 50
    release_creates.set()
    await bulk_task
    assert create_invited.await_count == 120


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
        "backend.data.invited_user._get_invited_user_tally_understanding",
        AsyncMock(
            side_effect=TallyExtractionTimeoutError(
                attempts=3,
                timeout_seconds=30,
            )
        ),
    )
    warning = mocker.patch("backend.data.invited_user.logger.warning")
    exception = mocker.patch("backend.data.invited_user.logger.exception")

    await _compute_invited_user_tally_seed("invite-1")

    assert invited_user_repo.update.await_count == 2
    assert warning.call_count == 1
    exception.assert_not_called()
    failure_update = invited_user_repo.update.await_args_list[-1].kwargs["data"]
    assert failure_update["tallyStatus"] == prisma.enums.TallyComputationStatus.FAILED
    assert "timed out after 3 attempts" in failure_update["tallyError"]


@pytest.mark.asyncio
async def test_get_invited_user_tally_understanding_bulk_serializes_fetch_and_limits_extraction(
    mocker: pytest_mock.MockerFixture,
) -> None:
    real_sleep = asyncio.sleep
    fetch_sleep = mocker.patch(
        "backend.data.invited_user.asyncio.sleep",
        AsyncMock(),
    )

    current_fetches = 0
    peak_fetches = 0
    current_extractions = 0
    peak_extractions = 0
    extraction_started = 0
    three_extractions_started = asyncio.Event()
    release_extractions = asyncio.Event()

    async def get_submission_side_effect(email: str, *, require_api_key: bool):
        nonlocal current_fetches, peak_fetches
        assert require_api_key is True
        current_fetches += 1
        peak_fetches = max(peak_fetches, current_fetches)
        await real_sleep(0)
        current_fetches -= 1
        return (
            {"responses": [{"questionId": "q1", "value": email}]},
            [{"id": "q1", "label": "Email", "type": "INPUT_EMAIL"}],
        )

    async def extract_side_effect(formatted: str, **kwargs):
        nonlocal current_extractions, peak_extractions, extraction_started
        assert kwargs == {"timeout_seconds": 30, "max_attempts": 3}
        current_extractions += 1
        extraction_started += 1
        peak_extractions = max(peak_extractions, current_extractions)
        if extraction_started >= 3:
            three_extractions_started.set()
        await release_extractions.wait()
        current_extractions -= 1
        return object()

    mocker.patch(
        "backend.data.invited_user.get_tally_submission_by_email",
        AsyncMock(side_effect=get_submission_side_effect),
    )
    mocker.patch(
        "backend.data.invited_user.extract_business_understanding_from_tally",
        AsyncMock(side_effect=extract_side_effect),
    )

    tasks = [
        asyncio.create_task(
            _get_invited_user_tally_understanding(
                f"user{i}@example.com",
                tally_mode="bulk",
            )
        )
        for i in range(4)
    ]

    await asyncio.wait_for(three_extractions_started.wait(), timeout=2)
    assert peak_fetches == 1
    assert peak_extractions == 3
    assert fetch_sleep.await_count == 4
    assert all(call.args == (0.6,) for call in fetch_sleep.await_args_list)

    release_extractions.set()
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# check_invite_eligibility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_invite_eligibility_returns_true_for_invited(
    mocker: pytest_mock.MockerFixture,
) -> None:
    invited = _invited_user_db_record(status=prisma.enums.InvitedUserStatus.INVITED)
    repo = Mock()
    repo.find_unique = AsyncMock(return_value=invited)
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=repo,
    )

    result = await check_invite_eligibility("invited@example.com")
    assert result is True
    repo.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_invite_eligibility_returns_false_for_no_record(
    mocker: pytest_mock.MockerFixture,
) -> None:
    repo = Mock()
    repo.find_unique = AsyncMock(return_value=None)
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=repo,
    )

    result = await check_invite_eligibility("unknown@example.com")
    assert result is False


@pytest.mark.asyncio
async def test_check_invite_eligibility_returns_false_for_claimed(
    mocker: pytest_mock.MockerFixture,
) -> None:
    claimed = _invited_user_db_record(status=prisma.enums.InvitedUserStatus.CLAIMED)
    repo = Mock()
    repo.find_unique = AsyncMock(return_value=claimed)
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=repo,
    )

    result = await check_invite_eligibility("claimed@example.com")
    assert result is False


@pytest.mark.asyncio
async def test_check_invite_eligibility_returns_false_for_revoked(
    mocker: pytest_mock.MockerFixture,
) -> None:
    revoked = _invited_user_db_record(status=prisma.enums.InvitedUserStatus.REVOKED)
    repo = Mock()
    repo.find_unique = AsyncMock(return_value=revoked)
    mocker.patch(
        "backend.data.invited_user.prisma.models.InvitedUser.prisma",
        return_value=repo,
    )

    result = await check_invite_eligibility("revoked@example.com")
    assert result is False
