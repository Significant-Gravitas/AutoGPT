"""Keeping a closed account: the status the sign-in screen reads, and the
reopen. routes_test.py runs both against the real database."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.restore import (
    TooLateToRestoreError,
    get_deletion_status,
    restore_account,
)
from backend.data.account_deletion import ReopenOutcome
from backend.util.posthog_events import PostHogEvent

MODULE = "backend.api.features.account_deletion.restore"
USER_ID = "user-1"
ERASE_AFTER = datetime.now(timezone.utc) + timedelta(days=10)


@pytest.mark.asyncio
async def test_an_open_account_has_nothing_scheduled(mocker: MockerFixture):
    """And this worker stops answering it as closed from its cache."""
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=None))
    forget = mocker.patch(f"{MODULE}.forget_account_state")

    status = await get_deletion_status(USER_ID)

    assert (status.status, status.erase_after) == ("open", None)
    forget.assert_called_once_with(USER_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"startedAt": datetime(2026, 11, 1, tzinfo=timezone.utc)},
        {"completedAt": datetime(2026, 11, 1, tzinfo=timezone.utc)},
        {"eraseAfter": datetime(2020, 1, 1, tzinfo=timezone.utc)},
    ],
    ids=["erasure-claimed", "erased", "date-passed"],
)
async def test_a_closed_account_past_keeping_is_too_late(
    mocker: MockerFixture, fields: dict
):
    request = MagicMock(
        **{"eraseAfter": ERASE_AFTER, "startedAt": None, "completedAt": None, **fields}
    )
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=request))

    status = await get_deletion_status(USER_ID)

    assert status.status == "too_late"
    assert status.erase_after == request.eraseAfter


@pytest.mark.asyncio
async def test_a_closed_account_shows_its_erasure_date(mocker: MockerFixture):
    request = MagicMock(eraseAfter=ERASE_AFTER, startedAt=None, completedAt=None)
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=request))

    status = await get_deletion_status(USER_ID)

    assert (status.status, status.erase_after) == ("scheduled", ERASE_AFTER)


@pytest.fixture
def edges(mocker: MockerFixture) -> MagicMock:
    """Every call restore_account makes besides the reopen, on one parent."""
    parent = MagicMock()
    for name, kind in (
        ("revoke_issued_access", AsyncMock),
        ("forget_account_state", MagicMock),
        ("forget_refusal", MagicMock),
        ("forget_store_listings", MagicMock),
        ("capture_anonymous", MagicMock),
    ):
        parent.attach_mock(mocker.patch(f"{MODULE}.{name}", new_callable=kind), name)
    mocker.patch(
        f"{MODULE}.get_deletion_request",
        AsyncMock(
            return_value=MagicMock(
                eraseAfter=ERASE_AFTER, startedAt=None, completedAt=None
            )
        ),
    )
    mocker.patch(
        f"{MODULE}.get_user_by_id",
        AsyncMock(
            return_value=MagicMock(created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        ),
    )
    return parent


def _reopen(mocker: MockerFixture, outcome: ReopenOutcome) -> AsyncMock:
    return mocker.patch(f"{MODULE}.reopen_account", AsyncMock(return_value=outcome))


def _called(edges: MagicMock) -> list[str]:
    return [name for name, _, _ in edges.mock_calls]


@pytest.mark.asyncio
async def test_keeping_the_account_lets_the_owner_and_its_work_back_at_once(
    mocker: MockerFixture, edges: MagicMock
):
    _reopen(mocker, ReopenOutcome.REOPENED)

    await restore_account(USER_ID)

    edges.revoke_issued_access.assert_awaited_once_with(USER_ID)
    edges.forget_account_state.assert_called_once_with(USER_ID)
    edges.forget_refusal.assert_called_once_with(USER_ID)
    edges.forget_store_listings.assert_called_once_with()
    [(event, properties)] = [c.args for c in edges.capture_anonymous.call_args_list]
    assert event is PostHogEvent.ACCOUNT_DELETION_KEPT
    assert properties["tenure_bucket"] == "90d+"
    assert 0 <= properties["days_left"] <= 14


@pytest.mark.asyncio
async def test_an_open_account_is_let_in_but_not_counted(
    mocker: MockerFixture, edges: MagicMock
):
    _reopen(mocker, ReopenOutcome.NOT_CLOSED)

    await restore_account(USER_ID)

    edges.forget_account_state.assert_called_once_with(USER_ID)
    edges.capture_anonymous.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_count_still_keeps_the_account(
    mocker: MockerFixture, edges: MagicMock
):
    _reopen(mocker, ReopenOutcome.REOPENED)
    edges.capture_anonymous.side_effect = RuntimeError("PostHog down")

    await restore_account(USER_ID)

    edges.forget_store_listings.assert_called_once_with()


@pytest.mark.asyncio
async def test_too_late_changes_nothing(mocker: MockerFixture, edges: MagicMock):
    """The reopen refuses it: the erasure claimed the request between the
    read and the reopen."""
    _reopen(mocker, ReopenOutcome.TOO_LATE)
    mocker.patch(
        f"{MODULE}.get_deletion_request",
        AsyncMock(
            return_value=MagicMock(
                eraseAfter=ERASE_AFTER,
                startedAt=datetime(2026, 11, 1, tzinfo=timezone.utc),
                completedAt=None,
            )
        ),
    )

    with pytest.raises(TooLateToRestoreError):
        await restore_account(USER_ID)

    assert not edges.mock_calls


@pytest.mark.asyncio
async def test_keys_made_after_the_close_are_revoked_before_it_opens(
    mocker: MockerFixture, edges: MagicMock
):
    """None can be made while the account is closed, so an active one was
    made in the seconds after the close, on a worker whose cache still let
    an old token through."""
    reopen = _reopen(mocker, ReopenOutcome.REOPENED)
    edges.attach_mock(reopen, "reopen_account")

    await restore_account(USER_ID)

    assert _called(edges)[:2] == ["revoke_issued_access", "reopen_account"]


@pytest.mark.asyncio
async def test_a_failed_revoke_leaves_the_account_closed(
    mocker: MockerFixture, edges: MagicMock
):
    """The owner can try again; opened first, a retry would find nothing to
    reopen and the key would keep working."""
    reopen = _reopen(mocker, ReopenOutcome.REOPENED)
    edges.revoke_issued_access.side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await restore_account(USER_ID)

    reopen.assert_not_awaited()
    edges.forget_account_state.assert_not_called()


@pytest.mark.asyncio
async def test_an_open_account_revokes_nothing(mocker: MockerFixture, edges: MagicMock):
    """A second Keep, or one from an account never closed: its keys are the
    owner's own."""
    _reopen(mocker, ReopenOutcome.NOT_CLOSED)
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=None))

    await restore_account(USER_ID)

    edges.revoke_issued_access.assert_not_awaited()
