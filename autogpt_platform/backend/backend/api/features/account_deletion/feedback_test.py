"""record_feedback: one anonymous PostHog event per answer or alternative click,
with the plan and account age computed here and nothing that names the person."""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.enums import SubscriptionTier
from pydantic import ValidationError
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.feedback import record_feedback
from backend.api.features.account_deletion.models import DeletionFeedbackRequest
from backend.util.posthog_events import PostHogEvent

MODULE = "backend.api.features.account_deletion.feedback"
USER_ID = "3f1c0e7a-feedback-user"
EMAIL = "leaver@example.com"
TEXT = "Too pricey for what I use it for"


def _user() -> MagicMock:
    return MagicMock(
        id=USER_ID,
        email=EMAIL,
        subscription_tier=SubscriptionTier.MAX,
        created_at=datetime.now(timezone.utc) - timedelta(days=45),
    )


@pytest.fixture
def rate_limit(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.enforce_feedback_rate_limit", new_callable=AsyncMock)


@pytest.fixture
def load_user(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.get_user_by_id", new_callable=AsyncMock, return_value=_user()
    )


@pytest.fixture
def capture(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(f"{MODULE}.capture_anonymous")


def _sent(capture: MagicMock) -> tuple[PostHogEvent, dict]:
    capture.assert_called_once()
    event, properties = capture.call_args.args
    return event, properties


@pytest.mark.asyncio
async def test_an_answer_is_sent_with_the_plan_and_tenure(
    rate_limit: AsyncMock, load_user: AsyncMock, capture: MagicMock
):
    body = DeletionFeedbackRequest(kind="answer", reason="cost", text=TEXT)

    await record_feedback(USER_ID, body)

    assert _sent(capture) == (
        PostHogEvent.ACCOUNT_DELETION_FEEDBACK,
        {
            "reason": "cost",
            "text": TEXT,
            "plan_state": "max",
            "tenure_bucket": "30-90d",
        },
    )
    rate_limit.assert_awaited_once_with(USER_ID)
    load_user.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_an_open_is_counted_with_the_surface_plan_and_tenure(
    rate_limit: AsyncMock, load_user: AsyncMock, capture: MagicMock
):
    body = DeletionFeedbackRequest(kind="opened", surface="paywall")

    await record_feedback(USER_ID, body)

    assert _sent(capture) == (
        PostHogEvent.ACCOUNT_DELETION_STARTED,
        {"surface": "paywall", "plan_state": "max", "tenure_bucket": "30-90d"},
    )
    rate_limit.assert_awaited_once_with(USER_ID)


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "opened"},
        {"kind": "opened", "surface": "settings", "reason": "cost"},
        {"kind": "opened", "surface": "settings", "text": "bye"},
        {"kind": "opened", "surface": "settings", "alternative": "billing"},
        {"kind": "answer", "reason": "cost", "surface": "settings"},
        {"kind": "alternative_clicked", "alternative": "billing", "surface": "paywall"},
    ],
    ids=[
        "open_without_surface",
        "open_with_reason",
        "open_with_text",
        "open_with_alternative",
        "answer_with_surface",
        "click_with_surface",
    ],
)
def test_only_an_open_carries_the_surface(fields: dict):
    assert DeletionFeedbackRequest(kind="opened", surface="settings")
    with pytest.raises(ValidationError):
        DeletionFeedbackRequest(**fields)


@pytest.mark.asyncio
async def test_an_alternative_click_is_sent_without_text(
    rate_limit: AsyncMock, load_user: AsyncMock, capture: MagicMock
):
    body = DeletionFeedbackRequest(
        kind="alternative_clicked", reason="privacy", alternative="manage_apps"
    )

    await record_feedback(USER_ID, body)

    assert _sent(capture) == (
        PostHogEvent.ACCOUNT_DELETION_ALTERNATIVE_CLICKED,
        {
            "reason": "privacy",
            "alternative": "manage_apps",
            "plan_state": "max",
            "tenure_bucket": "30-90d",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, absent",
    [
        (DeletionFeedbackRequest(kind="answer", reason="other"), {"text"}),
        (DeletionFeedbackRequest(kind="answer", text=TEXT), {"reason"}),
        (DeletionFeedbackRequest(kind="answer", reason="other", text="   "), {"text"}),
        (
            DeletionFeedbackRequest(kind="alternative_clicked", alternative="billing"),
            {"reason", "text"},
        ),
    ],
    ids=["no_text", "no_reason", "blank_text", "click_without_reason"],
)
async def test_missing_values_are_left_out(
    rate_limit: AsyncMock,
    load_user: AsyncMock,
    capture: MagicMock,
    body: DeletionFeedbackRequest,
    absent: set[str],
):
    await record_feedback(USER_ID, body)

    _, properties = _sent(capture)
    assert not absent & properties.keys()
    assert None not in properties.values()


@pytest.mark.asyncio
async def test_nothing_identifies_the_person(
    rate_limit: AsyncMock, load_user: AsyncMock, capture: MagicMock
):
    await record_feedback(
        USER_ID, DeletionFeedbackRequest(kind="answer", reason="cost", text=TEXT)
    )

    _, properties = _sent(capture)
    assert set(properties) == {"reason", "text", "plan_state", "tenure_bucket"}
    assert not any(
        USER_ID in str(value) or EMAIL in str(value) for value in properties.values()
    )


@pytest.mark.asyncio
async def test_over_the_limit_nothing_is_sent(
    rate_limit: AsyncMock, load_user: AsyncMock, capture: MagicMock
):
    rate_limit.side_effect = HTTPException(status_code=429, detail="slow down")

    with pytest.raises(HTTPException):
        await record_feedback(USER_ID, DeletionFeedbackRequest(kind="answer"))

    load_user.assert_not_awaited()
    capture.assert_not_called()


@pytest.mark.asyncio
async def test_the_typed_text_is_never_logged(
    rate_limit: AsyncMock,
    load_user: AsyncMock,
    capture: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.DEBUG):
        await record_feedback(
            USER_ID, DeletionFeedbackRequest(kind="answer", reason="cost", text=TEXT)
        )

    assert TEXT not in caplog.text
