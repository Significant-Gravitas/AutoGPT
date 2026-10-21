"""Tests for ``copilot.feedback``: reading the chat UI's message ids, and
sending a rating to PostHog LLM analytics.

The route tests (``backend/api/features/chat/feedback_test.py``) cover the
Postgres and Langfuse halves.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from prisma.models import ChatMessage as PrismaChatMessage

from backend.copilot import feedback, feedback_db
from backend.copilot.feedback import (
    record_message_feedback,
    sequence_from_ui_message_id,
)
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

SESSION_ID = "8d2f5c1e-2b7a-4c55-9d0e-3f6a1b2c4d5e"
USER_ID = "3b1f0e2a-7c4d-4e5f-8a9b-0c1d2e3f4a5b"
MESSAGE_ROW_ID = "0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b"
FEEDBACK_ID = "c3a1b2d4-e5f6-4789-8abc-def012345678"


@pytest.fixture
def reply(mocker: pytest_mock.MockerFixture) -> PrismaChatMessage:
    message = PrismaChatMessage(
        id=MESSAGE_ROW_ID,
        createdAt=datetime.now(UTC),
        sessionId=SESSION_ID,
        role="assistant",
        content="Here is the report.",
        sequence=7,
        langfuseTraceId="1edf31f11b1693cc6103f358c1481694",
    )
    mocker.patch.object(
        feedback_db, "get_rateable_message", new=AsyncMock(return_value=message)
    )
    mocker.patch.object(
        feedback_db,
        "upsert_message_feedback",
        new=AsyncMock(return_value=FEEDBACK_ID),
    )
    return message


@pytest.fixture
def langfuse_score(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    secrets = feedback.settings.secrets
    mocker.patch.object(secrets, "langfuse_public_key", "pk-lf-test")
    mocker.patch.object(secrets, "langfuse_secret_key", "sk-lf-test")
    langfuse = MagicMock()
    langfuse.async_api.score.create = AsyncMock()
    mocker.patch.object(feedback, "get_client", return_value=langfuse)
    return langfuse.async_api.score.create


async def _rate(**overrides):
    kwargs = {
        "user_id": USER_ID,
        "session_id": SESSION_ID,
        "message_id": f"{SESSION_ID}-seq-7",
        "score_name": "user-feedback",
        "score_value": 0,
        "comment": "It edited the wrong file",
        **overrides,
    }
    return await record_message_feedback(**kwargs)


@pytest.mark.asyncio
@pytest.mark.usefixtures("reply")
async def test_thumbs_down_is_an_ai_metric_on_the_user_and_chat(
    mocker: pytest_mock.MockerFixture, langfuse_score: AsyncMock
) -> None:
    capture = mocker.patch.object(feedback, "capture")

    response = await _rate()

    assert response is not None
    capture.assert_called_once_with(
        USER_ID,
        PostHogEvent.AI_METRIC,
        {
            "$ai_metric_name": "user-feedback",
            "$ai_metric_value": 0,
            "$ai_session_id": SESSION_ID,
            "chat_session_id": SESSION_ID,
            "message_id": MESSAGE_ROW_ID,
            "message_sequence": 7,
            "feedback_id": FEEDBACK_ID,
            "impersonated": False,
            "$ai_feedback_text": "It edited the wrong file",
        },
    )
    langfuse_score.assert_awaited_once()
    assert response.langfuse_target == "trace"


@pytest.mark.asyncio
@pytest.mark.usefixtures("reply", "langfuse_score")
async def test_copy_carries_no_feedback_text(
    mocker: pytest_mock.MockerFixture,
) -> None:
    capture = mocker.patch.object(feedback, "capture")

    await _rate(score_name="copy", score_value=1, comment=None)

    properties = capture.call_args.args[2]
    assert properties["$ai_metric_name"] == "copy"
    assert properties["$ai_metric_value"] == 1
    assert "$ai_feedback_text" not in properties


@pytest.mark.asyncio
@pytest.mark.usefixtures("reply", "langfuse_score")
async def test_impersonated_rating_is_marked(
    mocker: pytest_mock.MockerFixture,
) -> None:
    capture = mocker.patch.object(feedback, "capture")

    await _rate(impersonated=True)

    assert capture.call_args.args[2]["impersonated"] is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("reply")
async def test_posthog_failure_keeps_the_rating_and_the_langfuse_score(
    mocker: pytest_mock.MockerFixture, langfuse_score: AsyncMock
) -> None:
    """The real ``capture`` runs against a client that raises."""
    client = MagicMock()
    client.capture.side_effect = RuntimeError("posthog is down")
    mocker.patch.object(posthog_client, "get_posthog_client", return_value=client)

    response = await _rate()

    client.capture.assert_called_once()
    assert client.capture.call_args.kwargs["distinct_id"] == USER_ID
    assert response is not None
    assert response.id == FEEDBACK_ID
    langfuse_score.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_reply_sends_nothing(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(
        feedback_db, "get_rateable_message", new=AsyncMock(return_value=None)
    )
    capture = mocker.patch.object(feedback, "capture")

    assert await _rate() is None
    capture.assert_not_called()


@pytest.mark.parametrize(
    ("message_id", "expected"),
    [
        (f"{SESSION_ID}-seq-0", 0),
        (f"{SESSION_ID}-seq-42", 42),
        (f"{SESSION_ID}-seq-999999999", 999_999_999),
        # A reply the UI has not re-keyed from the stream yet, or a row id.
        ("5b0c2a8e-4f1d-4c3b-9a7e-6d5f4e3c2b1a", None),
        # The idx fallback names a render position, not a row.
        (f"{SESSION_ID}-idx-3", None),
        # Another chat's reply never resolves against this one.
        ("11111111-2222-4333-8444-555555555555-seq-3", None),
        (f"{SESSION_ID}-seq-", None),
        (f"{SESSION_ID}-seq-3a", None),
        (f"{SESSION_ID}-seq--3", None),
        (f"{SESSION_ID}-seq-²", None),
        # Past int4: no chat is that long, and the lookup must not overflow.
        (f"{SESSION_ID}-seq-2147483648", None),
    ],
)
def test_sequence_from_ui_message_id(message_id: str, expected: int | None) -> None:
    assert sequence_from_ui_message_id(SESSION_ID, message_id) == expected
