"""Route tests for ``/api/feedback-prompts``.

The data layer and PostHog are mocked at the boundary;
``backend/data/feedback_prompts_test.py`` proves the limits against Postgres.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.models import FeedbackPrompt

from backend.api.features.feedback_prompts import analytics, routes
from backend.api.features.feedback_prompts.model import MAX_NOTE_CHARS
from backend.util.posthog_events import PostHogEvent

app = fastapi.FastAPI()
app.include_router(routes.router)
client = fastapi.testclient.TestClient(app)

PROMPT_ID = "c3a1b2d4-e5f6-4789-8abc-def012345678"
NOW = datetime(2026, 12, 1, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def posthog_capture(mocker: pytest_mock.MockerFixture) -> MagicMock:
    return mocker.patch.object(analytics, "capture")


@pytest.fixture
def impersonating(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(routes, "is_impersonated_request", return_value=True)


def _row(test_user_id: str, kind: str = "trial_cancel", **fields) -> FeedbackPrompt:
    return FeedbackPrompt(
        id=PROMPT_ID, userId=test_user_id, kind=kind, shownAt=NOW, **fields
    )


def _mock_get(
    mocker: pytest_mock.MockerFixture, row: FeedbackPrompt | None
) -> AsyncMock:
    return mocker.patch.object(
        routes, "get_feedback_prompt", new=AsyncMock(return_value=row)
    )


def _mock_update(
    mocker: pytest_mock.MockerFixture, row: FeedbackPrompt | None
) -> AsyncMock:
    return mocker.patch.object(
        routes, "update_feedback_prompt", new=AsyncMock(return_value=row)
    )


def _mock_claim(mocker: pytest_mock.MockerFixture, prompt_id: str | None) -> AsyncMock:
    return mocker.patch.object(
        routes, "claim_feedback_prompt", new=AsyncMock(return_value=prompt_id)
    )


def test_claim_returns_id_and_sends_shown(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock, test_user_id: str
) -> None:
    claim = _mock_claim(mocker, PROMPT_ID)

    response = client.post("/claim", json={"kind": "top_up"})

    assert response.status_code == 200
    assert response.json() == {"id": PROMPT_ID}
    claim.assert_awaited_once_with(user_id=test_user_id, kind="top_up")
    posthog_capture.assert_called_once_with(
        test_user_id,
        PostHogEvent.FEEDBACK_PROMPT_SHOWN,
        {"prompt_id": PROMPT_ID, "kind": "top_up"},
    )


def test_refused_claim_sends_nothing(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock
) -> None:
    _mock_claim(mocker, None)

    response = client.post("/claim", json={"kind": "nps"})

    assert response.status_code == 200
    assert response.json() == {"id": None}
    posthog_capture.assert_not_called()


def test_claim_rejects_an_unknown_kind(mocker: pytest_mock.MockerFixture) -> None:
    claim = _mock_claim(mocker, PROMPT_ID)

    assert client.post("/claim", json={"kind": "survey"}).status_code == 422
    claim.assert_not_awaited()


def test_patch_sends_the_answer_as_saved(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock, test_user_id: str
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(
        mocker,
        _row(
            test_user_id,
            reason="too_expensive",
            note="Too dear",
            score=None,
            offerTaken=True,
            answeredAt=NOW,
        ),
    )

    response = client.patch(
        f"/{PROMPT_ID}", json={"note": "Too dear", "offer_taken": True}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": PROMPT_ID,
        "kind": "trial_cancel",
        "answered": True,
        "dismissed": False,
        "reason": "too_expensive",
        "score": None,
    }
    update.assert_awaited_once_with(
        user_id=test_user_id,
        prompt_id=PROMPT_ID,
        reason=None,
        note="Too dear",
        score=None,
        dismissed=False,
        offer_taken=True,
    )
    posthog_capture.assert_called_once_with(
        test_user_id,
        PostHogEvent.FEEDBACK_PROMPT_ANSWERED,
        {
            "prompt_id": PROMPT_ID,
            "kind": "trial_cancel",
            "reason": "too_expensive",
            "note": "Too dear",
            "score": None,
            "offer_taken": True,
        },
    )


@pytest.mark.parametrize("reason", ["no_mic", "", "Too expensive"])
def test_reason_outside_the_kinds_list_is_rejected(
    mocker: pytest_mock.MockerFixture, test_user_id: str, reason: str
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(mocker, None)

    assert client.patch(f"/{PROMPT_ID}", json={"reason": reason}).status_code == 422
    update.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [
        {"note": "x" * (MAX_NOTE_CHARS + 1)},
        {"score": 11},
        {"score": -1},
        {"reason": "x" * 65},
    ],
)
def test_out_of_bounds_values_are_rejected(
    mocker: pytest_mock.MockerFixture, test_user_id: str, body: dict
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(mocker, None)

    assert client.patch(f"/{PROMPT_ID}", json=body).status_code == 422
    update.assert_not_awaited()


def test_blank_note_is_saved_as_nothing(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock, test_user_id: str
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(mocker, _row(test_user_id))

    assert client.patch(f"/{PROMPT_ID}", json={"note": "   "}).status_code == 200
    assert update.await_args is not None
    assert update.await_args.kwargs["note"] is None
    posthog_capture.assert_not_called()


@pytest.mark.parametrize(
    ("method", "body"),
    [("patch", {"reason": "other"}), ("get", None)],
)
def test_another_users_id_is_not_found(
    mocker: pytest_mock.MockerFixture,
    posthog_capture: MagicMock,
    method: str,
    body: dict | None,
) -> None:
    _mock_get(mocker, None)
    update = _mock_update(mocker, None)

    response = client.request(method, f"/{PROMPT_ID}", json=body)

    assert response.status_code == 404
    update.assert_not_awaited()
    posthog_capture.assert_not_called()


def test_dismissing_sends_no_answer(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock, test_user_id: str
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(mocker, _row(test_user_id, dismissedAt=NOW))

    response = client.patch(f"/{PROMPT_ID}", json={"dismissed": True})

    assert response.status_code == 200
    assert response.json()["dismissed"] is True
    assert response.json()["answered"] is False
    update.assert_awaited_once()
    posthog_capture.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "reason"),
    [("work_check", "too_slow"), ("otto_mistake", "made_up")],
)
def test_chat_card_answers_reach_posthog_only_as_ratings(
    mocker: pytest_mock.MockerFixture,
    posthog_capture: MagicMock,
    test_user_id: str,
    kind: str,
    reason: str,
) -> None:
    _mock_get(mocker, _row(test_user_id, kind=kind))
    update = _mock_update(
        mocker, _row(test_user_id, kind=kind, reason=reason, answeredAt=NOW)
    )

    response = client.patch(f"/{PROMPT_ID}", json={"reason": reason})

    assert response.status_code == 200
    assert response.json()["answered"] is True
    update.assert_awaited_once()
    posthog_capture.assert_not_called()


def test_get_returns_the_cards_state(
    mocker: pytest_mock.MockerFixture, test_user_id: str
) -> None:
    get = _mock_get(
        mocker,
        _row(test_user_id, kind="work_check", reason="too_slow", answeredAt=NOW),
    )

    response = client.get(f"/{PROMPT_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "id": PROMPT_ID,
        "kind": "work_check",
        "answered": True,
        "dismissed": False,
        "reason": "too_slow",
        "score": None,
    }
    get.assert_awaited_once_with(user_id=test_user_id, prompt_id=PROMPT_ID)


@pytest.mark.usefixtures("impersonating")
def test_impersonated_claim_creates_nothing(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock
) -> None:
    claim = _mock_claim(mocker, PROMPT_ID)

    response = client.post("/claim", json={"kind": "top_up"})

    assert response.status_code == 200
    assert response.json() == {"id": None}
    claim.assert_not_awaited()
    posthog_capture.assert_not_called()


@pytest.mark.usefixtures("impersonating")
def test_impersonated_patch_writes_and_sends_nothing(
    mocker: pytest_mock.MockerFixture, posthog_capture: MagicMock, test_user_id: str
) -> None:
    _mock_get(mocker, _row(test_user_id))
    update = _mock_update(mocker, _row(test_user_id, reason="other"))

    response = client.patch(f"/{PROMPT_ID}", json={"reason": "other"})

    assert response.status_code == 200
    assert response.json()["reason"] is None
    update.assert_not_awaited()
    posthog_capture.assert_not_called()
