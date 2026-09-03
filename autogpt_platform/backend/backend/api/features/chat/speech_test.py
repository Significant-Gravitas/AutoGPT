"""Tests for the voice-mode speech route, and the metering it must perform."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import speech as speech_routes
from backend.copilot import speech as speech_module

app = fastapi.FastAPI()
app.include_router(speech_routes.router)
client = fastapi.testclient.TestClient(app)

AUDIO = b"ID3-fake-mp3-bytes"


def test_speech_meters_the_synthesis_against_the_users_plan(
    record_usage: AsyncMock, test_user_id: str
) -> None:
    response = client.post(
        "/speech", json={"text": "Two sentences here.", "session_id": "sess-1"}
    )

    assert response.status_code == 200
    assert response.content == AUDIO
    assert response.headers["content-type"] == speech_module.AUDIO_MEDIA_TYPE

    kwargs = record_usage.await_args.kwargs
    assert kwargs["user_id"] == test_user_id
    assert kwargs["provider"] == "openai"
    assert kwargs["block_name_override"] == speech_module.TTS_BLOCK_NAME
    assert kwargs["graph_exec_id_override"] == "sess-1"
    assert kwargs["cost_usd"] == speech_module.speech_cost_usd(
        len("Two sentences here.")
    )
    assert kwargs["extra_metadata"]["characters"] == len("Two sentences here.")
    assert kwargs["extra_metadata"]["surface"] == "voice_mode"


def test_speech_charges_per_character(record_usage: AsyncMock) -> None:
    client.post("/speech", json={"text": "a"})
    short = record_usage.await_args.kwargs["cost_usd"]

    client.post("/speech", json={"text": "a" * 100})
    long = record_usage.await_args.kwargs["cost_usd"]

    assert long == pytest.approx(short * 100)


def test_speech_404s_when_the_flag_is_off(
    record_usage: AsyncMock, mocker: pytest_mock.MockerFixture
) -> None:
    mocker.patch.object(
        speech_routes, "is_feature_enabled", new=AsyncMock(return_value=False)
    )

    response = client.post("/speech", json={"text": "hello"})

    assert response.status_code == 404
    record_usage.assert_not_awaited()


def test_speech_rejects_an_unknown_voice(record_usage: AsyncMock) -> None:
    response = client.post("/speech", json={"text": "hello", "voice": "morgan"})

    assert response.status_code == 400
    record_usage.assert_not_awaited()


def test_speech_rejects_blank_text(record_usage: AsyncMock) -> None:
    response = client.post("/speech", json={"text": "   "})

    assert response.status_code == 400
    record_usage.assert_not_awaited()


def test_speech_503s_without_a_configured_key(
    record_usage: AsyncMock, mocker: pytest_mock.MockerFixture
) -> None:
    mocker.patch.object(
        speech_module,
        "_speech_client",
        side_effect=speech_module.SpeechUnavailable("no key"),
    )

    response = client.post("/speech", json={"text": "hello"})

    assert response.status_code == 503
    record_usage.assert_not_awaited()


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, mocker: pytest_mock.MockerFixture):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    mocker.patch.object(
        speech_routes, "is_feature_enabled", new=AsyncMock(return_value=True)
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def record_usage(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    """Stub OpenAI and capture what the route hands the copilot cost path."""
    audio_response = MagicMock()
    audio_response.aread = AsyncMock(return_value=AUDIO)
    openai_client = MagicMock()
    openai_client.audio.speech.create = AsyncMock(return_value=audio_response)
    mocker.patch.object(speech_module, "_speech_client", return_value=openai_client)

    recorder = AsyncMock(return_value=0)
    mocker.patch.object(speech_module, "persist_and_record_usage", new=recorder)
    return recorder
