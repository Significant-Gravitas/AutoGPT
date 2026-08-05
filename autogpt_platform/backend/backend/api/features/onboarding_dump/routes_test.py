"""Endpoint tests for the onboarding brain-dump routes.

Everything below the router (Prisma, Redis, cloud storage, OpenAI) is
mocked at the point of use, so these run without a live stack.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import routes
from backend.api.features.onboarding_dump.models import (
    MAX_DURATION_SECS,
    MAX_PART_BYTES,
    MAX_RECORDING_BYTES,
    SuggestedPrompt,
)
from backend.api.features.store.exceptions import VirusDetectedError
from backend.data.understanding import BusinessUnderstandingInput

app = fastapi.FastAPI()
app.include_router(routes.router)
client = fastapi.testclient.TestClient(app)

PARTS_URL = "/onboarding/brain-dump/parts"
FINALIZE_URL = "/onboarding/brain-dump/finalize"
STATUS_URL = "/onboarding/brain-dump/status"
RECORDING_URL = "/onboarding/brain-dump/recording"
DISCARD_URL = "/onboarding/brain-dump/"
INTRO_URL = "/onboarding/brain-dump/intro"
INTRO_COMPLETE_URL = "/onboarding/brain-dump/intro/complete"
RECOMMENDED_URL = "/onboarding/brain-dump/recommended-providers"

RECORDING_ID = "rec-1"
TRANSCRIPT = "I run a small bakery and I want the weekly order emails handled."
GREETING = "Let's get those weekly order emails off your plate."


class DumpStore:
    """In-memory stand-in for the one-row-per-user brain dump table."""

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None
        self.statuses: list[BrainDumpStatus] = []

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def start_dump(
        self, user_id: str, recording_id: str, input_mode: BrainDumpInputMode
    ) -> OnboardingBrainDump:
        # Mirrors the real `start_dump`: a take already moving through
        # the pipeline is returned untouched, so a replayed part 0 or a
        # repeated finalize cannot reset it.
        if (
            self.row is not None
            and self.row.status
            in (
                BrainDumpStatus.recording_uploaded,
                BrainDumpStatus.transcribing,
                BrainDumpStatus.transcribed,
                BrainDumpStatus.extracting,
                BrainDumpStatus.completed,
            )
            and (
                self.row.recordingId == recording_id
                or input_mode != BrainDumpInputMode.voice
            )
        ):
            return self.row
        self.row = OnboardingBrainDump.model_construct(
            userId=user_id,
            recordingId=recording_id,
            status=BrainDumpStatus.recording_uploaded,
            inputMode=input_mode,
            transcript=None,
            greeting=None,
            suggestedPrompts=[],
            greetingSeen=False,
            audioPath=None,
            errorCode=None,
            recommendedProviders=None,
        )
        self.statuses.append(BrainDumpStatus.recording_uploaded)
        return self.row

    async def update_dump(self, user_id: str, recording_id: str, **fields: Any) -> bool:
        # Mirrors the scoped UPDATE: a write from a take the row has
        # already moved past hits nothing.
        if self.row is None or self.row.recordingId != recording_id:
            return False
        for name, value in fields.items():
            setattr(self.row, name, value)
        status = fields.get("status")
        if status is not None:
            self.statuses.append(status)
        return True

    async def claim_transition(
        self,
        user_id: str,
        recording_id: str,
        *,
        expected: BrainDumpStatus,
        new: BrainDumpStatus,
        **fields: Any,
    ) -> bool:
        """Mirrors the conditional UPDATE: only matching rows transition."""
        if (
            self.row is None
            or self.row.recordingId != recording_id
            or self.row.status != expected
        ):
            return False
        await self.update_dump(user_id, recording_id, status=new, **fields)
        return True

    async def mark_failed(
        self, user_id: str, recording_id: str, error_code: str
    ) -> None:
        await self.update_dump(
            user_id,
            recording_id,
            status=BrainDumpStatus.failed,
            errorCode=error_code,
        )

    async def mark_greeting_seen(self, user_id: str) -> None:
        if self.row is None:
            self.row = OnboardingBrainDump.model_construct(
                userId=user_id,
                recordingId="greeting-only",
                status=BrainDumpStatus.completed,
                inputMode=BrainDumpInputMode.skipped,
                transcript=None,
                greeting=None,
                suggestedPrompts=[],
                greetingSeen=True,
                audioPath=None,
                errorCode=None,
            )
        else:
            self.row.greetingSeen = True


# ``require_brain_dump_flag`` runs the real ``is_feature_enabled``, which
# consults this env override before LaunchDarkly — so the gate is driven
# here exactly the way it is driven locally, with no mock in the path.
FLAG_ENV_VAR = "FORCE_FLAG_ONBOARDING_BRAIN_DUMP"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG_ENV_VAR, "true")


@pytest.fixture(autouse=True)
def dumps(mocker: MockerFixture) -> DumpStore:
    store = DumpStore()
    module = "backend.api.features.onboarding_dump.db"
    mocker.patch(f"{module}.get_dump", new=store.get_dump)
    mocker.patch(f"{module}.start_dump", new=store.start_dump)
    mocker.patch(f"{module}.update_dump", new=store.update_dump)
    mocker.patch(f"{module}.mark_failed", new=store.mark_failed)
    mocker.patch(f"{module}.claim_transition", new=store.claim_transition)
    mocker.patch(f"{module}.mark_greeting_seen", new=store.mark_greeting_seen)
    return store


@pytest.fixture(autouse=True)
def storage_mocks(mocker: MockerFixture) -> dict[str, AsyncMock]:
    module = "backend.api.features.onboarding_dump.storage"
    mocks = {
        "buffered_size": AsyncMock(return_value=0),
        "append_part": AsyncMock(return_value=1024),
        "assemble_parts": AsyncMock(return_value=b"opus-audio-bytes"),
        "store_audio": AsyncMock(return_value="brain-dumps/rec-1.webm"),
        "discard_parts": AsyncMock(),
        "audio_download_url": AsyncMock(return_value="https://signed.example/audio"),
    }
    for name, mock in mocks.items():
        mocker.patch(f"{module}.{name}", new=mock)
    return mocks


@pytest.fixture(autouse=True)
def stt_create(mocker: MockerFixture) -> AsyncMock:
    create = AsyncMock(return_value=MagicMock(text=TRANSCRIPT, language="en"))
    stt = MagicMock()
    stt.audio.transcriptions.create = create
    mocker.patch(
        "backend.api.features.onboarding_dump.transcription.get_stt_client",
        return_value=stt,
    )
    return create


@pytest.fixture(autouse=True)
def extraction(mocker: MockerFixture) -> dict[str, AsyncMock]:
    module = "backend.api.features.onboarding_dump.service"
    mocks = {
        "scan_content_safe": AsyncMock(),
        "get_business_understanding": AsyncMock(return_value=None),
        "extract_business_understanding": AsyncMock(
            return_value=BusinessUnderstandingInput.model_construct()
        ),
        "upsert_business_understanding": AsyncMock(),
    }
    for name, mock in mocks.items():
        mocker.patch(f"{module}.{name}", new=mock)
    return mocks


@pytest.fixture(autouse=True)
def generation(mocker: MockerFixture) -> dict[str, AsyncMock]:
    """Stub the two LLM jobs the background half of finalize kicks off.

    Both degrade instead of raising, so without this the tests still pass
    — while quietly building a real client and reaching for the network
    on every finalize. Deterministic stubs keep the suite offline and
    give the intro assertions something fixed to match.
    """
    intro_mock = AsyncMock(
        return_value=(
            GREETING,
            [
                SuggestedPrompt(title=f"Prompt {index}", prompt="Do the thing")
                for index in range(5)
            ],
        )
    )
    mocker.patch(
        "backend.api.features.onboarding_dump.intro.generate_intro", new=intro_mock
    )
    recommend_mock = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.onboarding_dump.recommend.generate_recommendations",
        new=recommend_mock,
    )
    return {"generate_intro": intro_mock, "generate_recommendations": recommend_mock}


def upload_part(
    part_index: int = 0,
    content: bytes = b"chunk",
    content_type: str = "audio/webm",
):
    return client.post(
        PARTS_URL,
        files={"file": (f"part-{part_index}.webm", content, content_type)},
        data={"recording_id": RECORDING_ID, "part_index": str(part_index)},
    )


def finalize(**overrides: Any):
    payload: dict[str, Any] = {"recording_id": RECORDING_ID}
    payload.update(overrides)
    return client.post(FINALIZE_URL, json=payload)


def test_parts_upload_then_finalize_completes_with_full_transcript(
    dumps: DumpStore, stt_create: AsyncMock, extraction: dict[str, AsyncMock]
):
    part = upload_part()
    assert part.status_code == 200
    assert part.json() == {
        "recording_id": RECORDING_ID,
        "part_index": 0,
        "received_bytes": len(b"chunk"),
        "total_bytes": 1024,
    }

    response = finalize()

    assert response.status_code == 200
    body = response.json()
    # Finalize answers as soon as the transcript is stored; extraction and
    # the greeting run as a background task (which the TestClient executes
    # before returning), so the row below still ends up completed.
    assert body["status"] == BrainDumpStatus.transcribed
    assert body["input_mode"] == BrainDumpInputMode.voice
    assert body["transcript_preview"] == TRANSCRIPT
    assert dumps.statuses == [
        BrainDumpStatus.recording_uploaded,
        # Written twice: once by the atomic claim that decides which
        # concurrent finalize proceeds, then again with the audio
        # metadata once the recording is stored.
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribed,
        BrainDumpStatus.extracting,
        BrainDumpStatus.completed,
    ]
    assert dumps.row is not None
    assert dumps.row.transcript == TRANSCRIPT
    assert dumps.row.audioPath == "brain-dumps/rec-1.webm"
    stt_create.assert_awaited_once()
    extraction["upsert_business_understanding"].assert_awaited_once()


def test_finalize_retry_is_idempotent_and_does_not_retranscribe(
    dumps: DumpStore, stt_create: AsyncMock
):
    upload_part()
    assert finalize().status_code == 200
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.completed

    retry = finalize()

    assert retry.status_code == 200
    assert retry.json()["status"] == BrainDumpStatus.completed
    assert retry.json()["transcript_preview"] == TRANSCRIPT
    stt_create.assert_awaited_once()


def test_virus_detected_rejects_and_marks_the_dump_failed(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    upload_part()
    extraction["scan_content_safe"].side_effect = VirusDetectedError("Eicar-Test")

    response = finalize()

    assert response.status_code == 400
    assert "Eicar-Test" in response.json()["detail"]
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == "virus_detected"


def test_part_over_the_per_part_limit_is_rejected():
    response = upload_part(content=b"a" * (MAX_PART_BYTES + 1))

    assert response.status_code == 413
    assert "part exceeds" in response.json()["detail"]


def test_part_pushing_the_recording_over_the_total_limit_is_rejected(
    storage_mocks: dict[str, AsyncMock]
):
    storage_mocks["buffered_size"].return_value = MAX_RECORDING_BYTES

    response = upload_part(part_index=1)

    assert response.status_code == 413
    assert "Recording exceeds" in response.json()["detail"]
    storage_mocks["append_part"].assert_not_awaited()


def test_a_racing_part_that_blows_the_total_limit_is_rejected_after_the_write(
    storage_mocks: dict[str, AsyncMock]
):
    """The pre-check races; the post-check is the authoritative one.

    Two parts in flight can both read a buffer that is under the cap and
    both be admitted. `append_part` reports the size from inside the same
    transaction that wrote the part, so that figure cannot race.
    """
    storage_mocks["buffered_size"].return_value = 0
    storage_mocks["append_part"].return_value = MAX_RECORDING_BYTES + 1

    response = upload_part(part_index=1)

    assert response.status_code == 413
    assert "Recording exceeds" in response.json()["detail"]
    storage_mocks["discard_parts"].assert_awaited_once()


def test_discard_targets_the_recording_the_caller_names(
    storage_mocks: dict[str, AsyncMock], dumps: DumpStore
):
    """A second tab must not be able to clear the wrong buffer.

    Without an explicit id the server drops whatever the row points at,
    which after another tab has claimed it is a take still being filled.
    """
    client.post(
        FINALIZE_URL, json={"recording_id": "rec-other", "input_mode": "skipped"}
    )

    response = client.delete(DISCARD_URL.rstrip("/"), params={"recording_id": "rec-1"})

    assert response.status_code == 200
    # The take the caller named, not the one the row happens to hold.
    assert storage_mocks["discard_parts"].await_args.args[1] == "rec-1"


def test_unsupported_mime_type_is_rejected():
    response = upload_part(content_type="video/mp4")

    assert response.status_code == 415
    assert "Unsupported audio type" in response.json()["detail"]


def test_typed_finalize_skips_audio_and_extracts_directly(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], stt_create: AsyncMock
):
    response = finalize(input_mode="typed", text="  I run a bakery.  ")

    assert response.status_code == 200
    assert response.json()["status"] == BrainDumpStatus.transcribed
    assert response.json()["input_mode"] == BrainDumpInputMode.typed
    assert dumps.row is not None
    assert dumps.row.transcript == "I run a bakery."
    assert dumps.row.audioPath is None
    storage_mocks["assemble_parts"].assert_not_awaited()
    stt_create.assert_not_awaited()


def test_a_duration_past_the_ceiling_is_rejected(stt_create: AsyncMock):
    """``duration_secs`` is not just metadata — it bounds the split loop.

    An unmuxed MediaRecorder stream carries no container duration, so the
    client's number is what ``split_audio`` iterates over: one ffmpeg run
    and one billed transcription per 10 minutes of whatever is claimed.
    """
    upload_part()

    response = finalize(duration_secs=MAX_DURATION_SECS + 1)

    assert response.status_code == 422
    stt_create.assert_not_awaited()


def test_typed_finalize_with_empty_text_is_rejected():
    response = finalize(input_mode="typed", text="   ")

    assert response.status_code == 422
    assert response.json()["detail"] == "Typed brain dump cannot be empty"


def test_skipped_finalize_records_the_skip_without_a_transcript(dumps: DumpStore):
    response = finalize(input_mode="skipped")

    assert response.status_code == 200
    assert response.json() == {
        "status": BrainDumpStatus.completed,
        "input_mode": BrainDumpInputMode.skipped,
        "transcript_preview": None,
        "error_code": None,
    }
    assert dumps.row is not None
    assert dumps.row.inputMode == BrainDumpInputMode.skipped
    assert dumps.row.status == BrainDumpStatus.completed
    assert dumps.row.transcript is None


def test_status_and_recording_reflect_the_stored_dump(dumps: DumpStore):
    upload_part()
    finalize()

    status = client.get(STATUS_URL)
    assert status.status_code == 200
    assert status.json() == {
        "status": BrainDumpStatus.completed,
        "input_mode": BrainDumpInputMode.voice,
        "error_code": None,
        "has_audio": True,
        # The greeting is stored in the same update that completes the
        # dump, so a completed voice dump is always greeting-ready.
        "greeting_ready": True,
    }

    recording = client.get(RECORDING_URL, follow_redirects=False)
    assert recording.status_code == 307
    assert recording.headers["location"] == "https://signed.example/audio"


def test_discard_clears_the_part_buffer_of_an_unfinished_take(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock]
):
    upload_part()

    response = client.delete(DISCARD_URL)

    assert response.status_code == 200
    assert response.json()["status"] == BrainDumpStatus.recording_uploaded
    storage_mocks["discard_parts"].assert_awaited_once()


def test_greeting_complete_hides_the_intro_for_good(dumps: DumpStore):
    upload_part()
    finalize(mime_type="audio/webm")

    complete = client.post(INTRO_COMPLETE_URL)
    assert complete.status_code == 200
    assert complete.json()["greeting_done"] is True

    intro = client.get(INTRO_URL)
    assert intro.status_code == 200
    body = intro.json()
    assert body["greeting_done"] is True
    assert body["greeting"] == ""
    assert body["prompts"] == []


def test_greeting_complete_sticks_without_a_dump_row(dumps: DumpStore):
    # A user who never recorded still gets a durable "seen" flag.
    complete = client.post(INTRO_COMPLETE_URL)
    assert complete.status_code == 200

    intro = client.get(INTRO_URL)
    assert intro.json()["greeting_done"] is True


def test_intro_returns_greeting_and_prompts_before_completion(dumps: DumpStore):
    upload_part()
    finalize(mime_type="audio/webm")

    response = client.get(INTRO_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["greeting_done"] is False
    assert body["greeting"]
    assert len(body["prompts"]) >= 5
    assert all(p["title"] and p["prompt"] for p in body["prompts"])


def test_recommended_providers_are_pending_until_the_job_writes(dumps: DumpStore):
    upload_part()
    finalize()
    assert dumps.row is not None
    # A null column is the job still running, not "nothing to recommend".
    dumps.row.recommendedProviders = None

    response = client.get(RECOMMENDED_URL)

    assert response.status_code == 200
    assert response.json() == {"ready": False, "providers": []}


def test_recommended_providers_returns_the_stored_picks(dumps: DumpStore):
    upload_part()
    finalize()
    assert dumps.row is not None
    dumps.row.recommendedProviders = [
        {"provider": "github", "reason": "You mentioned chasing issues."}
    ]

    response = client.get(RECOMMENDED_URL)

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "providers": [
            {"provider": "github", "reason": "You mentioned chasing issues."}
        ],
    }


@pytest.mark.parametrize(
    "call",
    [
        lambda: upload_part(),
        lambda: finalize(),
        lambda: client.get(STATUS_URL),
        lambda: client.get(RECORDING_URL, follow_redirects=False),
        lambda: client.delete(DISCARD_URL),
        lambda: client.get(INTRO_URL),
        lambda: client.post(INTRO_COMPLETE_URL),
        lambda: client.get(RECOMMENDED_URL),
    ],
    ids=[
        "parts",
        "finalize",
        "status",
        "recording",
        "discard",
        "intro",
        "complete",
        "recommended-providers",
    ],
)
def test_every_endpoint_is_404_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, call
):
    monkeypatch.setenv(FLAG_ENV_VAR, "false")

    response = call()

    assert response.status_code == 404
    assert response.json()["detail"] == "Feature not available"
