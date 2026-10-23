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

from backend.api.features.onboarding_dump import quality, routes
from backend.api.features.onboarding_dump.models import (
    MAX_DURATION_SECS,
    MAX_PART_BYTES,
    MAX_RECORDING_BYTES,
    SuggestedPrompt,
)
from backend.api.features.store.exceptions import VirusDetectedError
from backend.data.understanding import BusinessUnderstandingInput
from backend.util.gcs_utils_test import _gcs_http_error

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
RECOMMENDED_EXPERTS_URL = "/onboarding/brain-dump/recommended-experts"

RECORDING_ID = "rec-1"
TRANSCRIPT = "I run a small bakery and I want the weekly order emails handled."
JUDGED_UNUSABLE = quality.QualityRejection(
    error_code=quality.INSUFFICIENT_CONTENT, reason="judge_unusable"
)
# Taken before the autouse fixture below replaces it, for the one test
# that drives the real gate.
_REAL_QUALITY_GATE = quality.check_transcript_quality
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

    async def clear_audio_paths(self, audio_paths: list[str]) -> int:
        if self.row is None or self.row.audioPath not in audio_paths:
            return 0
        self.row.audioPath = None
        return 1

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
        self,
        user_id: str,
        recording_id: str,
        error_code: str,
        reason: str | None = None,
    ) -> None:
        await self.update_dump(
            user_id,
            recording_id,
            status=BrainDumpStatus.failed,
            errorCode=error_code,
            errorReason=reason,
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


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def personalization_admission(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        "backend.api.features.onboarding_dump.service.enforce_personalization_budget",
        new=AsyncMock(),
    )


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
    mocker.patch(f"{module}.clear_audio_paths", new=store.clear_audio_paths)
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
        "audio_available": AsyncMock(return_value=True),
        "delete_audio": AsyncMock(),
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


@pytest.fixture(autouse=True)
def quality_gate(mocker: MockerFixture) -> AsyncMock:
    """The quality gate passes by default so route tests stay about the
    routes; rejection tests flip the return value."""
    mock = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.onboarding_dump.quality.check_transcript_quality",
        new=mock,
    )
    return mock


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
    dumps: DumpStore,
    stt_create: AsyncMock,
    extraction: dict[str, AsyncMock],
    storage_mocks: dict[str, AsyncMock],
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
    # Only the transcript is kept: the recording went once it was stored.
    assert dumps.row.audioPath is None
    storage_mocks["delete_audio"].assert_awaited_once_with("brain-dumps/rec-1.webm")
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
    storage_mocks: dict[str, AsyncMock],
):
    storage_mocks["buffered_size"].return_value = MAX_RECORDING_BYTES

    response = upload_part(part_index=1)

    assert response.status_code == 413
    assert "Recording exceeds" in response.json()["detail"]
    storage_mocks["append_part"].assert_not_awaited()


def test_a_racing_part_that_blows_the_total_limit_is_rejected_after_the_write(
    storage_mocks: dict[str, AsyncMock],
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


def test_quality_rejected_voice_finalize_reports_the_error_code(
    dumps: DumpStore, quality_gate: AsyncMock, extraction: dict[str, AsyncMock]
):
    """The API state the frontend branches on: a 200 envelope with
    ``failed`` + a quality error code, distinct from a transcription
    failure — with the transcript still on the row. The recording is not:
    it went once the transcript was stored, and a re-record starts a new
    take from the browser."""
    upload_part()
    quality_gate.return_value = JUDGED_UNUSABLE

    response = finalize()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == BrainDumpStatus.failed
    assert body["error_code"] == "insufficient_content"
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == "insufficient_content"
    assert dumps.row.transcript == TRANSCRIPT
    assert dumps.row.audioPath is None
    extraction["upsert_business_understanding"].assert_not_awaited()


def test_quality_rejected_typed_finalize_reports_the_error_code(
    dumps: DumpStore, quality_gate: AsyncMock, extraction: dict[str, AsyncMock]
):
    quality_gate.return_value = JUDGED_UNUSABLE

    response = finalize(input_mode="typed", text="hello hello testing")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == BrainDumpStatus.failed
    assert body["error_code"] == "insufficient_content"
    extraction["upsert_business_understanding"].assert_not_awaited()


def test_a_judge_outage_reports_its_own_code_and_reason(
    dumps: DumpStore,
    quality_gate: AsyncMock,
    extraction: dict[str, AsyncMock],
    mocker: MockerFixture,
):
    """Through the real gate: a judge that times out is reported as an
    outage the frontend offers a retry for, never as "not enough", and the
    row and the response both say what decided it."""
    quality_gate.side_effect = _REAL_QUALITY_GATE
    judge = MagicMock()
    judge.chat.completions.create = AsyncMock(side_effect=TimeoutError())
    mocker.patch.object(quality, "get_openai_client", return_value=judge)
    upload_part()

    response = finalize()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == BrainDumpStatus.failed
    assert body["error_code"] == "quality_check_unavailable"
    assert body["error_reason"] == "judge_timeout"
    assert body["transcript_words"] == 13
    assert dumps.row is not None
    assert dumps.row.errorCode == "quality_check_unavailable"
    assert dumps.row.errorReason == "judge_timeout"
    extraction["upsert_business_understanding"].assert_not_awaited()


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
        "error_reason": None,
        "transcript_words": None,
    }
    assert dumps.row is not None
    assert dumps.row.inputMode == BrainDumpInputMode.skipped
    assert dumps.row.status == BrainDumpStatus.completed
    assert dumps.row.transcript is None


def test_a_transcribed_take_offers_no_recording(dumps: DumpStore):
    upload_part()
    finalize()

    status = client.get(STATUS_URL)
    assert status.status_code == 200
    assert status.json() == {
        "status": BrainDumpStatus.completed,
        "input_mode": BrainDumpInputMode.voice,
        "error_code": None,
        "has_audio": False,
        # The greeting is stored in the same update that completes the
        # dump, so a completed voice dump is always greeting-ready.
        "greeting_ready": True,
    }

    recording = client.get(RECORDING_URL, follow_redirects=False)
    assert recording.status_code == 404
    assert recording.json()["detail"] == "No recording found"


def test_a_failed_take_keeps_its_recording_downloadable(
    dumps: DumpStore, stt_create: AsyncMock
):
    upload_part()
    stt_create.side_effect = RuntimeError("provider down")
    finalize()

    status = client.get(STATUS_URL)
    assert status.json()["status"] == BrainDumpStatus.failed
    assert status.json()["has_audio"] is True

    recording = client.get(RECORDING_URL, follow_redirects=False)
    assert recording.status_code == 307
    assert recording.headers["location"] == "https://signed.example/audio"


def test_recording_deleted_by_retention_is_404_not_a_dead_link(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], stt_create: AsyncMock
):
    upload_part()
    stt_create.side_effect = RuntimeError("provider down")
    finalize()
    storage_mocks["audio_available"].return_value = False

    recording = client.get(RECORDING_URL, follow_redirects=False)

    assert recording.status_code == 404
    assert recording.json()["detail"] == "Recording no longer available"
    storage_mocks["audio_download_url"].assert_not_awaited()
    # The status stops offering a download that can only 404.
    assert dumps.row is not None and dumps.row.audioPath is None


def test_recording_check_failing_at_the_bucket_is_503_without_the_path(
    dumps: DumpStore,
    storage_mocks: dict[str, AsyncMock],
    stt_create: AsyncMock,
    mocker: MockerFixture,
    test_user_id: str,
):
    """A GCS outage or auth error must not become a 500 whose traceback
    carries the object URL, and with it the user ID."""
    upload_part()
    stt_create.side_effect = RuntimeError("provider down")
    finalize()
    storage_mocks["audio_available"].side_effect = _gcs_http_error(
        503, f"https://storage.googleapis.com/storage/v1/b/b/o/{test_user_id}%2Fa.webm"
    )
    warning = mocker.patch.object(routes.logger, "warning")

    recording = client.get(RECORDING_URL, follow_redirects=False)

    assert recording.status_code == 503
    assert recording.json()["detail"] == (
        "Recording temporarily unavailable, try again"
    )
    storage_mocks["audio_download_url"].assert_not_awaited()
    # Still the user's recording: the row keeps pointing at it.
    assert dumps.row is not None and dumps.row.audioPath is not None
    logged = " ".join(str(c.args) for c in warning.call_args_list)
    assert "ClientResponseError" in logged
    assert test_user_id not in logged and "storage.googleapis.com" not in logged


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


def test_recommended_experts_returns_the_stored_team(
    dumps: DumpStore, mocker: MockerFixture
):
    upload_part()
    finalize()
    assert dumps.row is not None
    dumps.row.recommendedExperts = {
        "diagnosis": "You have a marketing problem.",
        "experts": [
            {
                "template_id": "tpl-maria",
                "name": "Maria",
                "role": "Marketing",
                "avatar_url": "/experts/maria.svg",
                "reason": "You post on LinkedIn weekly.",
                "workflow_names": ["LinkedIn Post Generator"],
            }
        ],
        "raise_suggestion": {"role": "support", "reason": "Tickets pile up."},
        "source": "llm",
    }

    response = client.get(RECOMMENDED_EXPERTS_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["team"]["source"] == "llm"
    assert [e["name"] for e in body["team"]["experts"]] == ["Maria"]
    assert body["team"]["raise_suggestion"]["role"] == "support"


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
