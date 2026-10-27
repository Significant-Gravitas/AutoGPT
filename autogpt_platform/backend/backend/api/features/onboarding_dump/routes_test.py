"""Endpoint tests for the onboarding brain-dump routes.

Everything below the router (Prisma, Redis, cloud storage, OpenAI) is
mocked at the point of use, so these run without a live stack.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert, HireResult
from backend.api.features.onboarding_dump import (
    intro,
    plan,
    plan_service,
    quality,
    recommend_experts,
    routes,
    service,
    suggested_tasks,
)
from backend.api.features.onboarding_dump.models import (
    MAX_DURATION_SECS,
    MAX_PART_BYTES,
    MAX_RECORDING_BYTES,
    ExpertRecommendations,
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.api.features.onboarding_dump.role_task_sets import ROLE_FALLBACK_TASKS
from backend.api.features.store.exceptions import VirusDetectedError
from backend.blocks.typesafe._client import JevCallResult
from backend.data.understanding import BusinessUnderstandingInput
from backend.util.exceptions import ExpertLimitExceededError
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
PLAN_URL = "/onboarding/brain-dump/plan"
PLAN_HIRE_URL = "/onboarding/brain-dump/plan/hire"
SUGGESTED_TASKS_URL = "/onboarding/brain-dump/suggested-tasks"

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
        self.clock = datetime(2026, 10, 1, tzinfo=UTC)

    def touch(self) -> None:
        # Prisma moves ``updatedAt`` on every write; the conditional writes
        # below compare against it.
        self.clock += timedelta(seconds=1)
        if self.row is not None:
            self.row.updatedAt = self.clock

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def ensure_dump(self, user_id: str) -> OnboardingBrainDump:
        if self.row is None:
            self.row = OnboardingBrainDump.model_construct(
                userId=user_id,
                recordingId="greeting-only",
                status=BrainDumpStatus.completed,
                inputMode=BrainDumpInputMode.skipped,
                transcript=None,
                greeting=None,
                suggestedPrompts=[],
                greetingSeen=False,
                audioPath=None,
                errorCode=None,
                recommendedProviders=None,
                recommendedExperts=None,
            )
            self.touch()
        return self.row

    async def replace_suggested_prompts(
        self, user_id: str, read_at: datetime, prompts: list[Any]
    ) -> bool:
        if self.row is None or self.row.updatedAt != read_at:
            return False
        self.row.suggestedPrompts = prompts
        self.touch()
        return True

    async def replace_plan(
        self,
        user_id: str,
        recording_id: str,
        read_at: datetime,
        prompts: list[Any],
        team: dict[str, Any],
    ) -> bool:
        if (
            self.row is None
            or self.row.recordingId != recording_id
            or self.row.updatedAt != read_at
        ):
            return False
        self.row.suggestedPrompts = prompts
        self.row.recommendedExperts = team
        self.touch()
        return True

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
            recommendedExperts=None,
        )
        self.touch()
        self.statuses.append(BrainDumpStatus.recording_uploaded)
        return self.row

    async def update_dump(self, user_id: str, recording_id: str, **fields: Any) -> bool:
        # Mirrors the scoped UPDATE: a write from a take the row has
        # already moved past hits nothing.
        if self.row is None or self.row.recordingId != recording_id:
            return False
        for name, value in fields.items():
            setattr(self.row, name, value.data if isinstance(value, Json) else value)
        self.touch()
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
    mocker.patch(f"{module}.ensure_dump", new=store.ensure_dump)
    mocker.patch(
        f"{module}.replace_suggested_prompts", new=store.replace_suggested_prompts
    )
    mocker.patch(f"{module}.replace_plan", new=store.replace_plan)
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
        "mark_discarded": AsyncMock(),
        "was_discarded": AsyncMock(return_value=False),
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


@pytest.fixture(autouse=True)
def reads(mocker: MockerFixture) -> dict[str, AsyncMock]:
    """The reads below the dump row, and the team job's model call, in memory.

    By default a user with no wizard answers, an empty roster and nobody
    hired, whose team job answers as it does when the model reads nothing.
    Left live, the plan and Home routes query the database on the
    TestClient's own event loop (see ``has_session``) and the team job
    calls a model on every finalize.

    Each is patched where it is used. The template read has three use
    sites (the team job, the plan, and Home's list) sharing one mock, so a
    test that fails it fails it everywhere. ``service.wizard_answers`` is
    left real: it is the degrading wrapper around ``read_wizard_answers``.
    The team job looks ``generate_expert_recommendations`` up on its
    module when it runs, so that is where it is replaced.
    """
    templates = AsyncMock(return_value=[])
    mocks = {
        "read_wizard_answers": AsyncMock(return_value=(None, [])),
        "list_templates": templates,
        "list_experts": AsyncMock(return_value=[]),
        "generate_expert_recommendations": AsyncMock(
            return_value=recommend_experts.fallback_expert_recommendations(None, [], [])
        ),
    }
    mocker.patch.object(
        service, "read_wizard_answers", new=mocks["read_wizard_answers"]
    )
    for module in (service, plan_service, suggested_tasks):
        mocker.patch.object(module, "list_templates", new=templates)
    mocker.patch.object(experts_db, "list_experts", new=mocks["list_experts"])
    mocker.patch.object(
        recommend_experts,
        "generate_expert_recommendations",
        new=mocks["generate_expert_recommendations"],
    )
    return mocks


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


def _roster_template(template_id: str, name: str, role: str) -> Expert:
    return Expert.model_construct(
        id=template_id,
        name=name,
        avatar_url=None,
        role=role,
        tagline=f"{name} handles {role.lower()}.",
        workflows=[],
    )


def _hire(template_id: str, name: str) -> Expert:
    return Expert.model_construct(
        id=f"expert-{name.lower()}", name=name, source_template_id=template_id
    )


ROSTER = [
    _roster_template("tpl-max", "Max", "Sales"),
    _roster_template("tpl-maria", "Maria", "Marketing"),
    _roster_template("tpl-mina", "Mina", "Bookkeeping"),
]
FOUNDER_TITLES = [task.title for task in ROLE_FALLBACK_TASKS["Founder/CEO"]]
HEARD_GREETING = (
    "You run a small bakery. Weekly order emails eat your mornings. "
    "Pick a task to start with."
)


def _llm_team(*templates: Expert) -> ExpertRecommendations:
    return ExpertRecommendations(
        experts=[
            RecommendedExpert(template_id=t.id, name=t.name, role=t.role)
            for t in templates
        ],
        source="llm",
    )


@pytest.fixture
def plan_world(
    reads: dict[str, AsyncMock], mocker: MockerFixture
) -> dict[str, AsyncMock]:
    """A Founder with a three-template roster and nobody hired yet."""
    hired: dict[str, Expert] = {}

    async def hire(user_id, template_id, name, surface=None, via=None):
        # Like the real hire, a template already hired comes back as is.
        if template_id not in hired:
            template = next(t for t in ROSTER if t.id == template_id)
            hired[template_id] = _hire(template_id, template.name)
        return HireResult(expert=hired[template_id])

    reads["list_templates"].return_value = ROSTER
    reads["read_wizard_answers"].return_value = ("Founder/CEO", [])
    reads["list_experts"].side_effect = lambda *a, **k: list(hired.values())
    hire_expert = AsyncMock(side_effect=hire)
    mocker.patch.object(experts_db, "hire_expert", new=hire_expert)
    return {**reads, "hire_expert": hire_expert}


@pytest.fixture
def lenient_client() -> Iterator[fastapi.testclient.TestClient]:
    """A client that hands back a server error as the 500 a browser gets."""
    with fastapi.testclient.TestClient(app, raise_server_exceptions=False) as lenient:
        yield lenient


def _hired_templates(hire_expert: AsyncMock) -> list[str]:
    return [call.args[1] for call in hire_expert.await_args_list]


def test_the_plan_is_pending_while_the_pipeline_runs(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    upload_part()
    finalize()
    assert dumps.row is not None
    # As the row reads while the greeting job is still writing.
    dumps.row.status = BrainDumpStatus.extracting
    dumps.row.suggestedPrompts = []
    dumps.row.greeting = None

    body = client.get(PLAN_URL).json()

    assert body["ready"] is False
    # What a client that stops waiting shows: the role plan, not stored.
    assert body["source"] == "fallback"
    assert body["skipped_dump"] is False
    assert body["tasks"][0]["title"].startswith("Build a list of 25 companies")
    assert dumps.row.suggestedPrompts == []


def test_a_skipped_dump_gets_the_role_plan_stored_on_first_read(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")

    body = client.get(PLAN_URL).json()

    assert body["ready"] is True
    assert body["source"] == "fallback"
    assert body["skipped_dump"] is True
    assert body["heard"] is None
    assert [e["name"] for e in body["team"]] == ["Max", "Maria"]
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-max",
        "tpl-maria",
        None,
        None,
        "tpl-maria",
    ]
    assert body["tasks"][0]["why"] == "Most founders start here."
    assert dumps.row is not None
    assert len(dumps.row.suggestedPrompts) == 5
    # The ids a first read hands out are the ones Home keeps using.
    again = client.get(PLAN_URL).json()
    assert [t["id"] for t in again["tasks"]] == [t["id"] for t in body["tasks"]]
    home = client.get(SUGGESTED_TASKS_URL).json()
    assert home["stored"] is True
    assert [t["title"] for t in home["tasks"]] == [t["title"] for t in body["tasks"]]


def test_a_read_dump_serves_its_plan_with_what_otto_heard(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    upload_part()
    finalize()
    assert dumps.row is not None
    dumps.row.greeting = (
        "You sell hand-drawn stickers on Etsy. Listings eat your evenings. "
        "Pick a task to start with, or tell me something else."
    )
    dumps.row.suggestedPrompts = [
        {"title": "Rewrite listings", "prompt": ".", "expert_template_id": "tpl-maria"},
        {"title": "Find shops", "prompt": ".", "expert_template_id": "tpl-max"},
        {"title": "Track orders", "prompt": ".", "why": "So nothing slips."},
        {"title": "Chase invoices", "prompt": ".", "expert_template_id": "tpl-nadia"},
    ]
    dumps.row.recommendedExperts = ExpertRecommendations(
        experts=[
            RecommendedExpert(template_id=t, name=n, role=r)
            for t, n, r in [
                ("tpl-maria", "Maria", "Marketing"),
                ("tpl-max", "Max", "Sales"),
                ("tpl-mina", "Mina", "Bookkeeping"),
                ("tpl-nadia", "Nadia", "Research"),
            ]
        ],
        source="llm",
    ).model_dump()

    body = client.get(PLAN_URL).json()

    assert body["ready"] is True
    assert body["source"] == "llm"
    assert body["skipped_dump"] is False
    assert body["heard"] == (
        "You sell hand-drawn stickers on Etsy. Listings eat your evenings."
    )
    assert [e["name"] for e in body["team"]] == ["Maria", "Max", "Mina"]
    # Nadia is not on the capped team, so her task is Otto's.
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-maria",
        "tpl-max",
        None,
        None,
    ]
    assert body["tasks"][2]["why"] == "So nothing slips."


def test_an_owner_reads_as_hired_once_the_expert_is_on_the_team(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")
    assert client.get(PLAN_URL).json()["tasks"][0]["expert_id"] is None

    plan_world["list_experts"].side_effect = lambda *a, **k: [_hire("tpl-max", "Max")]

    tasks = client.get(SUGGESTED_TASKS_URL).json()["tasks"]
    assert [t["expert_id"] for t in tasks] == [
        "expert-max",
        None,  # Maria is not hired
        None,
        None,
        None,
    ]


def test_hiring_the_plan_hires_its_team_once(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")

    first = client.post(PLAN_HIRE_URL)

    assert first.status_code == 200
    assert first.json() == {
        "hired": [
            {"template_id": "tpl-max", "expert_id": "expert-max", "name": "Max"},
            {"template_id": "tpl-maria", "expert_id": "expert-maria", "name": "Maria"},
        ],
        "failed": [],
    }
    for call in plan_world["hire_expert"].await_args_list:
        assert call.kwargs == {"surface": "onboarding", "via": "plan"}

    second = client.post(PLAN_HIRE_URL)

    assert second.json() == first.json()
    assert plan_world["hire_expert"].await_count == 2


def test_a_failed_hire_is_reported_and_the_rest_still_join(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")
    succeed = plan_world["hire_expert"].side_effect

    async def hire(user_id, template_id, name, **kwargs):
        if template_id == "tpl-max":
            raise RuntimeError("setup failed")
        return await succeed(user_id, template_id, name, **kwargs)

    plan_world["hire_expert"].side_effect = hire

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json() == {
        "hired": [
            {"template_id": "tpl-maria", "expert_id": "expert-maria", "name": "Maria"}
        ],
        "failed": [{"template_id": "tpl-max", "name": "Max"}],
    }


def test_the_expert_limit_stops_the_hire_and_reports_the_rest(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")
    plan_world["hire_expert"].side_effect = ExpertLimitExceededError(20)

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json()["hired"] == []
    assert [f["name"] for f in response.json()["failed"]] == ["Max", "Maria"]
    plan_world["hire_expert"].assert_awaited_once()


def test_a_brain_dump_becomes_a_ranked_owned_plan_and_a_hired_team_on_home(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    generation: dict[str, AsyncMock],
    mocker: MockerFixture,
):
    """The whole path: a typed dump, its greeting's prompts ranked and owned
    by Jev, the plan step, the hire after the trial, and Home's list."""
    generation["generate_intro"].return_value = (
        "You sell stickers on Etsy. Listings eat your evenings.",
        [
            SuggestedPrompt(title="Track orders", prompt="Track.", icon="chart-bar"),
            SuggestedPrompt(title="Rewrite listings", prompt="Rewrite."),
            SuggestedPrompt(title="Find gift shops", prompt="Find.", icon="target"),
            SuggestedPrompt(title="Plan a drop", prompt="Plan."),
            SuggestedPrompt(title="Draft posts", prompt="Draft.", icon="megaphone"),
        ],
    )
    plan_world["generate_expert_recommendations"].return_value = _llm_team(*ROSTER)
    answers: dict[str, Any] = {}
    for index, (score, owner) in enumerate(
        [
            (1, "tpl-mina"),
            (4, "tpl-maria"),
            (3, "tpl-max"),
            (0, "otto"),
            (2, "tpl-maria"),
        ]
    ):
        answers[f"rank_{index}"] = {"score": score}
        answers[f"owner_{index}"] = {"choice": owner, "probabilities": {owner: 0.9}}
    mocker.patch.object(plan, "_api_key", "test-key")
    mocker.patch.object(
        plan,
        "call_jev",
        new=AsyncMock(
            return_value=JevCallResult(
                answers=answers,
                request="{}",
                response="{}",
                latency_ms=90.0,
                input_tokens=1,
                output_tokens=1,
                request_id="req-1",
                truncated=False,
                truncation_note="",
            )
        ),
    )

    finalize(input_mode="typed", text="I sell stickers on Etsy and need help.")
    shown = client.get(PLAN_URL).json()

    assert shown["ready"] is True
    assert shown["source"] == "llm"
    assert [t["title"] for t in shown["tasks"]] == [
        "Rewrite listings",
        "Find gift shops",
        "Draft posts",
        "Track orders",
        "Plan a drop",
    ]
    assert [e["name"] for e in shown["team"]] == ["Maria", "Max", "Mina"]

    hired = client.post(PLAN_HIRE_URL).json()
    assert [h["name"] for h in hired["hired"]] == ["Maria", "Max", "Mina"]

    home = client.get(SUGGESTED_TASKS_URL).json()["tasks"]
    assert [(t["title"], t["expert_id"]) for t in home] == [
        ("Rewrite listings", "expert-maria"),
        ("Find gift shops", "expert-max"),
        ("Draft posts", "expert-maria"),
        ("Track orders", "expert-mina"),
        ("Plan a drop", None),
    ]


def test_the_expert_limit_still_reports_teammates_already_on_the_team(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")
    # Max cannot join; Maria, later in the team, is already on it.
    plan_world["list_experts"].side_effect = lambda *a, **k: [
        _hire("tpl-maria", "Maria")
    ]
    plan_world["hire_expert"].side_effect = ExpertLimitExceededError(20)

    body = client.post(PLAN_HIRE_URL).json()

    assert body == {
        "hired": [
            {"template_id": "tpl-maria", "expert_id": "expert-maria", "name": "Maria"}
        ],
        "failed": [{"template_id": "tpl-max", "name": "Max"}],
    }


def test_a_failed_take_plans_and_hires_from_the_role_not_the_team_it_stored(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], generation: dict[str, AsyncMock]
):
    """The greeting died after the team job stored its read of the take.
    With no tasks from that read, its team would own nothing, so the plan
    is the role's, and so is the team that gets hired."""
    plan_world["generate_expert_recommendations"].return_value = _llm_team(ROSTER[2])
    generation["generate_intro"].side_effect = RuntimeError("model down")
    finalize(input_mode="typed", text=TRANSCRIPT)
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == "understanding_failed"
    assert dumps.row.transcript == TRANSCRIPT
    assert not dumps.row.greeting
    assert dumps.row.recommendedExperts["source"] == "llm"

    first = client.get(PLAN_URL).json()
    second = client.get(PLAN_URL).json()

    for body in (first, second):
        assert body["ready"] is True
        assert body["source"] == "fallback"
        assert body["skipped_dump"] is False
        assert body["heard"] is None
        assert [e["name"] for e in body["team"]] == ["Max", "Maria"]
    assert [t["title"] for t in first["tasks"]] == FOUNDER_TITLES
    assert second["tasks"] == first["tasks"]

    hired = client.post(PLAN_HIRE_URL).json()

    assert [h["name"] for h in hired["hired"]] == ["Max", "Maria"]
    assert hired["failed"] == []
    assert _hired_templates(plan_world["hire_expert"]) == ["tpl-max", "tpl-maria"]


def test_a_user_with_no_dump_gets_the_role_plan_stored_on_first_read(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    body = client.get(PLAN_URL).json()

    assert body["ready"] is True
    assert body["skipped_dump"] is True
    assert body["source"] == "fallback"
    assert body["heard"] is None
    assert [t["title"] for t in body["tasks"]] == FOUNDER_TITLES
    assert dumps.row is not None
    assert len(dumps.row.suggestedPrompts) == len(FOUNDER_TITLES)
    home = client.get(SUGGESTED_TASKS_URL).json()
    assert home["stored"] is True
    assert [t["id"] for t in home["tasks"]] == [t["id"] for t in body["tasks"]]


def test_a_voice_take_still_transcribing_is_pending_and_stores_nothing(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    """No transcript yet, which on its own reads like a skip: the plan must
    wait for it rather than store the role plan over the take."""
    upload_part()
    assert dumps.row is not None
    dumps.row.status = BrainDumpStatus.transcribing

    body = client.get(PLAN_URL).json()

    assert body["ready"] is False
    assert body["skipped_dump"] is False
    assert [t["title"] for t in body["tasks"]] == FOUNDER_TITLES
    assert dumps.row.suggestedPrompts == []


def test_a_generic_greeting_swapped_for_the_role_set_keeps_the_dumps_team(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], generation: dict[str, AsyncMock]
):
    """The greeting fell back to the generic starters, so the plan job
    stored the role's set owned from the team the take was read for. Otto
    wrote none of those tasks from what he heard, so he claims nothing."""
    generation["generate_intro"].return_value = (
        HEARD_GREETING,
        intro.fallback_prompts(),
    )
    plan_world["generate_expert_recommendations"].return_value = _llm_team(*ROSTER)
    finalize(input_mode="typed", text=TRANSCRIPT)

    body = client.get(PLAN_URL).json()

    assert body["ready"] is True
    assert body["source"] == "fallback"
    assert body["skipped_dump"] is False
    assert body["heard"] is None
    # Mina is not on the role's own team; she is on the one this take got.
    assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
    assert [t["title"] for t in body["tasks"]] == FOUNDER_TITLES
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-max",
        "tpl-maria",
        None,
        None,
        "tpl-maria",
    ]


FAILED_READS = pytest.mark.parametrize(
    "failing_read",
    ["read_wizard_answers", "list_templates"],
    ids=["role read", "template read"],
)


@FAILED_READS
def test_a_plan_whose_read_fails_is_an_error_not_a_stored_degraded_plan(
    failing_read: str,
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    finalize(input_mode="skipped")
    assert dumps.row is not None
    plan_world[failing_read].side_effect = RuntimeError("database unavailable")

    response = lenient_client.get(PLAN_URL)

    assert response.status_code == 500
    assert dumps.row.suggestedPrompts == []
    # Once the read is back, the first plan stored is the whole one.
    plan_world[failing_read].side_effect = None
    body = lenient_client.get(PLAN_URL).json()
    assert [e["name"] for e in body["team"]] == ["Max", "Maria"]
    assert len(dumps.row.suggestedPrompts) == len(FOUNDER_TITLES)


def test_a_team_left_empty_by_a_failed_roster_read_is_replaced_by_the_role_team(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    # The team job read no roster and stored nobody, with nothing to raise
    # either: that is no answer, so the plan and the hire use the role team.
    finalize(input_mode="typed", text=TRANSCRIPT)
    assert dumps.row is not None
    dumps.row.recommendedExperts = {
        "source": "fallback",
        "diagnosis": "",
        "experts": [],
        "raise_suggestion": None,
    }

    body = lenient_client.get(PLAN_URL).json()

    assert [e["name"] for e in body["team"]] == ["Max", "Maria"]
    lenient_client.post(PLAN_HIRE_URL)
    assert plan_world["hire_expert"].await_count == 2


@FAILED_READS
def test_a_dump_whose_team_was_never_stored_fails_the_hire_rather_than_hire_nobody(
    failing_read: str,
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    # The team job's write failed: the dump completed with no stored team, so
    # the plan's team is the role team, read the way the role plan reads it.
    finalize(input_mode="typed", text=TRANSCRIPT)
    assert dumps.row is not None
    dumps.row.recommendedExperts = None
    shown = lenient_client.get(PLAN_URL).json()
    assert [e["name"] for e in shown["team"]] == ["Max", "Maria"]
    plan_world[failing_read].side_effect = RuntimeError("database unavailable")

    response = lenient_client.post(PLAN_HIRE_URL)

    assert response.status_code == 500
    plan_world["hire_expert"].assert_not_awaited()


def test_home_still_lists_tasks_when_the_roster_read_fails_but_stores_none(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    finalize(input_mode="skipped")
    assert dumps.row is not None
    plan_world["list_templates"].side_effect = RuntimeError("database unavailable")

    response = lenient_client.get(SUGGESTED_TASKS_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is False
    assert [t["title"] for t in body["tasks"]] == FOUNDER_TITLES
    # Without the roster, nobody can be named an owner.
    assert all(t["expert_template_id"] is None for t in body["tasks"])
    assert dumps.row.suggestedPrompts == []


def test_home_lists_nothing_when_the_role_read_fails_rather_than_another_list(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    # The generic set under the role set's ids would let an x remove a task
    # the user never saw.
    finalize(input_mode="skipped")
    assert dumps.row is not None
    plan_world["read_wizard_answers"].side_effect = RuntimeError("database down")

    response = lenient_client.get(SUGGESTED_TASKS_URL)

    assert response.status_code == 500
    assert dumps.row.suggestedPrompts == []


@FAILED_READS
def test_a_task_change_whose_read_fails_is_an_error_and_stores_nothing(
    failing_read: str,
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    finalize(input_mode="skipped")
    assert dumps.row is not None
    plan_world[failing_read].side_effect = RuntimeError("database unavailable")

    response = lenient_client.post(
        SUGGESTED_TASKS_URL, json={"task_ids": ["0"], "state": "dismissed"}
    )

    assert response.status_code == 500
    assert dumps.row.suggestedPrompts == []


def _finalize_with_the_whole_roster(plan_world: dict[str, AsyncMock]) -> None:
    """A read take whose team is all three of the roster, in roster order."""
    plan_world["generate_expert_recommendations"].return_value = _llm_team(*ROSTER)
    finalize(input_mode="typed", text=TRANSCRIPT)
    team = client.get(PLAN_URL).json()["team"]
    assert [e["name"] for e in team] == ["Max", "Maria", "Mina"]


def test_the_expert_limit_on_the_second_hire_keeps_the_first_and_reports_the_rest(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    _finalize_with_the_whole_roster(plan_world)
    succeed = plan_world["hire_expert"].side_effect

    async def hire(user_id, template_id, name, **kwargs):
        if template_id != "tpl-max":
            raise ExpertLimitExceededError(20)
        return await succeed(user_id, template_id, name, **kwargs)

    plan_world["hire_expert"].side_effect = hire

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json() == {
        "hired": [{"template_id": "tpl-max", "expert_id": "expert-max", "name": "Max"}],
        "failed": [
            {"template_id": "tpl-maria", "name": "Maria"},
            {"template_id": "tpl-mina", "name": "Mina"},
        ],
    }
    # Nobody is tried after the limit.
    assert _hired_templates(plan_world["hire_expert"]) == ["tpl-max", "tpl-maria"]


def test_hiring_again_after_a_failed_hire_hires_only_who_failed(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    _finalize_with_the_whole_roster(plan_world)
    succeed = plan_world["hire_expert"].side_effect

    async def hire(user_id, template_id, name, **kwargs):
        if template_id == "tpl-maria":
            raise RuntimeError("setup failed")
        return await succeed(user_id, template_id, name, **kwargs)

    plan_world["hire_expert"].side_effect = hire
    first = client.post(PLAN_HIRE_URL).json()
    assert [h["name"] for h in first["hired"]] == ["Max", "Mina"]
    assert first["failed"] == [{"template_id": "tpl-maria", "name": "Maria"}]
    plan_world["hire_expert"].side_effect = succeed
    tried_before = plan_world["hire_expert"].await_count

    second = client.post(PLAN_HIRE_URL).json()

    retried = _hired_templates(plan_world["hire_expert"])[tried_before:]
    assert retried == ["tpl-maria"]
    assert [h["name"] for h in second["hired"]] == ["Max", "Maria", "Mina"]
    assert second["failed"] == []


def test_a_roster_read_that_fails_leaves_the_plan_hire_to_hire_expert(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    """Without the roster nobody reads as already hired, so every teammate
    goes to ``hire_expert``, which answers for one already on the team."""
    finalize(input_mode="skipped")
    assert client.post(PLAN_HIRE_URL).status_code == 200
    plan_world["list_experts"].side_effect = RuntimeError("database unavailable")

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json() == {
        "hired": [
            {"template_id": "tpl-max", "expert_id": "expert-max", "name": "Max"},
            {"template_id": "tpl-maria", "expert_id": "expert-maria", "name": "Maria"},
        ],
        "failed": [],
    }
    assert _hired_templates(plan_world["hire_expert"]) == [
        "tpl-max",
        "tpl-maria",
        # The second time, from no roster: asked again, nobody hired twice.
        "tpl-max",
        "tpl-maria",
    ]
