"""Endpoint tests for the onboarding brain-dump routes.

Everything below the router (Prisma, Redis, cloud storage, OpenAI) is
mocked at the point of use, so these run without a live stack.
"""

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert, HireResult
from backend.api.features.onboarding_dump import db as dump_db
from backend.api.features.onboarding_dump import (
    interview,
    intro,
    plan,
    plan_hire,
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
    FinalizeResponse,
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.api.features.onboarding_dump.role_task_sets import ROLE_FALLBACK_TASKS
from backend.api.features.store.exceptions import VirusDetectedError
from backend.blocks.typesafe._client import JevCallResult
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    INTERVIEW_STAGE_WRAP_UP_ISSUED,
    ONBOARDING_INTERVIEW_KIND,
    ChatMessage,
    ChatSession,
)
from backend.copilot.onboarding_interview import INTERVIEW_TURN_CAP
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.understanding import BusinessUnderstandingInput
from backend.util.exceptions import (
    ConflictError,
    ExpertLimitExceededError,
    NotFoundError,
)
from backend.util.gcs_utils_test import _gcs_http_error
from backend.util.posthog_events import PostHogEvent

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
PLAN_SETTLE_URL = "/onboarding/brain-dump/plan/settle"
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

    async def store_greeting(
        self, user_id: str, recording_id: str, greeting: str, prompts: list[Any]
    ) -> bool:
        """Mirrors the real write: the tasks go in only while the take has none."""
        if self.row is None or self.row.recordingId != recording_id:
            return False
        fields: dict[str, Any] = {"greeting": greeting, "errorCode": None}
        if not self.row.suggestedPrompts:
            fields["suggestedPrompts"] = prompts
        return await self.update_dump(user_id, recording_id, **fields)

    async def replace_team(
        self, user_id: str, recording_id: str, read_at: datetime, team: Any
    ) -> bool:
        if (
            self.row is None
            or self.row.recordingId != recording_id
            or self.row.updatedAt != read_at
        ):
            return False
        self.row.recommendedExperts = team
        self.touch()
        return True

    async def start_dump(
        self,
        user_id: str,
        recording_id: str,
        input_mode: BrainDumpInputMode,
        *,
        call_take: bool = False,
    ) -> OnboardingBrainDump:
        # Mirrors the real `start_dump`: a take already moving through
        # the pipeline is returned untouched, so a replayed part 0 or a
        # repeated finalize cannot reset it, and a typed take displaces
        # another only as a call's take displaces another call's.
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
                or (
                    (input_mode != BrainDumpInputMode.voice or call_take)
                    and not dump_db.call_take_displaces(
                        recording_id, self.row.recordingId, call_take=call_take
                    )
                )
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
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    # A paid-up user, unless a test says otherwise.
    app.dependency_overrides[enforce_payment_paywall] = lambda: None
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
    mocker.patch(f"{module}.store_greeting", new=store.store_greeting)
    mocker.patch(f"{module}.replace_team", new=store.replace_team)
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
def reads(mocker: MockerFixture) -> Iterator[dict[str, AsyncMock]]:
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
    module when it runs, so that is where it is replaced. The plan's
    cached roster is cleared on both sides, so no test reads another's.
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
    plan_service._roster.cache_clear()
    yield mocks
    plan_service._roster.cache_clear()


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


SERVER_NAMED_TAKES = ["interview-" + "a" * 32, "interview-a-typed-interview"]


@pytest.mark.parametrize("recording_id", SERVER_NAMED_TAKES)
@pytest.mark.parametrize("input_mode", ["voice", "typed", "skipped"])
def test_a_finalize_naming_a_take_only_the_server_makes_is_refused(
    dumps: DumpStore, recording_id: str, input_mode: str
):
    """The server names interview and call takes itself; a client's take is
    never read as one."""
    response = client.post(
        FINALIZE_URL,
        json={"recording_id": recording_id, "input_mode": input_mode, "text": "Hi."},
    )

    assert response.status_code == 422
    assert dumps.row is None


@pytest.mark.parametrize("recording_id", SERVER_NAMED_TAKES)
def test_a_part_or_a_discard_naming_a_take_only_the_server_makes_is_refused(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], recording_id: str
):
    part = client.post(
        PARTS_URL,
        files={"file": ("part-0.webm", b"chunk", "audio/webm")},
        data={"recording_id": recording_id, "part_index": "0"},
    )
    discard = client.delete(
        DISCARD_URL.rstrip("/"), params={"recording_id": recording_id}
    )

    assert (part.status_code, discard.status_code) == (422, 422)
    assert dumps.row is None
    storage_mocks["append_part"].assert_not_awaited()
    storage_mocks["mark_discarded"].assert_not_awaited()


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


# The Founder/CEO role team from this roster is all three: Max (sales) and
# Maria (marketing) for the role's own needs, then Mina (ops), who fills the
# row so everyone starts with a team of three.
ROSTER = [
    _roster_template("tpl-max", "Max", "Sales"),
    _roster_template("tpl-maria", "Maria", "Marketing"),
    _roster_template("tpl-mina", "Mina", "Ops"),
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
        joined = template_id not in hired
        if joined:
            template = next(t for t in ROSTER if t.id == template_id)
            hired[template_id] = _hire(template_id, template.name)
        return HireResult.of(hired[template_id], joined=joined)

    reads["list_templates"].return_value = ROSTER
    reads["read_wizard_answers"].return_value = ("Founder/CEO", [])
    reads["list_experts"].side_effect = lambda *a, **k: list(hired.values())
    hire_expert = AsyncMock(side_effect=hire)
    mocker.patch.object(experts_db, "hire_expert", new=hire_expert)
    claims: set[str] = set()

    async def claim(key, value, nx, ex):
        is_new = key not in claims
        claims.add(key)
        return is_new or None

    redis = MagicMock()
    redis.set = AsyncMock(side_effect=claim)
    mocker.patch.object(plan_hire, "get_redis_async", new=AsyncMock(return_value=redis))
    return {**reads, "hire_expert": hire_expert}


@pytest.fixture
def hire_events(mocker: MockerFixture) -> MagicMock:
    """What the plan hire sends to PostHog."""
    return mocker.patch.object(plan_hire, "emit_funnel_event")


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
    assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-max",
        "tpl-maria",
        "tpl-mina",
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
    # Nadia owns the fourth task, so she takes the last seat before Mina,
    # who owns none.
    assert [e["name"] for e in body["team"]] == ["Maria", "Max", "Nadia"]
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-maria",
        "tpl-max",
        None,
        "tpl-nadia",
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
            {"template_id": "tpl-mina", "expert_id": "expert-mina", "name": "Mina"},
        ],
        "failed": [],
    }
    for call in plan_world["hire_expert"].await_args_list:
        assert call.kwargs == {"surface": "onboarding", "via": "plan"}

    second = client.post(PLAN_HIRE_URL)

    assert second.json() == first.json()
    assert plan_world["hire_expert"].await_count == 3


def test_a_plan_that_could_not_be_settled_hires_nobody(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    hire_events: MagicMock,
    mocker: MockerFixture,
):
    """Three lost writes leave the temporary team, which the pipeline may
    still replace, so the client is told to try again."""
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    mocker.patch.object(
        plan_service.db, "replace_plan", new=AsyncMock(return_value=False)
    )

    # The API's handlers answer this with a 409; this test app has none.
    with pytest.raises(ConflictError):
        client.post(PLAN_HIRE_URL)

    plan_world["hire_expert"].assert_not_awaited()
    hire_events.assert_not_called()


def test_the_plan_hire_reports_who_joined_once(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    hire_events: MagicMock,
    test_user_id: str,
):
    """A repeat of the hire, as a reload of the preparing screen sends,
    changes nothing and reports nothing."""
    finalize(input_mode="skipped")

    client.post(PLAN_HIRE_URL)
    client.post(PLAN_HIRE_URL)

    hire_events.assert_called_once_with(
        test_user_id,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": 3, "already_on_team": 0, "failed": 0},
    )


def test_a_repeat_hire_that_cannot_read_the_roster_reports_nobody_joined(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    hire_events: MagicMock,
):
    """With the roster unreadable the repeat asks for every hire again, and
    each comes back as the expert already on the team: nobody new joined."""
    finalize(input_mode="skipped")
    client.post(PLAN_HIRE_URL)
    plan_world["list_experts"].side_effect = RuntimeError("roster read failed")

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert plan_world["hire_expert"].await_count == 6
    hire_events.assert_called_once()


def test_a_failed_hire_is_reported_and_the_rest_still_join(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
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
            {"template_id": "tpl-maria", "expert_id": "expert-maria", "name": "Maria"},
            {"template_id": "tpl-mina", "expert_id": "expert-mina", "name": "Mina"},
        ],
        "failed": [{"template_id": "tpl-max", "name": "Max"}],
    }
    hire_events.assert_called_once_with(
        ANY,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": 2, "already_on_team": 0, "failed": 1},
    )


def _max_cannot_join(plan_world: dict[str, AsyncMock]) -> None:
    succeed = plan_world["hire_expert"].side_effect

    async def hire(user_id, template_id, name, **kwargs):
        if template_id == "tpl-max":
            raise RuntimeError("setup failed")
        return await succeed(user_id, template_id, name, **kwargs)

    plan_world["hire_expert"].side_effect = hire


def test_a_repeat_that_finds_the_same_failure_reports_nothing(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
):
    """A reload re-asks how the hire went: the answer still says who could not
    join, and the event stays the first call's."""
    finalize(input_mode="skipped")
    _max_cannot_join(plan_world)

    client.post(PLAN_HIRE_URL)
    repeat = client.post(PLAN_HIRE_URL)

    assert repeat.json()["failed"] == [{"template_id": "tpl-max", "name": "Max"}]
    hire_events.assert_called_once_with(
        ANY,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": 2, "already_on_team": 0, "failed": 1},
    )


def test_a_repeat_nobody_could_join_again_reports_nothing(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
):
    finalize(input_mode="skipped")
    plan_world["hire_expert"].side_effect = ExpertLimitExceededError(20)

    client.post(PLAN_HIRE_URL)
    client.post(PLAN_HIRE_URL)

    hire_events.assert_called_once_with(
        ANY,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": 0, "already_on_team": 0, "failed": 3},
    )


def test_a_repeat_that_hires_someone_who_failed_reports_them(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
):
    finalize(input_mode="skipped")
    succeed = plan_world["hire_expert"].side_effect
    _max_cannot_join(plan_world)
    client.post(PLAN_HIRE_URL)

    plan_world["hire_expert"].side_effect = succeed
    client.post(PLAN_HIRE_URL)

    assert hire_events.call_count == 2
    assert hire_events.call_args.args[2] == {
        "hired": 1,
        "already_on_team": 2,
        "failed": 0,
    }


def test_a_failed_hire_is_still_reported_when_redis_is_down(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    hire_events: MagicMock,
    mocker: MockerFixture,
):
    finalize(input_mode="skipped")
    _max_cannot_join(plan_world)
    mocker.patch.object(
        plan_hire, "get_redis_async", new=AsyncMock(side_effect=OSError("down"))
    )

    client.post(PLAN_HIRE_URL)

    hire_events.assert_called_once()


def test_a_redis_that_never_answers_does_not_hold_the_hire(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    hire_events: MagicMock,
    mocker: MockerFixture,
):
    finalize(input_mode="skipped")
    _max_cannot_join(plan_world)

    async def hang():
        await asyncio.Event().wait()

    mocker.patch.object(plan_hire, "get_redis_async", new=hang)
    mocker.patch.object(plan_hire, "_FAILURE_CLAIM_TIMEOUT_SECONDS", 0.01)

    assert client.post(PLAN_HIRE_URL).status_code == 200
    hire_events.assert_called_once()


def test_a_user_still_behind_the_paywall_hires_nobody(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    lenient_client: fastapi.testclient.TestClient,
):
    # Left Stripe without paying: the plan was shown, the hire is refused.
    finalize(input_mode="skipped")

    def paywalled() -> None:
        raise fastapi.HTTPException(status_code=402)

    app.dependency_overrides[enforce_payment_paywall] = paywalled

    assert lenient_client.get(PLAN_URL).status_code == 200
    assert lenient_client.post(PLAN_HIRE_URL).status_code == 402
    plan_world["hire_expert"].assert_not_awaited()


def test_a_plan_with_nobody_on_its_team_hires_nobody(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
):
    # A catalog with no template for the role: the plan is all Otto's.
    plan_world["list_templates"].return_value = []
    finalize(input_mode="skipped")
    assert client.get(PLAN_URL).json()["team"] == []

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json() == {"hired": [], "failed": []}
    plan_world["hire_expert"].assert_not_awaited()
    hire_events.assert_not_called()


def test_the_expert_limit_stops_the_hire_and_reports_the_rest(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="skipped")
    plan_world["hire_expert"].side_effect = ExpertLimitExceededError(20)

    response = client.post(PLAN_HIRE_URL)

    assert response.status_code == 200
    assert response.json()["hired"] == []
    assert [f["name"] for f in response.json()["failed"]] == ["Max", "Maria", "Mina"]
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
    dumps: DumpStore, plan_world: dict[str, AsyncMock], hire_events: MagicMock
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
        "failed": [
            {"template_id": "tpl-max", "name": "Max"},
            {"template_id": "tpl-mina", "name": "Mina"},
        ],
    }
    # Maria was on the team before this hire, so she is not counted as hired.
    hire_events.assert_called_once_with(
        ANY,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": 0, "already_on_team": 1, "failed": 2},
    )


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
        assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
    assert [t["title"] for t in first["tasks"]] == FOUNDER_TITLES
    assert second["tasks"] == first["tasks"]

    hired = client.post(PLAN_HIRE_URL).json()

    assert [h["name"] for h in hired["hired"]] == ["Max", "Maria", "Mina"]
    assert hired["failed"] == []
    assert _hired_templates(plan_world["hire_expert"]) == [
        "tpl-max",
        "tpl-maria",
        "tpl-mina",
    ]


def _mid_pipeline(dumps: DumpStore) -> None:
    """The row as it reads while the greeting and team jobs are still running."""
    assert dumps.row is not None
    dumps.row.status = BrainDumpStatus.extracting
    dumps.row.suggestedPrompts = []
    dumps.row.greeting = None
    dumps.row.recommendedExperts = None


def test_the_pending_plan_says_how_far_the_jobs_have_got(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    assert dumps.row is not None

    progress = [client.get(PLAN_URL).json()["progress"]]
    dumps.row.recommendedExperts = _llm_team(*ROSTER).model_dump()
    progress.append(client.get(PLAN_URL).json()["progress"])
    dumps.row.greeting = HEARD_GREETING
    progress.append(client.get(PLAN_URL).json()["progress"])
    dumps.row.status = BrainDumpStatus.completed
    ready = client.get(PLAN_URL).json()

    assert progress == [
        {"tasks_written": False, "team_chosen": False},
        {"tasks_written": False, "team_chosen": True},
        {"tasks_written": True, "team_chosen": True},
    ]
    assert ready["ready"] is True
    assert ready["progress"] is None


def test_settling_mid_pipeline_stores_the_role_plan_the_step_shows(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    shown = client.get(PLAN_URL).json()

    settled = client.post(PLAN_SETTLE_URL).json()
    again = client.get(PLAN_URL).json()

    assert shown["ready"] is False
    for body in (settled, again):
        assert body["ready"] is True
        assert body["source"] == "fallback"
        assert body["heard"] is None
        assert body["progress"] is None
        assert body["tasks"] == shown["tasks"]
        assert body["team"] == shown["team"]
    assert [t["title"] for t in settled["tasks"]] == FOUNDER_TITLES
    assert [e["name"] for e in settled["team"]] == ["Max", "Maria", "Mina"]
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.extracting
    assert [e["name"] for e in dumps.row.recommendedExperts["experts"]] == [
        "Max",
        "Maria",
        "Mina",
    ]


def test_a_take_settled_while_it_is_transcribed_is_not_called_skipped(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    """A retried finalize can hand the step a take whose transcription is
    still running; giving up on it settles a recorded take, not a skip."""
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    assert dumps.row is not None
    dumps.row.status = BrainDumpStatus.transcribing
    dumps.row.transcript = None

    settled = client.post(PLAN_SETTLE_URL).json()

    assert settled["ready"] is True
    assert settled["skipped_dump"] is False
    assert [t["title"] for t in settled["tasks"]] == FOUNDER_TITLES
    assert [e["name"] for e in settled["team"]] == ["Max", "Maria", "Mina"]


def test_settling_a_ready_plan_changes_nothing(
    dumps: DumpStore, plan_world: dict[str, AsyncMock], generation: dict[str, AsyncMock]
):
    generation["generate_intro"].return_value = (
        HEARD_GREETING,
        [SuggestedPrompt(title=f"Task {i}", prompt="Do it.") for i in range(5)],
    )
    plan_world["generate_expert_recommendations"].return_value = _llm_team(*ROSTER)
    finalize(input_mode="typed", text=TRANSCRIPT)
    ready = client.get(PLAN_URL).json()
    assert dumps.row is not None
    stored = (dumps.row.suggestedPrompts, dumps.row.recommendedExperts)

    settled = client.post(PLAN_SETTLE_URL).json()

    assert settled == ready
    assert settled["source"] == "llm"
    assert (dumps.row.suggestedPrompts, dumps.row.recommendedExperts) == stored


def test_a_plan_the_step_gave_up_on_is_what_the_trial_hires_and_home_lists(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    generation: dict[str, AsyncMock],
):
    """The whole path when the wait runs out: the step shows the role plan
    while both jobs are still running, and their answers, arriving after,
    change neither the plan, nor who is hired, nor Home's list."""
    seen: dict[str, Any] = {}

    async def slow_greeting(transcript: str):
        # The step's wait runs out while the greeting is being written.
        seen["shown"] = (await plan_service.settle_plan(dumps.row.userId)).model_dump(
            mode="json"
        )
        return (
            HEARD_GREETING,
            [SuggestedPrompt(title=f"Task {i}", prompt="Do it.") for i in range(5)],
        )

    generation["generate_intro"].side_effect = slow_greeting
    plan_world["generate_expert_recommendations"].return_value = _llm_team(ROSTER[2])

    finalize(input_mode="typed", text=TRANSCRIPT)
    shown = seen["shown"]
    after = client.get(PLAN_URL).json()
    hired = client.post(PLAN_HIRE_URL).json()
    home = client.get(SUGGESTED_TASKS_URL).json()["tasks"]

    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.completed
    assert dumps.row.greeting == HEARD_GREETING
    assert [t["title"] for t in shown["tasks"]] == FOUNDER_TITLES
    assert after["tasks"] == shown["tasks"]
    assert after["team"] == shown["team"]
    assert [h["name"] for h in hired["hired"]] == [e["name"] for e in shown["team"]]
    owners = {e["template_id"]: f"expert-{e['name'].lower()}" for e in shown["team"]}
    assert [(t["title"], t["expert_id"]) for t in home] == [
        (t["title"], owners.get(t["expert_template_id"])) for t in shown["tasks"]
    ]


async def _greeting_stored(dumps: DumpStore) -> None:
    """Yield to the greeting job running beside the caller until it has
    stored the greeting and its tasks."""
    for _ in range(100):
        if dumps.row is not None and dumps.row.greeting:
            return
        await asyncio.sleep(0)
    raise AssertionError("The greeting job never stored its tasks")


def test_a_plan_settled_after_the_greeting_wrote_its_tasks_keeps_them(
    dumps: DumpStore,
    plan_world: dict[str, AsyncMock],
    generation: dict[str, AsyncMock],
    test_user_id: str,
):
    """The wait runs out once the greeting has stored its tasks but while the
    team job is still choosing: the step shows those tasks and what Otto
    heard, with the role team, and the team job's answer, arriving after,
    changes neither the plan, nor who is hired, nor Home's list."""
    written = [f"Task {i}" for i in range(5)]
    generation["generate_intro"].return_value = (
        HEARD_GREETING,
        [SuggestedPrompt(title=title, prompt="Do it.") for title in written],
    )
    seen: dict[str, Any] = {}

    async def team_after_the_greeting(*args: Any, **kwargs: Any):
        await _greeting_stored(dumps)
        seen["shown"] = (await plan_service.settle_plan(test_user_id)).model_dump(
            mode="json"
        )
        return _llm_team(ROSTER[2])

    plan_world["generate_expert_recommendations"].side_effect = team_after_the_greeting

    finalize(input_mode="typed", text=TRANSCRIPT)
    shown = seen["shown"]
    after = client.get(PLAN_URL).json()
    hired = client.post(PLAN_HIRE_URL).json()
    home = client.get(SUGGESTED_TASKS_URL).json()["tasks"]

    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.completed
    assert shown["source"] == "llm"
    assert shown["heard"] == plan_service.heard_from(HEARD_GREETING)
    assert [t["title"] for t in shown["tasks"]] == written
    assert [e["name"] for e in shown["team"]] == ["Max", "Maria", "Mina"]
    assert (after["tasks"], after["team"], after["heard"]) == (
        shown["tasks"],
        shown["team"],
        shown["heard"],
    )
    assert [h["name"] for h in hired["hired"]] == ["Max", "Maria", "Mina"]
    assert [t["title"] for t in home] == written


def test_settling_while_the_plan_job_ranks_keeps_the_greetings_tasks_and_team(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    assert dumps.row is not None
    written = [f"Task {i}" for i in range(5)]
    dumps.row.greeting = HEARD_GREETING
    dumps.row.suggestedPrompts = [
        SuggestedPrompt(title=title, prompt="Do it.").model_dump() for title in written
    ]
    dumps.row.recommendedExperts = _llm_team(ROSTER[2]).model_dump()

    settled = client.post(PLAN_SETTLE_URL).json()
    again = client.get(PLAN_URL).json()

    for body in (settled, again):
        assert body["ready"] is True
        assert body["source"] == "llm"
        assert body["heard"] == plan_service.heard_from(HEARD_GREETING)
        assert [t["title"] for t in body["tasks"]] == written
        assert [e["name"] for e in body["team"]] == ["Mina"]
    assert dumps.row.status == BrainDumpStatus.extracting
    assert dumps.row.recommendedExperts["source"] == "llm"


def test_settling_on_the_generic_starters_settles_on_the_role_plan(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    assert dumps.row is not None
    dumps.row.greeting = GREETING
    dumps.row.suggestedPrompts = [p.model_dump() for p in intro.fallback_prompts()]

    settled = client.post(PLAN_SETTLE_URL).json()

    assert settled["source"] == "fallback"
    assert settled["heard"] is None
    assert [t["title"] for t in settled["tasks"]] == FOUNDER_TITLES


def test_polls_while_the_pipeline_runs_read_the_roster_once(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)
    plan_world["list_templates"].reset_mock()

    bodies = [client.get(PLAN_URL).json() for _ in range(3)]

    assert [body["ready"] for body in bodies] == [False, False, False]
    assert bodies[2]["tasks"] == bodies[0]["tasks"]
    assert plan_world["list_templates"].await_count == 1


def test_hiring_while_the_jobs_run_settles_the_plan_first(
    dumps: DumpStore, plan_world: dict[str, AsyncMock]
):
    """With no paywall, Continue can come before the plan is ready: the team
    hired is the one whose tasks Home then lists."""
    finalize(input_mode="typed", text=TRANSCRIPT)
    _mid_pipeline(dumps)

    hired = client.post(PLAN_HIRE_URL).json()
    plan_after = client.get(PLAN_URL).json()

    assert [h["name"] for h in hired["hired"]] == ["Max", "Maria", "Mina"]
    assert plan_after["ready"] is True
    assert [e["name"] for e in plan_after["team"]] == ["Max", "Maria", "Mina"]
    assert [t["title"] for t in plan_after["tasks"]] == FOUNDER_TITLES


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
    # The numbers are ops work, so that task is Mina's rather than Otto's.
    assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
    assert [t["title"] for t in body["tasks"]] == FOUNDER_TITLES
    assert [t["expert_template_id"] for t in body["tasks"]] == [
        "tpl-max",
        "tpl-maria",
        "tpl-mina",
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
    assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
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

    assert [e["name"] for e in body["team"]] == ["Max", "Maria", "Mina"]
    lenient_client.post(PLAN_HIRE_URL)
    assert plan_world["hire_expert"].await_count == 3


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
    assert [e["name"] for e in shown["team"]] == ["Max", "Maria", "Mina"]
    plan_world[failing_read].side_effect = RuntimeError("database unavailable")
    # The roster that read cached has expired by the time of the hire.
    plan_service._roster.cache_clear()

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
            {"template_id": "tpl-mina", "expert_id": "expert-mina", "name": "Mina"},
        ],
        "failed": [],
    }
    assert _hired_templates(plan_world["hire_expert"]) == [
        "tpl-max",
        "tpl-maria",
        "tpl-mina",
        # The second time, from no roster: asked again, nobody hired twice.
        "tpl-max",
        "tpl-maria",
        "tpl-mina",
    ]


# ---------- the onboarding interview ----------

INTERVIEW_URL = "/onboarding/brain-dump/interview"
INTERVIEW_FINISH_URL = "/onboarding/brain-dump/interview/finish"
# The id recorded on the person's onboarding row when they first opened it.
INTERVIEW_SESSION_ID = "interview-session-1"


def _interview(user_id: str, stage: str | None = None) -> ChatSession:
    session = ChatSession.new(user_id, dry_run=False)
    session.session_id = INTERVIEW_SESSION_ID
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    session.metadata.onboarding_interview_stage = stage
    return session


@pytest.fixture
def start_interview(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(interview, "start_onboarding_interview", new=AsyncMock())


@pytest.fixture
def finalize_interview(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        interview, "finalize_onboarding_interview", new=AsyncMock()
    )


@pytest.fixture
def find_interview(mocker: MockerFixture, test_user_id: str) -> AsyncMock:
    """The lookup of the person's interview by the id on their onboarding
    row: theirs unless a test says they have none."""
    return mocker.patch.object(
        routes,
        "find_onboarding_interview_session",
        new=AsyncMock(return_value=_interview(test_user_id)),
    )


def test_starting_the_interview_answers_it_with_otto_s_first_turn(
    start_interview: AsyncMock, test_user_id: str
):
    session = _interview(test_user_id)
    start_interview.return_value = interview.StartedInterview(
        session=session, turn_id="turn-1"
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session.session_id,
        "turn_id": "turn-1",
        "stage": None,
    }
    # Opened from who the person is, outside the org and team the request
    # runs in.
    start_interview.assert_awaited_once_with(test_user_id)


def test_reopening_an_interview_under_way_starts_no_turn(
    start_interview: AsyncMock, test_user_id: str
):
    session = _interview(test_user_id, stage=INTERVIEW_STAGE_WRAP_UP_ISSUED)
    start_interview.return_value = interview.StartedInterview(
        session=session, turn_id=None
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session.session_id,
        "turn_id": None,
        "stage": INTERVIEW_STAGE_WRAP_UP_ISSUED,
    }


def test_a_greeting_already_being_written_is_followed_not_refused(
    start_interview: AsyncMock, find_interview: AsyncMock, test_user_id: str
):
    """A reload or a second tab while Otto greets: the client reads the
    session, which resumes the turn still running."""
    start_interview.side_effect = fastapi.HTTPException(
        status_code=409, detail=interview.TURN_IN_PROGRESS
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": INTERVIEW_SESSION_ID,
        "turn_id": None,
        "stage": None,
    }
    find_interview.assert_awaited_once_with(test_user_id)


def test_a_running_turn_with_no_interview_on_record_stays_a_409(
    start_interview: AsyncMock, find_interview: AsyncMock
):
    start_interview.side_effect = fastapi.HTTPException(
        status_code=409, detail=interview.TURN_IN_PROGRESS
    )
    find_interview.return_value = None

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 409


def test_an_interview_that_is_over_is_a_200_with_its_stage(
    start_interview: AsyncMock, test_user_id: str
):
    """Back or a reload after the interview ended shows how it ended."""
    session = _interview(test_user_id, stage=INTERVIEW_STAGE_CLOSED)
    start_interview.return_value = interview.StartedInterview(
        session=session, turn_id=None
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": INTERVIEW_SESSION_ID,
        "turn_id": None,
        "stage": INTERVIEW_STAGE_CLOSED,
    }


def test_a_spent_interview_budget_is_a_429_with_its_wait_in_the_body(
    start_interview: AsyncMock,
):
    """The browser's API client keeps a refusal's body, not its headers, so
    the wait is in both."""
    start_interview.side_effect = fastapi.HTTPException(
        status_code=429,
        detail="You have sent a lot of messages in a short time.",
        headers={"Retry-After": "1200"},
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1200"
    assert response.json()["detail"] == {
        "message": "You have sent a lot of messages in a short time.",
        "retry_after_seconds": 1200,
    }


def test_a_429_with_no_wait_names_none(start_interview: AsyncMock):
    """The cap on turns running at once sets no ``Retry-After``."""
    start_interview.side_effect = fastapi.HTTPException(
        status_code=429, detail="Too many chats running at once."
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "message": "Too many chats running at once.",
        "retry_after_seconds": None,
    }


def test_an_unreadable_budget_or_stream_is_a_503(start_interview: AsyncMock):
    start_interview.side_effect = fastapi.HTTPException(
        status_code=503,
        detail="The interview is temporarily unavailable.",
        headers={"Retry-After": "30"},
    )

    response = client.post(INTERVIEW_URL)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"


@pytest.fixture
def interview_starts(mocker: MockerFixture) -> dict[str, AsyncMock]:
    """What only the typed start may do: create an interview, charge its
    budget and write Otto's kickoff."""
    return {
        name: mocker.patch.object(interview, name, new=AsyncMock())
        for name in (
            "get_or_create_onboarding_interview_session",
            "enforce_onboarding_interview_allowance",
            "kick_off_onboarding_interview",
        )
    }


LIVEKIT_URL = "wss://agpt-test.livekit.cloud"


@pytest.fixture
def voice_offer(mocker: MockerFixture) -> AsyncMock:
    """Whether voice can be offered here: not, unless a test says so. Where
    it can, the call's LiveKit project is ``LIVEKIT_URL``."""
    mocker.patch.object(routes.voice, "livekit_url", return_value=LIVEKIT_URL)
    return mocker.patch.object(
        routes.voice, "voice_available", new=AsyncMock(return_value=False)
    )


@pytest.fixture
def background_hand_off(mocker: MockerFixture) -> tuple[MagicMock, MagicMock]:
    """The hand-off an interview that is over gets behind the request, and
    the spawn that runs it."""
    hand_off = mocker.patch.object(
        interview, "hand_off_closed_interview", new=MagicMock(return_value="hand-off")
    )
    spawn = mocker.patch.object(interview, "spawn_background_task")
    return hand_off, spawn


def _stored_interview(mocker: MockerFixture, session: ChatSession | None) -> AsyncMock:
    return mocker.patch.object(
        interview,
        "find_onboarding_interview_session",
        new=AsyncMock(return_value=session),
    )


def test_reading_a_finished_interview_answers_it_over_and_starts_nothing(
    mocker: MockerFixture,
    test_user_id: str,
    interview_starts: dict[str, AsyncMock],
    background_hand_off: tuple[MagicMock, MagicMock],
    voice_offer: AsyncMock,
):
    """The talk-or-type choice learns the interview is over, so it offers
    no Skip, without the start's kickoff row or budget charge."""
    found = _stored_interview(
        mocker, _interview(test_user_id, stage=INTERVIEW_STAGE_HANDED_OFF)
    )

    response = client.get(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": INTERVIEW_SESSION_ID,
        "stage": INTERVIEW_STAGE_HANDED_OFF,
        "is_over": True,
        "voice_available": False,
        "livekit_url": None,
    }
    found.assert_awaited_once_with(test_user_id)
    for start in interview_starts.values():
        start.assert_not_awaited()
    # Handed off already: nothing left to settle.
    background_hand_off[1].assert_not_called()


def test_reading_with_no_interview_answers_none_open_and_creates_none(
    mocker: MockerFixture,
    interview_starts: dict[str, AsyncMock],
    background_hand_off: tuple[MagicMock, MagicMock],
    voice_offer: AsyncMock,
):
    """The step's choice reads, for someone who has not opened their
    interview yet, whether it may offer them a call."""
    _stored_interview(mocker, None)
    voice_offer.return_value = True

    response = client.get(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": None,
        "stage": None,
        "is_over": False,
        "voice_available": True,
        "livekit_url": LIVEKIT_URL,
    }
    for start in interview_starts.values():
        start.assert_not_awaited()
    background_hand_off[1].assert_not_called()


def test_reading_says_whether_voice_can_be_offered_to_the_person(
    mocker: MockerFixture,
    test_user_id: str,
    interview_starts: dict[str, AsyncMock],
    background_hand_off: tuple[MagicMock, MagicMock],
    voice_offer: AsyncMock,
):
    _stored_interview(mocker, _interview(test_user_id))

    voice_offer.return_value = True
    offered = client.get(INTERVIEW_URL).json()
    voice_offer.return_value = False
    not_offered = client.get(INTERVIEW_URL).json()

    assert (offered["voice_available"], not_offered["voice_available"]) == (
        True,
        False,
    )
    voice_offer.assert_awaited_with(test_user_id)


def test_reading_an_interview_over_by_its_cap_hands_it_off_as_the_start_does(
    mocker: MockerFixture,
    test_user_id: str,
    interview_starts: dict[str, AsyncMock],
    background_hand_off: tuple[MagicMock, MagicMock],
    voice_offer: AsyncMock,
):
    """Its replies are used up before its stage says so: it reads as over,
    and the hand-off the start would spawn runs behind the read."""
    session = _interview(test_user_id)
    session.messages = [
        message
        for n in range(INTERVIEW_TURN_CAP)
        for message in (
            ChatMessage(role="user", content=f"Answer {n}."),
            ChatMessage(role="assistant", content=f"Reply {n}."),
        )
    ]
    _stored_interview(mocker, session)
    hand_off, spawn = background_hand_off

    response = client.get(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": INTERVIEW_SESSION_ID,
        "stage": None,
        "is_over": True,
        "voice_available": False,
        "livekit_url": None,
    }
    hand_off.assert_called_once_with(test_user_id, INTERVIEW_SESSION_ID)
    assert spawn.call_args.args == ("hand-off",)
    for start in interview_starts.values():
        start.assert_not_awaited()


def test_reading_an_interview_under_way_answers_it_open(
    mocker: MockerFixture,
    test_user_id: str,
    interview_starts: dict[str, AsyncMock],
    background_hand_off: tuple[MagicMock, MagicMock],
    voice_offer: AsyncMock,
):
    _stored_interview(
        mocker, _interview(test_user_id, stage=INTERVIEW_STAGE_WRAP_UP_ISSUED)
    )

    response = client.get(INTERVIEW_URL)

    assert response.status_code == 200
    assert response.json() == {
        "session_id": INTERVIEW_SESSION_ID,
        "stage": INTERVIEW_STAGE_WRAP_UP_ISSUED,
        "is_over": False,
        "voice_available": False,
        "livekit_url": None,
    }
    background_hand_off[1].assert_not_called()


def test_finishing_hands_the_interview_off_and_answers_its_take(
    finalize_interview: AsyncMock, find_interview: AsyncMock, test_user_id: str
):
    finalize_interview.return_value = FinalizeResponse(
        status=BrainDumpStatus.transcribed,
        input_mode=BrainDumpInputMode.typed,
        transcript_preview="I run a bakery with five staff.",
    )

    response = client.post(INTERVIEW_FINISH_URL)

    assert response.status_code == 200
    assert response.json()["status"] == "transcribed"
    assert response.json()["input_mode"] == "typed"
    # The person's own interview, found from who they are, never from the
    # request; the pipeline's jobs run after the response.
    find_interview.assert_awaited_once_with(test_user_id)
    user_id, session_id, background_tasks = finalize_interview.await_args.args
    assert user_id == test_user_id
    assert session_id == INTERVIEW_SESSION_ID
    assert isinstance(background_tasks, fastapi.BackgroundTasks)


def test_finishing_with_no_interview_on_record_is_a_404(
    finalize_interview: AsyncMock, find_interview: AsyncMock
):
    find_interview.return_value = None

    response = client.post(INTERVIEW_FINISH_URL)

    assert response.status_code == 404
    finalize_interview.assert_not_awaited()


def test_finishing_an_interview_that_is_gone_is_a_404(
    finalize_interview: AsyncMock, find_interview: AsyncMock
):
    finalize_interview.side_effect = NotFoundError("Onboarding interview not found")

    response = client.post(INTERVIEW_FINISH_URL)

    assert response.status_code == 404


def test_a_finish_the_pipeline_s_budget_refuses_is_a_429_with_its_wait_in_the_body(
    finalize_interview: AsyncMock, find_interview: AsyncMock
):
    finalize_interview.side_effect = fastapi.HTTPException(
        status_code=429,
        detail="You have made several personalization attempts.",
        headers={"Retry-After": "600"},
    )

    response = client.post(INTERVIEW_FINISH_URL)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "600"
    assert response.json()["detail"] == {
        "message": "You have made several personalization attempts.",
        "retry_after_seconds": 600,
    }
