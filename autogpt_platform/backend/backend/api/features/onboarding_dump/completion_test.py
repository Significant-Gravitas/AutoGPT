"""Completed transcripts reach PostHog without affecting onboarding."""

from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import completion, db
from backend.util import posthog_client

USER_ID = "user-1"
RECORDING_ID = "recording-1"
TRANSCRIPT = "I run a bakery and need help with weekly orders."


@pytest.fixture
def dump() -> OnboardingBrainDump:
    return OnboardingBrainDump.model_construct(
        id="dump-1",
        userId=USER_ID,
        recordingId=RECORDING_ID,
        status=BrainDumpStatus.extracting,
        inputMode=BrainDumpInputMode.voice,
        transcript=TRANSCRIPT,
        transcriptLang="en",
        durationSecs=66.25,
    )


@pytest.fixture
def persistence(mocker: MockerFixture, dump: OnboardingBrainDump):
    read = mocker.patch.object(db, "get_dump", AsyncMock(return_value=dump))
    claim = mocker.patch.object(db, "claim_transition", AsyncMock(return_value=True))
    return read, claim


@pytest.fixture
def client(mocker: MockerFixture):
    client = MagicMock()
    mocker.patch.object(posthog_client, "get_posthog_client", return_value=client)
    return client


@pytest.mark.asyncio
async def test_completed_voice_is_persisted_before_the_full_event(
    dump: OnboardingBrainDump, persistence, client: MagicMock
):
    _, claim = persistence
    dump.transcript = TRANSCRIPT * 1000

    persistence_counts = []

    def captured(**event):
        persistence_counts.append(claim.await_count)

    client.capture.side_effect = captured
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_called_once()
    assert persistence_counts == [1]
    claim.assert_awaited_once()
    event = client.capture.call_args.kwargs
    properties = event["properties"]
    assert event["event"] == "brain_dump_transcribed"
    assert event["distinct_id"] == USER_ID
    assert properties["transcript"] == dump.transcript
    assert properties["brain_dump_id"] == "dump-1"
    assert properties["recording_id"] == RECORDING_ID
    assert properties["user_id"] == USER_ID
    assert properties["input_mode"] == "voice"
    assert properties["duration_seconds"] == 66.25
    assert properties["transcript_language"] == "en"
    completed_at = claim.await_args.kwargs["updatedAt"]
    assert completed_at.tzinfo == timezone.utc
    assert event["timestamp"] == completed_at
    assert properties["completed_at"] == completed_at.isoformat()
    assert properties["transcript_chars"] == len(dump.transcript)
    assert properties["transcript_truncated"] is False
    assert properties["source"] == "platform"
    assert "email" not in properties
    assert "audioPath" not in properties
    assert claim.await_args.args == (USER_ID, RECORDING_ID)
    assert claim.await_args.kwargs["expected"] == BrainDumpStatus.extracting
    assert claim.await_args.kwargs["new"] == BrainDumpStatus.completed


@pytest.mark.asyncio
async def test_typed_completion_has_no_invented_duration_or_language(
    dump: OnboardingBrainDump, persistence, client: MagicMock
):
    dump.inputMode = BrainDumpInputMode.typed
    dump.transcript = "a" * 20_000
    dump.durationSecs = None
    dump.transcriptLang = None

    await completion.complete_dump(USER_ID, RECORDING_ID)

    properties = client.capture.call_args.kwargs["properties"]
    assert properties["input_mode"] == "typed"
    assert properties["transcript"] == dump.transcript
    assert properties["duration_seconds"] is None
    assert properties["transcript_language"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", [None, "", "  \n"])
async def test_empty_transcripts_are_not_emitted(
    dump: OnboardingBrainDump, persistence, client: MagicMock, transcript: str | None
):
    dump.transcript = transcript
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_a_skipped_dump_is_not_emitted(
    dump: OnboardingBrainDump, persistence, client: MagicMock
):
    dump.inputMode = BrainDumpInputMode.skipped
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_a_lost_completion_claim_does_not_emit(persistence, client: MagicMock):
    _, claim = persistence
    claim.return_value = False
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_completion_is_only_emitted_by_the_winner(
    persistence, client: MagicMock
):
    _, claim = persistence
    claim.side_effect = [True, False]
    await completion.complete_dump(USER_ID, RECORDING_ID)
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["userId", "recordingId"])
async def test_a_different_takes_snapshot_is_not_emitted(
    dump: OnboardingBrainDump, persistence, client: MagicMock, field: str
):
    setattr(dump, field, "another-id")
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_read_failure_does_not_prevent_completion(
    persistence, client: MagicMock, caplog: pytest.LogCaptureFixture
):
    read, claim = persistence
    read.side_effect = RuntimeError("secret transcript in exception")
    await completion.complete_dump(USER_ID, RECORDING_ID)
    claim.assert_awaited_once()
    client.capture.assert_not_called()
    assert "secret transcript" not in caplog.text


@pytest.mark.asyncio
async def test_missing_snapshot_does_not_prevent_completion(
    persistence, client: MagicMock
):
    read, claim = persistence
    read.return_value = None
    await completion.complete_dump(USER_ID, RECORDING_ID)
    claim.assert_awaited_once()
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_persistence_failure_is_not_mistaken_for_completion(
    persistence, client: MagicMock
):
    _, claim = persistence
    claim.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_not_called()


@pytest.mark.asyncio
async def test_capture_failure_does_not_escape_or_log_transcript(
    persistence, client: MagicMock, caplog: pytest.LogCaptureFixture
):
    client.capture.side_effect = RuntimeError(TRANSCRIPT)
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_called_once()
    assert TRANSCRIPT not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_disabled_or_broken_analytics_cannot_prevent_completion(
    mocker: MockerFixture, persistence, raises: bool
):
    client = mocker.patch.object(
        posthog_client, "get_posthog_client", return_value=None
    )
    if raises:
        client.side_effect = RuntimeError("analytics configuration unavailable")
    await completion.complete_dump(USER_ID, RECORDING_ID)
    _, claim = persistence
    claim.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_identity_is_per_user_and_recording_not_row(
    dump: OnboardingBrainDump, persistence, client: MagicMock
):
    await completion.complete_dump(USER_ID, RECORDING_ID)
    first = client.capture.call_args.kwargs
    await completion.complete_dump(USER_ID, RECORDING_ID)
    repeated = client.capture.call_args.kwargs
    assert first["uuid"] == repeated["uuid"]
    assert first["properties"]["$insert_id"] == repeated["properties"]["$insert_id"]

    dump.recordingId = "recording-2"
    await completion.complete_dump(USER_ID, "recording-2")
    assert client.capture.call_args.kwargs["uuid"] != first["uuid"]

    dump.userId = "user-2"
    dump.recordingId = RECORDING_ID
    await completion.complete_dump("user-2", RECORDING_ID)
    assert client.capture.call_args.kwargs["uuid"] != first["uuid"]


@pytest.mark.asyncio
async def test_transcripts_are_not_added_to_sentry_breadcrumbs(
    mocker: MockerFixture, persistence, client: MagicMock
):
    breadcrumb = mocker.patch("sentry_sdk.add_breadcrumb")
    await completion.complete_dump(USER_ID, RECORDING_ID)
    client.capture.assert_called_once()
    breadcrumb.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_transcript_reports_metadata_without_truncation_or_raw_text(
    dump: OnboardingBrainDump,
    persistence,
    client: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    dump.transcript = "\U0001f9c1" * 100_000
    await completion.complete_dump(USER_ID, RECORDING_ID)

    event = client.capture.call_args.kwargs
    assert event["event"] == "brain_dump_transcript_export_failed"
    assert "transcript" not in event["properties"]
    assert event["properties"]["transcript_chars"] == 100_000
    assert event["properties"]["error_code"] == "transcript_too_large"
    assert (
        event["properties"]["transcript_json_bytes"]
        > event["properties"]["limit_bytes"]
    )
    assert "\U0001f9c1" not in caplog.text
