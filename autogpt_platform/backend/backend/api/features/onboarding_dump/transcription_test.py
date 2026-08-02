"""Unit tests for the speech-to-text helpers. No network, no ffmpeg."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import transcription


def test_stitch_drops_the_duplicated_overlap_at_the_seam():
    stitched = transcription.stitch_transcripts(
        [
            "so the first thing I want to automate is the weekly order email",
            "the weekly order email goes out every monday morning",
        ]
    )

    assert stitched == (
        "so the first thing I want to automate is the weekly order email "
        "goes out every monday morning"
    )


def test_stitch_matches_the_seam_case_insensitively():
    stitched = transcription.stitch_transcripts(
        ["we ship on friday", "Friday is when the boxes go out"]
    )

    assert stitched == "we ship on friday is when the boxes go out"


def test_stitch_joins_segments_that_do_not_overlap():
    stitched = transcription.stitch_transcripts(
        ["alpha beta gamma", "delta epsilon zeta"]
    )

    assert stitched == "alpha beta gamma delta epsilon zeta"


def test_stitch_keeps_a_phrase_that_legitimately_repeats_outside_the_seam():
    stitched = transcription.stitch_transcripts(
        [
            "we ship on friday we ship on friday every single week",
            "every single week without fail",
        ]
    )

    assert stitched == (
        "we ship on friday we ship on friday every single week without fail"
    )


def test_stitch_does_not_collapse_a_repeat_longer_than_the_overlap_window():
    phrase = [f"word{index}" for index in range(transcription.MAX_OVERLAP_WORDS + 5)]
    stitched = transcription.stitch_transcripts([" ".join(phrase), " ".join(phrase)])

    # The seam search never looks further back than MAX_OVERLAP_WORDS, so a
    # repetition longer than the window is left intact rather than being
    # mistaken for a duplicated overlap and silently deleted.
    assert stitched.split() == phrase + phrase


def test_stitch_ignores_empty_segments():
    assert transcription.stitch_transcripts(["", "hello there", "  "]) == "hello there"


@pytest.mark.asyncio
async def test_transcribe_falls_back_to_whisper_after_the_primary_exhausts_retries(
    mocker: MockerFixture,
):
    mocker.patch.object(transcription.asyncio, "sleep", new=AsyncMock())
    attempted_models: list[str] = []

    async def create(*, model: str, file: tuple[str, bytes]):
        attempted_models.append(model)
        if model == transcription.PRIMARY_MODEL:
            raise RuntimeError("primary model is down")
        return MagicMock(text="hello there", language="en")

    stt = MagicMock()
    stt.audio.transcriptions.create = AsyncMock(side_effect=create)
    mocker.patch.object(transcription, "get_stt_client", return_value=stt)

    transcript, language = await transcription.transcribe(b"opus", "dump.webm")

    assert (transcript, language) == ("hello there", "en")
    assert attempted_models == (
        [transcription.PRIMARY_MODEL] * transcription.MAX_ATTEMPTS
        + [transcription.FALLBACK_MODEL]
    )


@pytest.mark.asyncio
async def test_transcribe_raises_once_every_model_and_retry_is_exhausted(
    mocker: MockerFixture,
):
    mocker.patch.object(transcription.asyncio, "sleep", new=AsyncMock())
    create = AsyncMock(side_effect=RuntimeError("everything is down"))
    stt = MagicMock()
    stt.audio.transcriptions.create = create
    mocker.patch.object(transcription, "get_stt_client", return_value=stt)

    with pytest.raises(transcription.TranscriptionFailedError):
        await transcription.transcribe(b"opus", "dump.webm")

    assert create.await_count == 2 * transcription.MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_transcribe_without_a_configured_provider_is_unavailable(
    mocker: MockerFixture,
):
    mocker.patch.object(transcription, "get_stt_client", return_value=None)

    with pytest.raises(transcription.TranscriptionUnavailableError):
        await transcription.transcribe(b"opus", "dump.webm")


@pytest.mark.asyncio
async def test_a_long_recording_is_split_and_stitched(mocker: MockerFixture):
    create = AsyncMock(
        side_effect=[
            MagicMock(text="first half of the dump", language="en"),
            MagicMock(text="of the dump and then the rest", language="en"),
        ]
    )
    stt = MagicMock()
    stt.audio.transcriptions.create = create
    mocker.patch.object(transcription, "get_stt_client", return_value=stt)
    split = mocker.patch.object(
        transcription, "split_audio", new=AsyncMock(return_value=[b"one", b"two"])
    )

    transcript, language = await transcription.transcribe(
        b"opus", "dump.webm", duration_secs=transcription.SINGLE_REQUEST_MAX_SECONDS + 1
    )

    split.assert_awaited_once()
    assert transcript == "first half of the dump and then the rest"
    # Language is unreliable across segments, so it is not reported.
    assert language is None
