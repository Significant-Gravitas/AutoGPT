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


def test_stitch_matches_the_seam_across_the_terminal_punctuation():
    """STT models end every segment as if it were a whole utterance.

    The same spoken word therefore comes back as ``schedule.`` on the left
    of the seam and ``schedule`` on the right. Comparing raw words finds
    no overlap at all, and the deliberate 5s overlap is duplicated into
    the transcript at every 10-minute seam.
    """
    stitched = transcription.stitch_transcripts(
        [
            "and then I have to go and rebuild the whole schedule.",
            "Rebuild the whole schedule, and that eats my monday.",
        ]
    )

    assert stitched == (
        "and then I have to go and rebuild the whole schedule. "
        "and that eats my monday."
    )


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

    result = await transcription.transcribe(b"opus", "dump.webm")

    assert (result.text, result.language) == ("hello there", "en")
    # The fallback is silent, so the result has to say which model won.
    assert result.model == transcription.FALLBACK_MODEL
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

    result = await transcription.transcribe(
        b"opus", "dump.webm", duration_secs=transcription.SINGLE_REQUEST_MAX_SECONDS + 1
    )

    split.assert_awaited_once()
    assert result.text == "first half of the dump and then the rest"
    # Language is unreliable across segments, so it is not reported.
    assert result.language is None
    assert result.model == transcription.PRIMARY_MODEL
    # Segments come back as ogg regardless of what the browser recorded, and
    # the client infers the format from the filename — announcing an ogg body
    # as ``.webm`` makes the provider reject it.
    sent = [call.kwargs["file"][0] for call in create.await_args_list]
    assert sent == ["segment-0.ogg", "segment-1.ogg"]


def _ffmpeg_banner(duration_line: str) -> bytes:
    return (
        b"ffmpeg version 6.0\n"
        b"  Input #0, matroska,webm, from 'source.webm':\n"
        + duration_line.encode()
        + b"\n  Stream #0:0: Audio: opus\n"
    )


@pytest.mark.parametrize(
    "duration_line",
    [
        # What an unmuxed MediaRecorder stream actually reports — the
        # normal case for the long takes that reach split_audio.
        "  Duration: N/A, start: 0.000000, bitrate: N/A",
        # A damaged header can produce values that would make the segment
        # loop spin forever or emit nothing.
        "  Duration: -00:00:01.00, start: 0.000000, bitrate: 32 kb/s",
        "  Duration: 00:00:00.00, start: 0.000000, bitrate: 32 kb/s",
    ],
)
@pytest.mark.asyncio
async def test_an_unusable_ffmpeg_duration_is_reported_as_unknown(
    mocker: MockerFixture, duration_line: str
):
    process = MagicMock()
    process.communicate = AsyncMock(return_value=(b"", _ffmpeg_banner(duration_line)))
    mocker.patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process))

    assert await transcription._probe_duration("ffmpeg", "source.webm") is None


@pytest.mark.asyncio
async def test_a_readable_ffmpeg_duration_is_parsed(mocker: MockerFixture):
    process = MagicMock()
    process.communicate = AsyncMock(
        return_value=(
            b"",
            _ffmpeg_banner(
                "  Duration: 00:21:30.50, start: 0.000000, bitrate: 32 kb/s"
            ),
        )
    )
    mocker.patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process))

    assert await transcription._probe_duration("ffmpeg", "source.webm") == 1290.5


@pytest.mark.asyncio
async def test_split_falls_back_to_the_browser_duration_when_ffmpeg_cannot_tell(
    mocker: MockerFixture,
):
    """The take must still be split when the container has no duration.

    ffmpeg reporting ``N/A`` is the norm for a browser MediaRecorder
    stream, so treating it as fatal would fail exactly the long
    recordings this code path exists to handle.
    """
    mocker.patch.object(transcription.shutil, "which", return_value="ffmpeg")
    mocker.patch.object(
        transcription, "_probe_duration", new=AsyncMock(return_value=None)
    )
    run = mocker.patch.object(transcription, "_run_ffmpeg", new=AsyncMock())
    mocker.patch.object(transcription, "_write_file", new=MagicMock())
    mocker.patch.object(
        transcription,
        "_read_file",
        new=MagicMock(return_value=b"s" * transcription.MIN_SEGMENT_BYTES),
    )

    hint = transcription.SEGMENT_SECONDS * 2 + 5
    segments = await transcription.split_audio(b"opus", "dump.webm", hint)

    assert len(segments) == 3
    assert run.await_count == 3


@pytest.mark.asyncio
async def test_split_stops_at_the_first_empty_segment(mocker: MockerFixture):
    """The duration hint is the client's number, and it can be a lie.

    ``-ss`` past the end of the stream is not an error — ffmpeg exits 0
    and writes an empty container — so a hostile ``duration_secs`` would
    otherwise buy one ffmpeg run and one billed transcription per 10
    minutes of it, however little audio was actually uploaded.
    """
    mocker.patch.object(transcription.shutil, "which", return_value="ffmpeg")
    mocker.patch.object(
        transcription, "_probe_duration", new=AsyncMock(return_value=None)
    )
    run = mocker.patch.object(transcription, "_run_ffmpeg", new=AsyncMock())
    mocker.patch.object(transcription, "_write_file", new=MagicMock())
    real = b"a" * transcription.MIN_SEGMENT_BYTES
    mocker.patch.object(
        transcription, "_read_file", new=MagicMock(side_effect=[real, b"OggS-header"])
    )

    segments = await transcription.split_audio(
        b"opus", "dump.webm", transcription.SEGMENT_SECONDS * 10_000
    )

    assert segments == [real]
    assert run.await_count == 2


@pytest.mark.asyncio
async def test_split_fails_cleanly_when_no_duration_is_available(
    mocker: MockerFixture,
):
    mocker.patch.object(transcription.shutil, "which", return_value="ffmpeg")
    mocker.patch.object(
        transcription, "_probe_duration", new=AsyncMock(return_value=None)
    )
    mocker.patch.object(transcription, "_write_file", new=MagicMock())

    with pytest.raises(transcription.TranscriptionFailedError):
        await transcription.split_audio(b"opus", "dump.webm", None)
