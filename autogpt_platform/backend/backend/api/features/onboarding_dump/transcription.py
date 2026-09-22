"""Speech-to-text for the onboarding brain dump.

``gpt-4o-transcribe`` is the primary model: on messy, rambling natural
speech it is markedly more accurate and hallucinates far less than
``whisper-1``, which is the failure mode that matters most here (a
hallucinated sentence becomes a *fact* about the user). ``whisper-1`` is
the automatic fallback so a model outage degrades rather than fails.

Both models take a 25 MB request and roughly 25 minutes of audio, so a
long dump is split with ffmpeg into overlapping segments and the
transcripts stitched back together. ffmpeg is present in the backend
image (``backend/Dockerfile``); if it ever isn't, the split path raises
and the caller records a ``failed`` status with the audio still stored.
"""

import asyncio
import logging
import math
import os
import shutil
import string
import tempfile

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

PRIMARY_MODEL = os.environ.get("BRAIN_DUMP_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")
FALLBACK_MODEL = os.environ.get("BRAIN_DUMP_TRANSCRIPTION_FALLBACK_MODEL", "whisper-1")

# Both caps are the provider's, minus headroom for the multipart envelope.
SINGLE_REQUEST_MAX_BYTES = 20 * 1024 * 1024
SINGLE_REQUEST_MAX_SECONDS = 20 * 60

SEGMENT_SECONDS = 10 * 60
# ``split_audio`` re-encodes to opus in an ogg container, so a segment must
# not inherit the source extension: the STT client infers the format from
# the filename, and an ogg body announced as ``.webm`` is rejected.
SEGMENT_SUFFIX = ".ogg"
# Cutting mid-word loses the word on both sides of the seam; a few seconds
# of overlap means the word survives in at least one segment and the
# stitcher drops the duplicate.
SEGMENT_OVERLAP_SECONDS = 5
# ``-ss`` past the end of the stream is not an error: ffmpeg exits 0 and
# writes a container with nothing in it. Any real segment is orders of
# magnitude bigger than an empty ogg/opus header, so this is how the
# segment loop finds the true end of an audio whose claimed duration is
# longer than what is actually there.
MIN_SEGMENT_BYTES = 1024

TRANSCRIBE_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 3


class TranscriptionUnavailableError(RuntimeError):
    """No STT provider is configured."""


class TranscriptionFailedError(RuntimeError):
    """Every model and retry was exhausted."""


class TranscriptionResult(BaseModel):
    """One transcription, plus which model actually produced it.

    ``model`` matters because the fallback is silent: a run that thinks it
    measured ``gpt-4o-transcribe`` may be reading ``whisper-1``'s output.
    For a stitched transcript it lists every model that contributed.
    """

    text: str
    language: str | None = None
    model: str


def get_stt_client() -> AsyncOpenAI | None:
    """Return a client for the audio-transcriptions endpoint.

    Deliberately not ``backend.util.clients.get_openai_client()``: that
    helper falls back to OpenRouter, which does not implement
    ``/audio/transcriptions``, so a deployment with only an OpenRouter key
    would get a confusing 404 from the provider instead of a clean
    "not configured" error here.

    Built per call rather than ``@cached``: ``AsyncOpenAI`` binds its
    connection pool to the first event loop that uses it, so a process-wide
    cache poisons itself across loops. ``backend/util/architecture_test.py``
    enforces this — the cached helpers in ``util/clients.py`` are a
    grandfathered allowlist that is being burned down, not the pattern to
    copy. Finalize runs once per user, so the pool churn is negligible.
    """
    api_key = (
        settings.secrets.openai_internal_api_key or settings.secrets.openai_api_key
    )
    if not api_key:
        return None
    return AsyncOpenAI(api_key=api_key)


async def transcribe(
    audio: bytes,
    filename: str,
    duration_secs: float | None = None,
) -> TranscriptionResult:
    """Transcribe ``audio``.

    ``language`` is never passed to the provider — a non-English dump must
    transcribe in the language it was spoken in, never error or silently
    translate. It is also not reported for a stitched transcript, where
    the per-segment answers can disagree.
    """
    client = get_stt_client()
    if client is None:
        raise TranscriptionUnavailableError(
            "Brain-dump transcription needs a direct OpenAI key. Set "
            "OPENAI_INTERNAL_API_KEY or OPENAI_API_KEY."
        )

    needs_split = len(audio) > SINGLE_REQUEST_MAX_BYTES or (
        duration_secs is not None and duration_secs > SINGLE_REQUEST_MAX_SECONDS
    )
    if not needs_split:
        return await _transcribe_one(client, audio, filename)

    segments = await split_audio(audio, filename, duration_secs)
    logger.info("Brain dump split into %s segments for transcription", len(segments))
    results = [
        await _transcribe_one(client, segment, f"segment-{index}{SEGMENT_SUFFIX}")
        for index, segment in enumerate(segments)
    ]
    return TranscriptionResult(
        text=stitch_transcripts([result.text for result in results]),
        model=",".join(dict.fromkeys(result.model for result in results)),
    )


async def _transcribe_one(
    client: AsyncOpenAI, audio: bytes, filename: str
) -> TranscriptionResult:
    last_error: Exception | None = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await asyncio.wait_for(
                    client.audio.transcriptions.create(
                        model=model,
                        file=(filename, audio),
                    ),
                    timeout=TRANSCRIBE_TIMEOUT_SECONDS,
                )
                language = getattr(response, "language", None)
                return TranscriptionResult(
                    text=response.text,
                    language=language if isinstance(language, str) else None,
                    model=model,
                )
            except Exception as e:  # retried across models below
                last_error = e
                logger.warning(
                    "Brain dump transcription failed (model=%s attempt=%s): %s",
                    model,
                    attempt + 1,
                    e,
                )
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2**attempt)
    raise TranscriptionFailedError(str(last_error))


async def split_audio(
    audio: bytes, filename: str, duration_hint: float | None = None
) -> list[bytes]:
    """Cut ``audio`` into overlapping segments with ffmpeg.

    Byte-slicing an opus stream would produce unplayable fragments, so
    this is a real container-aware re-cut. Segments are re-encoded to
    16 kHz mono opus, which is what the STT models downsample to anyway
    and keeps every segment far inside the request cap.

    ``duration_hint`` is only ever an upper bound: it usually comes from
    the browser's wall clock, so the loop stops at the first empty segment
    rather than trusting it to the second.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise TranscriptionFailedError(
            "ffmpeg is required to transcribe recordings over "
            f"{SINGLE_REQUEST_MAX_SECONDS // 60} minutes but is not installed"
        )

    suffix = os.path.splitext(filename)[1] or ".webm"
    with tempfile.TemporaryDirectory() as workdir:
        source = os.path.join(workdir, f"source{suffix}")
        # Off the event loop: these are multi-megabyte reads and writes,
        # and this runs as a background task alongside live requests.
        await asyncio.to_thread(_write_file, source, audio)

        # A browser MediaRecorder stream is not muxed, so its header
        # usually carries no duration at all — for precisely the long
        # takes that land here. The browser timed the take with a wall
        # clock, so its figure is the fallback rather than a failure.
        duration = await _probe_duration(ffmpeg, source) or duration_hint
        if duration is None or duration <= 0:
            raise TranscriptionFailedError(
                f"Could not determine the duration of {os.path.basename(filename)}"
            )
        segments: list[bytes] = []
        start = 0.0
        index = 0
        while start < duration:
            target = os.path.join(workdir, f"segment-{index}{SEGMENT_SUFFIX}")
            await _run_ffmpeg(
                ffmpeg,
                "-ss",
                str(start),
                "-t",
                str(SEGMENT_SECONDS + SEGMENT_OVERLAP_SECONDS),
                "-i",
                source,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libopus",
                target,
            )
            segment = await asyncio.to_thread(_read_file, target)
            if len(segment) < MIN_SEGMENT_BYTES:
                # Past the real end of the audio. Everything from here on
                # would be another empty container, another ffmpeg run and
                # another billed transcription of silence.
                break
            segments.append(segment)
            start += SEGMENT_SECONDS
            index += 1
        return segments


async def _probe_duration(ffmpeg: str, path: str) -> float | None:
    """Read the container duration via ffmpeg's stderr banner.

    ffprobe is not guaranteed to be installed alongside ffmpeg, so this
    parses the ``Duration: HH:MM:SS.ms`` line instead of shelling out to a
    second binary.

    Best-effort by design: ``None`` means "ffmpeg would not tell us",
    which is the normal answer for an unmuxed stream, and the caller
    falls back to the duration the browser measured.
    """
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-i",
        path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    for line in stderr.decode(errors="replace").splitlines():
        marker = "Duration:"
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].split(",", 1)[0].strip()
        # ffmpeg writes the sign on the hours field, and ``int("-00")`` is
        # ``0`` — so a negative duration would otherwise parse as a
        # positive one of the same magnitude.
        if raw.startswith("-"):
            return None
        try:
            hours, minutes, seconds = raw.split(":")
            probed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except ValueError:
            # ``Duration: N/A``, which is what an unmuxed MediaRecorder
            # stream reports.
            return None
        # A damaged header can also yield zero or a non-finite value,
        # which would make the segment loop produce nothing or spin.
        if not math.isfinite(probed) or probed <= 0:
            return None
        return probed
    return None


def _write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


async def _run_ffmpeg(ffmpeg: str, *args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise TranscriptionFailedError(
            f"ffmpeg failed: {stderr.decode(errors='replace')[-500:]}"
        )


def stitch_transcripts(transcripts: list[str]) -> str:
    """Join segment transcripts, dropping the duplicated overlap words.

    The overlap is a few seconds of speech, so the seam is found by
    looking for the longest word-sequence that ends the previous segment
    and starts the next one.
    """
    stitched: list[str] = []
    for transcript in transcripts:
        words = transcript.split()
        if not words:
            continue
        if not stitched:
            stitched = words
            continue
        overlap = _overlap_length(stitched, words)
        stitched.extend(words[overlap:])
    return " ".join(stitched)


# A 5s overlap is at most ~25 spoken words; searching beyond that risks
# collapsing a genuinely repeated phrase.
MAX_OVERLAP_WORDS = 40


def _overlap_length(left: list[str], right: list[str]) -> int:
    limit = min(MAX_OVERLAP_WORDS, len(left), len(right))
    normalised_left = [_seam_word(w) for w in left[-limit:]] if limit else []
    normalised_right = [_seam_word(w) for w in right[:limit]] if limit else []
    for size in range(limit, 0, -1):
        if normalised_left[-size:] == normalised_right[:size]:
            return size
    return 0


def _seam_word(word: str) -> str:
    """Fold a word to what the seam comparison should care about.

    STT models punctuate each segment as if it were a whole utterance, so
    the same spoken word comes back as ``schedule.`` at the end of one
    segment and ``schedule`` at the start of the next. Comparing them raw
    finds no overlap at all and the whole 5s overlap is duplicated into
    the stitched transcript at every seam.
    """
    return word.lower().strip(string.punctuation)
