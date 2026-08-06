"""Request / response models for the onboarding brain dump."""

from typing import Annotated

from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import BaseModel, Field, StringConstraints

# ``recording_id`` is concatenated into a cloud-storage object key and a
# Redis key, so anything with a path or namespace separator in it would
# let a caller reach outside their own prefix. The client sends a
# ``crypto.randomUUID()``; this is the server refusing everything else.
RECORDING_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
RecordingId = Annotated[str, StringConstraints(pattern=RECORDING_ID_PATTERN)]

# A 3s MediaRecorder timeslice of webm/opus voice is ~10 KB; 2 MB per part
# leaves three orders of magnitude of headroom while still rejecting a
# client that tries to stream a movie through the endpoint.
MAX_PART_BYTES = 2 * 1024 * 1024

# ~30 min of webm/opus voice is ~6 MB. 50 MB is the cumulative ceiling for
# one recording — generous enough that no honest dump ever hits it.
MAX_RECORDING_BYTES = 50 * 1024 * 1024

# Typed fallback. Copilot's own message ceiling is 64,000 chars; the typed
# composer is a paragraph or two, so cap well below that.
MAX_TYPED_CHARS = 20_000

# The client stops recording at 30 minutes, and this number is not just
# metadata: an unmuxed MediaRecorder stream carries no container duration,
# so it becomes the loop bound in ``transcription.split_audio`` — one
# ffmpeg run and one billed STT call per 10 minutes of it. Four hours
# leaves room for clock skew and a paused-and-resumed take while keeping
# a hostile value from turning a 10 KB upload into six figures of both.
MAX_DURATION_SECS = 4 * 60 * 60

ALLOWED_AUDIO_MIME_TYPES = frozenset(
    {
        "audio/webm",
        "audio/mp4",
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
    }
)


class UploadPartResponse(BaseModel):
    recording_id: str
    part_index: int
    received_bytes: int
    total_bytes: int


class FinalizeRequest(BaseModel):
    """Finalize one take.

    ``text`` is the typed fallback: when set, no audio is expected and the
    dump is stored with ``input_mode=typed``. ``input_mode=skipped``
    records the deliberate skip so Path B can be chosen server-side.
    """

    recording_id: RecordingId
    input_mode: BrainDumpInputMode = BrainDumpInputMode.voice
    duration_secs: float | None = Field(default=None, ge=0, le=MAX_DURATION_SECS)
    mime_type: str | None = None
    text: str | None = Field(default=None, max_length=MAX_TYPED_CHARS)


class FinalizeResponse(BaseModel):
    status: BrainDumpStatus
    input_mode: BrainDumpInputMode
    transcript_preview: str | None = None
    error_code: str | None = None


class DumpStatusResponse(BaseModel):
    status: BrainDumpStatus | None = None
    input_mode: BrainDumpInputMode | None = None
    error_code: str | None = None
    has_audio: bool = False
    greeting_ready: bool = False
    """True once the copilot greeting is stored and ready to render — the
    onboarding loading screen holds the user until this flips (or the
    pipeline terminally fails)."""


class SuggestedPrompt(BaseModel):
    """One pickable prompt under the greeting.

    ``title`` is the short line the user sees; ``prompt`` is the full
    message that gets sent when they pick it. ``icon`` is a Phosphor icon
    slug from ``intro.PROMPT_ICONS`` — the frontend maps it to the actual
    icon component and falls back to a sparkle for anything unknown.
    """

    title: str
    prompt: str
    icon: str = "sparkle"


class IntroCardResponse(BaseModel):
    """Content for the copilot home's onboarding greeting.

    ``path`` is "A" when we have something to reflect back and "B" when
    the greeting's job is to invite a recording instead. ``greeting_done``
    is the server-side "already seen it" flag — when true the client
    shows nothing and caches the fact locally.
    """

    path: str
    greeting: str
    prompts: list[SuggestedPrompt] = []
    greeting_done: bool = False
    transcript: str | None = None
    """The full transcript of the recorded dump, so the greeting page can
    offer a copy button. Only present on Path A while the greeting is
    still showing."""


class RecommendedProvider(BaseModel):
    """One provider Claude picked from the transcript.

    ``provider`` is a registry id (validated against the live provider
    list before storage); ``reason`` is the one-liner shown under it in
    the welcome dialog's Recommended section.
    """

    provider: str
    reason: str = ""


class RecommendedProvidersResponse(BaseModel):
    """Recommendations for the "Connect your tools" panel.

    ``ready`` is false while the background job is still running — the
    client keeps polling. Once true, ``providers`` is the final answer,
    empty included ("nothing worth recommending" is a real result).
    """

    ready: bool
    providers: list[RecommendedProvider] = []


class GreetingDoneResponse(BaseModel):
    greeting_done: bool


class RecordingUrlResponse(BaseModel):
    url: str
    mime_type: str | None = None
