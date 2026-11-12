"""Request / response models for the onboarding brain dump."""

from datetime import datetime
from typing import Annotated, Literal

from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import AfterValidator, BaseModel, Field, StringConstraints

# ``recording_id`` is concatenated into a cloud-storage object key and a
# Redis key, so anything with a path or namespace separator in it would
# let a caller reach outside their own prefix. The client sends a
# ``crypto.randomUUID()``; this is the server refusing everything else.
RECORDING_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# The takes the server makes itself, an interview's and a voice call's, are
# named this and the interview's id. A client never names a take so.
INTERVIEW_RECORDING_PREFIX = "interview-"


def clients_recording_id(recording_id: str) -> str:
    """*recording_id*, when it is a take a client may name: none named as
    the server names its own takes."""
    if recording_id.startswith(INTERVIEW_RECORDING_PREFIX):
        raise ValueError("recording_id names a take only the server makes")
    return recording_id


RecordingId = Annotated[
    str,
    StringConstraints(pattern=RECORDING_ID_PATTERN),
    AfterValidator(clients_recording_id),
]

# A 3s MediaRecorder timeslice of webm/opus voice is ~10 KB; 2 MB per part
# leaves three orders of magnitude of headroom while still rejecting a
# client that tries to stream a movie through the endpoint.
MAX_PART_BYTES = 2 * 1024 * 1024

# The kinds of too-short take Otto has a fixed first line for on the
# not-enough screen (``otto_reply`` picks one; the lines are in the
# frontend's ``insufficient-copy.ts``).
OttoReplyKind = Literal[
    "mic_check",
    "greeting",
    "unsure",
    "reluctant",
    "what_to_say",
    "wants_to_skip",
    "about_otto",
    "busy",
    "annoyed",
    "some_work",
    "garbled",
    "test",
    "other",
]

# ~30 min of webm/opus voice is ~6 MB. 50 MB is the cumulative ceiling for
# one recording — generous enough that no honest dump ever hits it.
MAX_RECORDING_BYTES = 50 * 1024 * 1024

# Typed fallback. Copilot's own message ceiling is 64,000 chars; the typed
# composer is a paragraph or two, so cap well below that.
MAX_TYPED_CHARS = 20_000

# What the person may add on an interview's ending screen that ends on a
# confirm.
MAX_CORRECTION_CHARS = 2_000

# A voice call's id, as the start route mints it (``voice_relay.tokens``).
VOICE_CALL_ID_PATTERN = r"^[0-9a-f]{32}$"

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

# Stored recordings are named ``brain-dump-{recording_id}.{ext}``. The
# retention sweep finds them in the bucket by this name, so the two must
# stay in step.
RECORDING_FILENAME_PREFIX = "brain-dump-"
AUDIO_FILE_EXTENSIONS = {
    "audio/webm": "webm",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
}


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
    """The transcript's start, on success and on an ``insufficient_content``
    reject; never on ``no_usable_speech``."""
    otto_reply_kind: OttoReplyKind | None = None
    """Which of Otto's fixed first lines fits what was said, on an
    ``insufficient_content`` reject. ``None`` and ``"other"`` get the default."""
    error_code: str | None = None
    error_reason: str | None = None
    """What decided a quality-gate rejection (``quality.QualityRejection``)."""
    transcript_words: int | None = None
    """A rejected transcript's length as the gate counts words."""


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
    expert_template_id: str | None = None
    """The roster template whose expert takes this task; null means Otto
    does it himself. Always a member of the plan's team (``plan.cap_team``)."""
    why: str | None = None
    """One short clause naming what the user said this task fixes, at most
    ``MAX_WHY_CHARS``; null when they said nothing it ties to."""
    expert_name: str | None = None
    """The owner's name when the plan was written, so a task whose expert has
    since left the team can still say whose it was; null for Otto's."""


MAX_WHY_CHARS = 60

SuggestedTaskState = Literal["open", "sent", "dismissed"]


class SuggestedTask(SuggestedPrompt):
    """One row of the Home checklist: a suggested prompt and what became of it.

    Stored inside ``OnboardingBrainDump.suggestedPrompts``; a row written
    before states existed reads as ``open``, with its list index as ``id``,
    and one written before owners existed reads as Otto's.
    """

    id: str
    state: SuggestedTaskState = "open"
    session_id: str | None = None
    """The chat a ``sent`` row started."""
    updated_at: datetime | None = None
    expert_id: str | None = None
    """The user's active hire of ``expert_template_id``, filled at read time
    and never stored: null while that expert is not hired, once it is
    archived, and for Otto's tasks."""


class SuggestedTasksResponse(BaseModel):
    """The user's suggested first tasks, every state included.

    ``stored`` is false when the user has no list of their own and these
    are the generic starters: the greeting shows them, the regular Home
    page does not, until a state change stores them.
    """

    tasks: list[SuggestedTask] = []
    stored: bool = False


class SuggestedTasksUpdate(BaseModel):
    task_ids: list[str] = Field(min_length=1, max_length=20)
    state: SuggestedTaskState
    """``open`` is Undo: a hidden row goes back to where it was, which is
    ``sent`` for one that had started a chat."""
    session_id: str | None = Field(default=None, max_length=64)


class RecommendedExpert(BaseModel):
    """One roster template Otto suggests hiring first.

    ``template_id`` is always a live template — the model only ever picks
    ids off the roster it was shown, and anything else is dropped before
    storage. ``name``/``role``/``avatar_url`` are copied from that
    template server-side rather than trusted from the model, so a card
    can never introduce an expert who does not exist.
    """

    template_id: str
    name: str
    role: str
    avatar_url: str | None = None
    reason: str = ""
    workflow_names: list[str] = []


class RaiseSuggestion(BaseModel):
    """The "raise your own" door, prefilled for a gap the roster can't fill.

    ``role`` is one of the ``/raise`` wizard's role ids so the link can
    seed the draft; a suggestion naming anything else is dropped.
    """

    role: str
    reason: str = ""


class ExpertRecommendations(BaseModel):
    """The team Otto proposes after the dump.

    ``source`` says where it came from: "llm" for a real read of the
    transcript, "fallback" for the deterministic role/pain-point mapping,
    and "disabled" for a run where the feature flag was off (recorded so
    the reader can tell it apart from a job that never finished).
    """

    diagnosis: str = ""
    experts: list[RecommendedExpert] = []
    raise_suggestion: RaiseSuggestion | None = None
    source: Literal["llm", "fallback", "disabled"] = "fallback"


class RecommendedExpertsResponse(BaseModel):
    """The team section's content, polled by the onboarding loading screen.

    ``ready`` is false only while the background job is still running.
    ``team`` is null when the feature is off — a team with no experts is
    a real answer, so the client cannot use emptiness to tell them apart.
    """

    ready: bool
    team: ExpertRecommendations | None = None


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
    greeting_pending: bool = False
    """True while the greeting is still being written. Both this state and
    "there is no greeting" carry an empty ``greeting``, so the client needs
    this flag to tell them apart and know whether to keep polling."""
    transcript: str | None = None
    """The full transcript of the recorded dump, so the greeting page can
    offer a copy button. Only present on Path A while the greeting is
    still showing."""
    team: ExpertRecommendations | None = None
    """The experts to hire first, with the "raise your own" door beside
    them. Null when the feature is off, and while ``team_pending``."""
    team_pending: bool = False
    """True while the team is still being assembled — the client keeps
    polling ``/recommended-experts`` rather than rendering an empty
    section over a team that is seconds away."""


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


class InterviewStartResponse(BaseModel):
    """The person's onboarding interview, open on the onboarding step."""

    session_id: str
    """The interview's chat session; each person has one."""
    turn_id: str | None = None
    """Otto's first turn, when this call started it. None when there was
    nothing to start: the client then reads the session, which resumes any
    turn still running in it."""
    stage: str | None = None
    """``ChatSessionMetadata.onboarding_interview_stage``; None while Otto is
    still asking."""


class VoiceCallCaps(BaseModel):
    """A voice call's limits. The relay enforces them; the browser shows them."""

    max_call_s: float
    soft_wrap_s: float
    """Otto is told to sum up this long before ``max_call_s``."""
    max_agent_turns: int
    soft_wrap_turns: int
    """Otto is told to sum up this many turns before ``max_agent_turns``."""
    idle_s: float
    """The call ends after this long with nothing heard from the person."""


class VoiceCallLeg(BaseModel):
    """How the call's audio travels. The control socket always carries the
    call's events; on the ``websocket`` leg it carries the audio too."""

    kind: Literal["websocket"]


class VoiceCallStartResponse(BaseModel):
    """A voice call the person can now open on the relay."""

    call_id: str
    relay_url: str
    """The relay's call socket."""
    token: str
    """The socket's first message carries it, ``{"type": "start", "token"}``.
    Good for this one call, within ``expires_in`` seconds."""
    expires_in: int
    leg: VoiceCallLeg
    caps: VoiceCallCaps


# A voice call's status, in the order a call moves through them: opening,
# live, over, its transcript of record being made, made and waiting for the
# person's confirm, its hand-off running, and handed to the onboarding
# pipeline or failed. ``handing_off`` is never stored: the call routes report
# it for a call whose transcript is made and whose hand-off is owed.
VoiceCallStatus = Literal[
    "connecting",
    "live",
    "ended",
    "transcribing",
    "transcribed",
    "handing_off",
    "handed_off",
    "failed",
]

# Who owns the person's interview as far as a voice call goes: the call's
# words took it, or what the person typed after the call superseded them.
VoiceCallInterview = Literal["taken", "superseded"]


class VoiceCallStatusResponse(BaseModel):
    """A voice call as it stands."""

    call_id: str
    status: VoiceCallStatus
    """``connecting`` and ``live`` while the call runs; ``ended`` and
    ``transcribing`` while its transcript of record is made. Then
    ``handing_off`` while the onboarding pipeline takes its words, or
    ``transcribed`` while the person's ending arm waits for their confirm;
    ``handed_off`` once the pipeline took the call's words or another take
    beat them (``hand_off`` says which); or ``failed``, with ``hand_off``
    saying why, while the interview stays open."""
    interview: VoiceCallInterview | None = None
    """``taken`` once the call's words took the person's interview, which is
    then over; ``superseded`` once the person typed into their interview
    after the call ended, so the typed interview owns it; None while neither
    has happened. Read before ``end_reason`` and ``status``."""
    end_reason: str | None = None
    """Why the call ended: ``finished`` once Otto said goodbye, the control
    that ended it (``wall_clock``, ``turn_cap``, ``idle``, ``scope``,
    ``spend_cap``, ``user_hangup``, ``superseded``, ``client_gone``,
    ``vendor_error``, ``server_restart``), or ``crashed`` when the server
    running it stopped."""
    leg: Literal["livekit", "websocket"]
    hand_off: FinalizeResponse | None = None
    """What the onboarding pipeline answered the call's hand-off, read the
    way the typed interview's finish answer is read; None until it has
    answered."""


class VoiceCallFinishRequest(BaseModel):
    """The person's confirm of a voice call."""

    corrections: str | None = Field(default=None, max_length=MAX_CORRECTION_CHARS)
    """What they changed on the ending screen, handed off as their own words
    after the call's."""
