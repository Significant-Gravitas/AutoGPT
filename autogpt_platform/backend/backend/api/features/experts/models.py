import json
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.data.expert_run_output import OutputType

ExpertRunStatus = Literal[
    "incomplete",
    "queued",
    "running",
    "completed",
    "terminated",
    "failed",
    "review",
]

AI_DISCLOSURE_RULE = "The expert discloses that it is AI when acting externally."
EXTERNAL_ACTION_APPROVAL_RULE = "External actions require approval."
PROTECTED_SOUL_RULES = (AI_DISCLOSURE_RULE, EXTERNAL_ACTION_APPROVAL_RULE)

_EXPERT_NAME_MAX_LENGTH = 100
EXPERT_IDENTITY_MAX_LENGTH = 10_000
_EXPERT_SOUL_TEXT_MAX_LENGTH = 4_000
EXPERT_COLOR_MAX_LENGTH = 32
EXPERT_AVATAR_URL_MAX_LENGTH = 2_000

# Avatars come from our own upload (an absolute https URL) or ship with a
# roster template (a path under /public). Anything else — notably data:,
# javascript: and plaintext http: — is refused so a stored value can never
# carry script or be fetched unencrypted.
# Backslashes and tab/CR/LF are refused outright: browsers strip the control
# characters and treat "\" as "/" for special schemes, so "/\evil.example/a.png"
# would resolve to a third-party origin despite looking site-relative.
_AVATAR_URL_FORBIDDEN_CHARS = ("\\", "\t", "\r", "\n")


def validate_avatar_url(value: str | None) -> str | None:
    """Accept an absolute https URL or a site-relative path, else reject."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if any(char in stripped for char in _AVATAR_URL_FORBIDDEN_CHARS):
        raise ValueError(
            "Avatar URL must not contain backslashes or control characters"
        )
    if stripped.startswith("//"):
        raise ValueError("Avatar URL must not be protocol-relative")
    if stripped.startswith("/"):
        return stripped
    parsed = urlparse(stripped)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Avatar URL must be an https URL or a relative path")
    return stripped


# The soul strippers run as "before" validators, ahead of the length
# constraints, so a padded value is measured after trimming and a blank one
# fails with the field's own message. Non-str input passes through untouched
# for Pydantic to reject with its type error.
def _strip_required_soul_field(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        raise ValueError("Field must not be blank")
    return stripped


def _strip_optional_soul_field(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


class VoiceSample(BaseModel):
    """A short writing sample in a persona's own voice, offered as a pick in
    the hire or raise flow. The first sample is choice "a", the second choice
    "b"."""

    label: str
    text: str


class ExpertWorkflowRef(BaseModel):
    id: str
    store_listing_version_id: str | None
    library_agent_id: str | None
    graph_id: str | None
    name: str | None
    description: str | None
    # Roster cadence + the schedule created from it at install time.
    # A cron with a null schedule_id means the schedule could not be
    # created yet (e.g. missing credentials) — the workflow needs setup.
    schedule_cron: str | None = None
    schedule_id: str | None = None


class ExpertIdentity(BaseModel):
    id: str
    name: str
    avatar_url: str | None
    role: str
    is_archived: bool


class Expert(BaseModel):
    id: str
    name: str
    avatar_url: str | None
    # Accent color token chosen while raising; "" when unset.
    color: str = ""
    role: str
    tagline: str | None
    bio: str | None
    skills: list[str]
    identity: str
    voice_preferences: str
    # Populated only on roster templates so the hire flow can offer a voice
    # pick; always empty on hired copies, which persist the user's plain-text
    # choice in voice_preferences instead.
    voice_samples: list[VoiceSample] = []
    boundaries: str
    protected_soul_rules: list[str]
    is_template: bool
    source_template_id: str | None
    is_archived: bool
    workflows: list[ExpertWorkflowRef]
    # Latest expert-attributed execution, for the /team card's status line.
    last_run_at: datetime | None = None
    last_run_status: str | None = None
    # Weekly credit guardrail: effective budget (expert's own or the
    # platform default; None = guardrail disabled), current-week spend,
    # and the pause flag set on budget breach or archive.
    weekly_budget: int | None = None
    weekly_spend: int = 0
    schedules_paused_at: datetime | None = None
    # Owner-scoped grouping. None = ungrouped ("unpodded").
    pod_id: str | None = None


# Membership is deliberately not embedded: clients already hold the expert
# list and each Expert carries `pod_id`, so the roster groups client-side.
class ExpertPod(BaseModel):
    """A named group of hired experts, scoped to the owner."""

    id: str
    name: str
    created_at: datetime


class ExpertRun(BaseModel):
    """One expert-attributed execution, for the /team Work surface."""

    execution_id: str
    graph_id: str
    agent_name: str
    library_agent_id: str | None
    status: ExpertRunStatus
    output_type: OutputType
    # Which output pin was classified, so the viewer opens exactly that value.
    output_key: str | None
    needs_review: bool
    started_at: datetime | None
    ended_at: datetime | None
    link: str | None


class ExpertDetachPreview(BaseModel):
    """What archiving the expert would pause — drives the confirm dialog."""

    schedule_names: list[str]
    trigger_names: list[str]


class HireResult(BaseModel):
    expert: Expert
    failed_preloads: list[str]


RaiseAttachmentKind = Literal["workflow", "skill"]
RaiseAttachmentSource = Literal["marketplace", "library"]
RaiseAttachmentFailureReason = Literal["unavailable", "installation_failed"]
MAX_RAISE_ATTACHMENTS = 20
WEEKLY_BUDGET_MAX_CREDITS = 1_000_000


class RaiseAttachment(BaseModel):
    """One workflow or skill to attach while raising an expert.

    ``id`` is a store listing version UUID (marketplace), a library agent
    UUID (library workflow), or a copilot skill slug (library skill).
    Marketplace skills use a store listing version UUID; the listing's
    public name is stored on ``Expert.skills``.
    """

    kind: RaiseAttachmentKind
    source: RaiseAttachmentSource
    id: str = Field(min_length=1, max_length=100)

    # "before" so the length bounds apply to the trimmed id: a padded but
    # in-range id is accepted, and a whitespace-only one fails with the
    # message below instead of the generic min_length error.
    @field_validator("id", mode="before")
    @classmethod
    def strip_id(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Attachment id must not be blank")
        return stripped


class RaiseAttachmentFailure(BaseModel):
    kind: RaiseAttachmentKind
    source: RaiseAttachmentSource
    id: str
    reason: RaiseAttachmentFailureReason


class RaiseResult(BaseModel):
    """Result of raising a blank expert.

    Attachments are validated before the expert row is created. A later
    install failure is non-fatal and listed in ``failed_attachments`` so
    the client can surface partial success.
    """

    expert: Expert
    failed_attachments: list[RaiseAttachmentFailure] = []


class ExpertSoulUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=_EXPERT_NAME_MAX_LENGTH)
    identity: str = Field(min_length=1, max_length=EXPERT_IDENTITY_MAX_LENGTH)
    voice_preferences: str = Field(max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH)
    boundaries: str = Field(max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH)

    @field_validator("name", "identity", mode="before")
    @classmethod
    def strip_required_fields(cls, value: object) -> object:
        return _strip_required_soul_field(value)

    @field_validator("voice_preferences", "boundaries", mode="before")
    @classmethod
    def strip_optional_fields(cls, value: object) -> object:
        return _strip_optional_soul_field(value)


class ExpertSoulFieldsPatch(BaseModel):
    """Partial Soul edit: only supplied fields are validated and written.

    Mirrors ``ExpertSoulUpdate``'s per-field rules (lengths, blank handling)
    but leaves ``None`` fields untouched so disjoint concurrent edits cannot
    clobber each other.
    """

    identity: str | None = Field(
        default=None, min_length=1, max_length=EXPERT_IDENTITY_MAX_LENGTH
    )
    voice_preferences: str | None = Field(
        default=None, max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH
    )
    boundaries: str | None = Field(
        default=None, max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH
    )

    @field_validator("identity", mode="before")
    @classmethod
    def strip_required_fields(cls, value: object) -> object:
        return _strip_required_soul_field(value)

    @field_validator("voice_preferences", "boundaries", mode="before")
    @classmethod
    def strip_optional_fields(cls, value: object) -> object:
        return _strip_optional_soul_field(value)


def encode_voice_preferences(description: str, samples: list[VoiceSample]) -> str:
    """Serialize a roster template's voice into the voicePreferences column: a
    plain-text description plus the writing samples the hire flow offers.

    Only template rows carry this JSON envelope. Hired copies persist the
    user's plain-text pick (or the template's description on skip), so the
    prompt builder never renders raw JSON into ``<voice_preferences>``.
    """
    return json.dumps(
        {
            "description": description,
            "samples": [sample.model_dump() for sample in samples],
        }
    )


def decode_voice_preferences(raw: str) -> tuple[str, list[VoiceSample]]:
    """Inverse of ``encode_voice_preferences``.

    Templates carry the JSON envelope; hired copies carry a plain string,
    which round-trips back unchanged with no samples. Once a value identifies
    itself as the template envelope, malformed fields degrade to safe empty
    values so serialized JSON can never leak into an expert's prompt.
    """
    if not raw:
        return "", []
    try:
        envelope = json.loads(raw)
    except (ValueError, TypeError):
        return raw, []
    if not isinstance(envelope, dict) or "samples" not in envelope:
        return raw, []
    description = envelope.get("description")
    envelope_samples = envelope.get("samples")
    if not isinstance(description, str) or not isinstance(envelope_samples, list):
        return "", []
    samples: list[VoiceSample] = []
    for item in envelope_samples:
        if not isinstance(item, dict):
            continue
        try:
            samples.append(VoiceSample.model_validate(item))
        except ValidationError:
            continue
    return description, samples
