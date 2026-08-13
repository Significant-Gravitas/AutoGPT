import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

AI_DISCLOSURE_RULE = "The expert discloses that it is AI when acting externally."
EXTERNAL_ACTION_APPROVAL_RULE = "External actions require approval."
PROTECTED_SOUL_RULES = (AI_DISCLOSURE_RULE, EXTERNAL_ACTION_APPROVAL_RULE)


class VoiceSample(BaseModel):
    """A short writing sample in a persona's own voice, offered as a pick in
    the hire flow. The first sample is choice "a", the second choice "b"."""

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


class Expert(BaseModel):
    id: str
    name: str
    avatar_url: str | None
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


class ExpertDetachPreview(BaseModel):
    """What archiving the expert would pause — drives the confirm dialog."""

    schedule_names: list[str]
    trigger_names: list[str]


class HireResult(BaseModel):
    expert: Expert
    failed_preloads: list[str]


class ExpertSoulUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    identity: str = Field(min_length=1, max_length=10_000)
    voice_preferences: str = Field(max_length=4_000)
    boundaries: str = Field(max_length=4_000)

    @field_validator("name", "identity")
    @classmethod
    def strip_required_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be blank")
        return stripped

    @field_validator("voice_preferences", "boundaries")
    @classmethod
    def strip_optional_fields(cls, value: str) -> str:
        # Whitespace-only input must collapse to "" so prompt rendering falls
        # back to "Not specified." instead of emitting a blank section.
        return value.strip()


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
    which round-trips back unchanged with no samples. Malformed or legacy
    values degrade to ``(raw, [])`` rather than raising.
    """
    if not raw:
        return "", []
    try:
        envelope = json.loads(raw)
    except (ValueError, TypeError):
        return raw, []
    if not isinstance(envelope, dict) or "samples" not in envelope:
        return raw, []
    samples = [
        VoiceSample(label=item["label"], text=item["text"])
        for item in envelope.get("samples", [])
        if isinstance(item, dict) and "label" in item and "text" in item
    ]
    return envelope.get("description") or "", samples
