"""Shapes of a roster seed entry, shared by the seed and the roster modules."""

# typing_extensions, not typing: pydantic models carry these on Python 3.11.
from typing_extensions import TypedDict

from backend.api.features.experts.models import ExpertDayOneItem, VoiceSample


class PreloadSeed(TypedDict):
    slug: str
    # Unix cron cadence for install-time scheduling (issue #13714); None
    # means the workflow installs without a schedule. Applied to template
    # rows on every seed run, but only copied to hires made afterwards —
    # existing hires keep the schedule they were created with.
    #
    # A cadence fires unattended from the day of hire, so it may only go on a
    # workflow that acts on nothing outside the platform — typically research.
    # The marketplace reviewer is that gate; nothing here enforces it.
    cron: str | None


class RoutineSeed(TypedDict):
    # Stable slug; renaming one orphans the old row on existing hires.
    key: str
    title: str
    # This proposal is rewritten with the owner's answers before scheduling.
    prompt: str
    # Five-field cron suggestions. H spreads the minute within an hour.
    crons: list[str]
    # Questions that must be answered before the routine can be enabled.
    asks: list[str]
    # THREAD keeps one durable chat; FRESH starts a new chat for each run.
    session_mode: str


class RosterEntry(TypedDict):
    # Stable catalog key (`maria`); the template row is resolved by this.
    key: str
    name: str
    role: str
    job_title: str
    tagline: str
    avatar_url: str | None
    bio: str
    # Skills Hub listing slugs a hire gets installed. Listing ids differ per
    # environment, so the seed resolves these to ids and the relation stores those.
    bundled_skills: list[str]
    # Canonical marketplace categories, so the category chip narrows the roster.
    # Declared here rather than derived from `role`: "Ops" folds onto no
    # canonical value, and a raised expert's role is free text.
    categories: list[str]
    identity: str
    voice_preferences: str
    # Two writing samples in the persona's voice; the hire flow shows these as
    # the "how should {name} write?" pick right after hire.
    voice_samples: list[VoiceSample]
    boundaries: str
    # Up to three rows for the profile's "sets up on day one"; empty hides it.
    day_one: list[ExpertDayOneItem]
    preloads: list[PreloadSeed]
    # Standing work proposals ship disabled and without credential grants.
    routines: list[RoutineSeed]
