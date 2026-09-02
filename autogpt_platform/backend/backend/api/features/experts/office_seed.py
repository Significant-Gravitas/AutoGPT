"""Dev seed for hireable "office" packs.

Run with: poetry run python -m backend.api.features.experts.office_seed

Upserts the three office packs (YouTuber, SaaS founder, Agency) by
OfficeTemplate name, so repeated runs keep the same office ids. Each pack
lists 2-3 expert templates plus an intro task per expert; template ids are
resolved by template NAME at seed time. Roster templates (Maria, Max,
Frankie — see ``seed.ROSTER``) are reused where they fit and must already
be seeded; the office-only gaps (a video Scriptwriter, an Account manager)
are upserted here as minimal templates with no preloads.
"""

import asyncio
import logging
from typing import TypedDict

import prisma.models

from backend.data import db as database
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class MinimalTemplateEntry(TypedDict):
    name: str
    role: str
    tagline: str
    identity: str
    bio: str
    skills: list[str]


class OfficeExpertSeed(TypedDict):
    # Resolved to a template id by name at seed time.
    template_name: str
    schedule_cron: str | None
    intro_task_title: str
    intro_task_spec: str


class OfficePackSeed(TypedDict):
    name: str
    description: str
    experts: list[OfficeExpertSeed]


# Office-only templates. Deliberately minimal (no preloads, no voice
# envelope): they exist so a pack can ship a persona the roster lacks.
MINIMAL_TEMPLATES: list[MinimalTemplateEntry] = [
    {
        "name": "Sasha",
        "role": "Scriptwriter",
        "tagline": "Turns your video ideas into scripts with hooks that hold.",
        "identity": (
            "You are Sasha, a video scriptwriter who has written for creators "
            "from 10k to 2M subscribers. You structure every script around a "
            "hook in the first ten seconds, an open loop that earns the next "
            "minute, and a payoff that matches the title's promise. You write "
            "in spoken language — short sentences, contractions, no prose that "
            "only works on paper. When given a topic you return a working "
            "title, a hook, a beat outline, and a full script draft. You flag "
            "claims that need a source and never invent statistics."
        ),
        "bio": (
            "I'm a video scriptwriter — hooks, open loops, and payoffs that "
            "match the title. Give me a topic and I'll hand back a working "
            "title, a beat outline, and a full draft in spoken language."
        ),
        "skills": [
            "Video scripts",
            "Hooks",
            "Story structure",
            "Titles & thumbnails",
        ],
    },
    {
        "name": "Alex",
        "role": "Account manager",
        "tagline": "Keeps every client warm: updates, check-ins, and renewals.",
        "identity": (
            "You are Alex, an account manager who has kept churn low at two "
            "agencies by treating every client like the only one. You keep a "
            "running picture of each account: what was promised, what shipped, "
            "what's next, and how the client feels about it. You draft status "
            "updates, check-in emails, and renewal conversations in a warm, "
            "professional voice. You never promise scope, dates, or discounts "
            "on the agency's behalf — you draft them and flag them for "
            "approval. When information about an account is missing, you list "
            "exactly what you need."
        ),
        "bio": (
            "I'm an account manager — I keep clients warm with status "
            "updates, check-ins, and renewal prep, and I flag anything that "
            "commits the agency before it goes out."
        ),
        "skills": [
            "Client updates",
            "Check-ins",
            "Renewals",
            "Account health",
        ],
    },
]


OFFICE_PACKS: list[OfficePackSeed] = [
    {
        "name": "YouTuber",
        "description": (
            "A creator's back office: a scriptwriter for your videos, a "
            "marketer to grow the channel, and ops to keep you on schedule."
        ),
        "experts": [
            {
                "template_name": "Sasha",
                "schedule_cron": None,
                "intro_task_title": "Draft a script for your next video",
                "intro_task_spec": (
                    "Ask what the channel is about and what the next video "
                    "should cover, then deliver a working title, a hook, a "
                    "beat outline, and a full script draft."
                ),
            },
            {
                "template_name": "Maria",
                "schedule_cron": "0 9 * * 1",
                "intro_task_title": "Plan this week's channel promotion",
                "intro_task_spec": (
                    "Learn the channel's niche and audience, then draft a "
                    "one-week promotion plan: three social posts and one "
                    "community post that point at the latest video."
                ),
            },
            {
                "template_name": "Frankie",
                "schedule_cron": None,
                "intro_task_title": "Set up your publishing checklist",
                "intro_task_spec": (
                    "Build a repeatable pre-publish checklist (title, "
                    "thumbnail, description, tags, end screen, community "
                    "post) tailored to how this channel works."
                ),
            },
        ],
    },
    {
        "name": "SaaS founder",
        "description": (
            "A founder's first three hires: outbound sales, marketing copy, "
            "and an ops brief that starts every day for you."
        ),
        "experts": [
            {
                "template_name": "Max",
                "schedule_cron": "0 9 * * 1",
                "intro_task_title": "Build your first prospect list",
                "intro_task_spec": (
                    "Sharpen the ideal customer profile (industry, size, "
                    "trigger events), then assemble a first list of ten "
                    "prospects with the decision-maker and a one-line reason "
                    "each is a fit."
                ),
            },
            {
                "template_name": "Maria",
                "schedule_cron": None,
                "intro_task_title": "Sharpen your landing page copy",
                "intro_task_spec": (
                    "Review the current landing page, then propose a "
                    "rewritten headline, subheadline, and three benefit "
                    "bullets positioned against doing nothing."
                ),
            },
            {
                "template_name": "Frankie",
                "schedule_cron": None,
                "intro_task_title": "Set up your weekly ops rhythm",
                "intro_task_spec": (
                    "Draft a weekly operating checklist: metrics to glance "
                    "at, follow-ups to send, and the one meeting brief that "
                    "should never be skipped."
                ),
            },
        ],
    },
    {
        "name": "Agency",
        "description": (
            "An agency pod: an account manager who keeps clients warm, "
            "sales to fill the pipeline, and marketing for your own brand."
        ),
        "experts": [
            {
                "template_name": "Alex",
                "schedule_cron": None,
                "intro_task_title": "Map your client accounts",
                "intro_task_spec": (
                    "List the agency's current clients and, for each, what "
                    "was promised, what shipped last, and the next touchpoint "
                    "— then flag the account most at risk of going quiet."
                ),
            },
            {
                "template_name": "Max",
                "schedule_cron": None,
                "intro_task_title": "Draft your new-business hit list",
                "intro_task_spec": (
                    "Define the agency's ideal client profile and assemble a "
                    "short hit list of prospects with decision-makers and a "
                    "personalised opening line for each."
                ),
            },
            {
                "template_name": "Maria",
                "schedule_cron": "0 9 * * 1",
                "intro_task_title": "Plan the agency's own marketing week",
                "intro_task_spec": (
                    "Agencies market everyone but themselves — draft one "
                    "case-study outline and two social posts that show off "
                    "recent client work."
                ),
            },
        ],
    },
]


async def seed_offices() -> list[str]:
    """Upsert the office-only templates and the office packs. Returns the
    OfficeTemplate ids."""
    for entry in MINIMAL_TEMPLATES:
        await _upsert_minimal_template(entry)

    template_ids = {
        name: await _resolve_template_id(name)
        for name in sorted(
            {e["template_name"] for pack in OFFICE_PACKS for e in pack["experts"]}
        )
    }

    office_ids = []
    for pack in OFFICE_PACKS:
        office = await _upsert_office(pack, template_ids)
        office_ids.append(office.id)
        logger.info(f"Seeded office pack '{pack['name']}' (#{office.id})")
    return office_ids


async def _upsert_minimal_template(entry: MinimalTemplateEntry) -> None:
    fields = {
        "role": entry["role"],
        "tagline": entry["tagline"],
        "identity": entry["identity"],
        "bio": entry["bio"],
        "skills": entry["skills"],
        "voicePreferences": "",
        "boundaries": "",
        "isArchived": False,
    }
    existing = await prisma.models.Expert.prisma().find_first(
        where={"isTemplate": True, "name": entry["name"]},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    if existing is None:
        await prisma.models.Expert.prisma().create(
            data={"name": entry["name"], "isTemplate": True, **fields}
        )
        return
    updated = await prisma.models.Expert.prisma().update(
        where={"id": existing.id}, data=fields
    )
    if updated is None:
        raise RuntimeError(f"Failed to update expert template '{entry['name']}'")


async def _resolve_template_id(name: str) -> str:
    """Template id for *name*. Roster names must be seeded first by
    ``backend.api.features.experts.seed`` — this module never creates them,
    or a minimal upsert would clobber the roster's full persona."""
    template = await prisma.models.Expert.prisma().find_first(
        where={"isTemplate": True, "name": name, "isArchived": False},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    if template is None:
        raise RuntimeError(
            f"Expert template '{name}' not found. Run "
            "`poetry run python -m backend.api.features.experts.seed` first."
        )
    return template.id


async def _upsert_office(
    pack: OfficePackSeed, template_ids: dict[str, str]
) -> prisma.models.OfficeTemplate:
    config = SafeJson(
        {
            "experts": [
                {
                    "template_id": template_ids[entry["template_name"]],
                    "schedule_cron": entry["schedule_cron"],
                    "intro_task_title": entry["intro_task_title"],
                    "intro_task_spec": entry["intro_task_spec"],
                }
                for entry in pack["experts"]
            ]
        }
    )
    existing = await prisma.models.OfficeTemplate.prisma().find_unique(
        where={"name": pack["name"]}
    )
    if existing is None:
        return await prisma.models.OfficeTemplate.prisma().create(
            data={
                "name": pack["name"],
                "description": pack["description"],
                "config": config,
            }
        )
    updated = await prisma.models.OfficeTemplate.prisma().update(
        where={"id": existing.id},
        data={"description": pack["description"], "config": config},
    )
    if updated is None:
        raise RuntimeError(f"Failed to update office template '{pack['name']}'")
    return updated


async def main() -> None:
    await database.connect()
    try:
        await seed_offices()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
