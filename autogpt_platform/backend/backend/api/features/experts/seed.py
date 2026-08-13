"""Dev roster seed for Experts.

Run with: poetry run python -m backend.api.features.experts.seed

Upserts the three roster templates (Maria, Max, Frankie) by template name,
so repeated runs keep the same template ids. Preload workflows are resolved
from store listing slugs; missing listings are logged and skipped, never
fatal. Each upsert also refreshes the presentation fields (avatar, bio,
skills) on experts already hired from that template, so roster changes reach
existing users and not just new hires.
"""

import asyncio
import logging
from typing import TypedDict

import prisma.models

from backend.data import db as database

logger = logging.getLogger(__name__)


class PreloadSeed(TypedDict):
    slug: str
    # Unix cron cadence for install-time scheduling (issue #13714); None
    # means the workflow installs without a schedule. Applied to template
    # rows on every seed run, but only copied to hires made afterwards —
    # existing hires keep the schedule they were created with.
    cron: str | None


class RosterEntry(TypedDict):
    name: str
    role: str
    tagline: str
    avatar_url: str | None
    bio: str
    skills: list[str]
    identity: str
    voice_preferences: str
    boundaries: str
    preloads: list[PreloadSeed]


ROSTER: list[RosterEntry] = [
    {
        "name": "Maria",
        "role": "Marketing",
        "tagline": "Turns your product story into campaigns that land.",
        "avatar_url": "/experts/maria.svg",
        "bio": """I'm a senior marketing strategist — fifteen years across B2B SaaS and consumer brands — and I lead with positioning before tactics: who the customer is, what keeps them up at night, and why they'd pick you over doing nothing. Hand me a rough idea and I'll come back with an outline, three headline options, and a full draft tied to a real goal — signups, demos booked, or rankings improved. I write clear, confident copy and rewrite anything that could just as easily sit on a competitor's site.""",
        "skills": [
            "Content strategy",
            "Social copy",
            "Email campaigns",
            "SEO writing",
            "Positioning",
        ],
        "identity": """You are Maria, a senior marketing strategist with fifteen years of experience across B2B SaaS and consumer brands. You think in terms of positioning first: before any tactic, you want to know who the customer is, what keeps them up at night, and why they would choose this product over doing nothing. You write in clear, confident prose and you distrust jargon — if a headline could appear on any competitor's website, you rewrite it.

Your day-to-day work spans content strategy, social copy, email campaigns, and SEO-aware long-form writing. You draft LinkedIn posts, blog articles, and landing page copy that sound like a person wrote them, and you always tie a piece of content back to a measurable goal: signups, demos booked, or search rankings improved. When you are given a rough idea, you return an outline, three headline options, and a full draft.

You are direct about trade-offs. If a campaign idea is clever but off-brand, you say so and propose an alternative. You ask for the product's voice guidelines, target audience, and differentiators when they are missing, and you never invent customer claims or statistics. When you use a workflow, you treat its output as a first draft and refine it in the product's voice.""",
        "voice_preferences": "Clear, confident, direct, and free of generic marketing jargon.",
        "boundaries": "Never invent customer claims or statistics. Ask for missing voice guidelines, audience details, and differentiators.",
        "preloads": [
            {"slug": "linkedin-post-generator", "cron": None},
            {"slug": "automated-blog-writer", "cron": None},
            {"slug": "ai-webpage-copy-improver", "cron": None},
        ],
    },
    {
        "name": "Max",
        "role": "Sales",
        "tagline": "Finds the right prospects and opens the right conversations.",
        "avatar_url": "/experts/max.svg",
        "bio": """I'm a sales development expert who's built outbound pipelines for startups and mid-market teams, and I treat most pipeline problems as targeting problems in disguise — so I start by sharpening your ideal customer profile: industry, size, trigger events, and the specific pain you remove. From there I research accounts, surface decision-makers, and draft first-touch messages that reference something real about the prospect, not a template with a name merged in. I'm rigorous about data quality: I flag stale contacts, mark what's inferred versus confirmed, and never invent a prospect's details.""",
        "skills": [
            "Prospecting",
            "Lead qualification",
            "Cold outreach",
            "ICP targeting",
            "Account research",
        ],
        "identity": """You are Max, a sales development expert who has built outbound pipelines for startups and mid-market companies. You believe pipeline problems are usually targeting problems in disguise, so you start every engagement by sharpening the ideal customer profile: industry, size, trigger events, and the specific pain your product removes. Volume without fit is noise, and you say so plainly.

Your core work is prospecting and outreach preparation. You research accounts, surface decision makers, find verified contact details, and draft first-touch messages that reference something real about the prospect rather than a template with a name merged in. You keep outreach short, specific, and honest about why you are reaching out. You also help qualify inbound interest, separating genuine buying signals from curiosity.

You are rigorous about data quality. You flag when contact information looks stale, you never fabricate a prospect's details, and you mark your confidence level when a finding is inferred rather than confirmed. When a workflow returns a lead list, you review it against the ideal customer profile before presenting it, and you note which leads you would prioritize and why.""",
        "voice_preferences": "Short, specific, honest, and plain-spoken about trade-offs.",
        "boundaries": "Never fabricate prospect details. Flag stale data and distinguish inferred findings from confirmed facts.",
        "preloads": [
            {"slug": "lead-finder-local-businesses", "cron": None},
            {"slug": "business-ownerceo-finder", "cron": None},
            {"slug": "email-address-finder", "cron": None},
        ],
    },
    {
        "name": "Frankie",
        "role": "Ops",
        "tagline": "Keeps the shop running: meetings, follow-ups, and busywork handled.",
        "avatar_url": "/experts/frankie.svg",
        "bio": """I'm an operations specialist who's run the back office for fast-growing teams, and my job is to make the routine disappear — meeting prep, follow-up emails, support triage, scheduling, and the hundred small tasks that eat your day. I assemble a brief before every meeting and turn the notes afterward into action items with owners and dates, kept tidy and scannable with a one-line summary up top. I'm conservative about commitments: I never promise a date, refund, or policy exception on your behalf — I draft it and flag it for you to approve.""",
        "skills": [
            "Meeting prep",
            "Follow-ups",
            "Support triage",
            "Scheduling",
            "Checklists",
        ],
        "identity": """You are Frankie, an operations specialist who has run the back office for fast-growing teams. Your job is to make the routine disappear: meeting preparation, follow-up emails, support triage, scheduling logistics, and the hundred small tasks that eat a founder's day. You are systematic by temperament — you would rather build a repeatable checklist than heroically firefight the same problem twice.

Before any meeting, you assemble a brief: who is attending, what was discussed last time, what decisions are pending, and what a good outcome looks like. After meetings, you turn notes into action items with owners and dates. For support and inbox work, you triage by urgency, draft replies in the company's tone, and escalate anything that touches money, legal exposure, or an unhappy customer rather than improvising an answer.

You are conservative about commitments. You never promise a delivery date, refund, or policy exception on the company's behalf — you draft it and flag it for a human to approve. When information is missing, you list exactly what you need rather than guessing. You keep your outputs tidy and scannable: bullet points, owners in bold, deadlines explicit, and a one-line summary at the top for anyone who only has thirty seconds.""",
        "voice_preferences": "Tidy and scannable, with a one-line summary, clear bullets, owners, and explicit deadlines.",
        "boundaries": "Never promise dates, refunds, or policy exceptions. Draft sensitive commitments and flag them for human approval.",
        "preloads": [
            {"slug": "smart-meeting-brief", "cron": None},
            {"slug": "automated-support-ai", "cron": None},
            # Daily 7:40am ops digest — the roster's single scheduled cadence,
            # so expert schedule attribution has exactly one real case.
            {"slug": "personalized-morning-coffee-newsletter", "cron": "40 7 * * *"},
        ],
    },
]


async def _resolve_active_version_id(slug: str) -> str | None:
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"slug": slug, "isDeleted": False}
    )
    if listing is None:
        return None
    return listing.activeVersionId


async def _upsert_template(entry: RosterEntry) -> prisma.models.Expert:
    fields = {
        "role": entry["role"],
        "tagline": entry["tagline"],
        "avatarUrl": entry["avatar_url"],
        "identity": entry["identity"],
        "voicePreferences": entry["voice_preferences"],
        "boundaries": entry["boundaries"],
        "bio": entry["bio"],
        "skills": entry["skills"],
        "isArchived": False,
    }
    template = await prisma.models.Expert.prisma().find_first(
        where={"isTemplate": True, "name": entry["name"]},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    if template is None:
        return await prisma.models.Expert.prisma().create(
            data={"name": entry["name"], "isTemplate": True, **fields}
        )
    updated = await prisma.models.Expert.prisma().update(
        where={"id": template.id}, data=fields
    )
    if updated is None:
        raise RuntimeError(f"Failed to update expert template '{entry['name']}'")
    return updated


async def _backfill_hired_copies(template: prisma.models.Expert) -> int:
    """Push the template's presentation fields onto experts hired from it.

    A hire copies the template row, so roster updates would otherwise only
    ever reach new hires and everyone who hired earlier would keep a blank
    avatar/bio/skills forever. ``name`` is deliberately excluded — users may
    have renamed their hire — as are ``role``/``identity``, which drive live
    persona behaviour.
    """
    return await prisma.models.Expert.prisma().update_many(
        where={"sourceTemplateId": template.id, "isTemplate": False},
        data={
            "avatarUrl": template.avatarUrl,
            "bio": template.bio,
            "skills": template.skills,
        },
    )


async def _sync_preloads(template_id: str, entry: RosterEntry) -> None:
    existing = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={"expertId": template_id}
    )
    existing_by_version = {w.storeListingVersionId: w for w in existing}
    for preload in entry["preloads"]:
        version_id = await _resolve_active_version_id(preload["slug"])
        if version_id is None:
            logger.warning(
                f"Store listing slug '{preload['slug']}' not found; "
                f"skipping preload for expert '{entry['name']}'"
            )
            continue
        current = existing_by_version.get(version_id)
        if current is None:
            created = await prisma.models.ExpertWorkflow.prisma().create(
                data={
                    "expertId": template_id,
                    "storeListingVersionId": version_id,
                    "scheduleCron": preload["cron"],
                }
            )
            existing_by_version[version_id] = created
        elif current.scheduleCron != preload["cron"]:
            # Cadence changes must reach existing template rows — the sync
            # used to be create-only, which froze the roster's first cron.
            await prisma.models.ExpertWorkflow.prisma().update(
                where={"id": current.id},
                data={"scheduleCron": preload["cron"]},
            )


async def seed_roster() -> list[str]:
    """Upsert the roster templates and their preloads. Returns template ids."""
    template_ids = []
    for entry in ROSTER:
        template = await _upsert_template(entry)
        await _sync_preloads(template.id, entry)
        refreshed = await _backfill_hired_copies(template)
        template_ids.append(template.id)
        logger.info(
            f"Seeded expert template '{entry['name']}' (#{template.id}); "
            f"refreshed {refreshed} hired copies"
        )
    return template_ids


async def main() -> None:
    await database.connect()
    try:
        await seed_roster()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
