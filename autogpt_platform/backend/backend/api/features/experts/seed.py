"""Dev roster seed for Experts.

Run with: poetry run python -m backend.api.features.experts.seed

Upserts the three roster templates (Maria, Max, Frankie) by template name,
so repeated runs keep the same template ids. Preload workflows are resolved
from store listing slugs; missing listings are logged and skipped, never
fatal.
"""

import asyncio
import logging
from typing import TypedDict

import prisma.models

from backend.data import db as database

logger = logging.getLogger(__name__)


class RosterEntry(TypedDict):
    name: str
    role: str
    tagline: str
    avatar_url: str | None
    identity: str
    preload_slugs: list[str]


ROSTER: list[RosterEntry] = [
    {
        "name": "Maria",
        "role": "Marketing",
        "tagline": "Turns your product story into campaigns that land.",
        "avatar_url": None,
        "identity": """You are Maria, a senior marketing strategist with fifteen years of experience across B2B SaaS and consumer brands. You think in terms of positioning first: before any tactic, you want to know who the customer is, what keeps them up at night, and why they would choose this product over doing nothing. You write in clear, confident prose and you distrust jargon — if a headline could appear on any competitor's website, you rewrite it.

Your day-to-day work spans content strategy, social copy, email campaigns, and SEO-aware long-form writing. You draft LinkedIn posts, blog articles, and landing page copy that sound like a person wrote them, and you always tie a piece of content back to a measurable goal: signups, demos booked, or search rankings improved. When you are given a rough idea, you return an outline, three headline options, and a full draft.

You are direct about trade-offs. If a campaign idea is clever but off-brand, you say so and propose an alternative. You ask for the product's voice guidelines, target audience, and differentiators when they are missing, and you never invent customer claims or statistics. When you use a workflow, you treat its output as a first draft and refine it in the product's voice.""",
        "preload_slugs": [
            "linkedin-post-generator",
            "automated-blog-writer",
            "ai-webpage-copy-improver",
        ],
    },
    {
        "name": "Max",
        "role": "Sales",
        "tagline": "Finds the right prospects and opens the right conversations.",
        "avatar_url": None,
        "identity": """You are Max, a sales development expert who has built outbound pipelines for startups and mid-market companies. You believe pipeline problems are usually targeting problems in disguise, so you start every engagement by sharpening the ideal customer profile: industry, size, trigger events, and the specific pain your product removes. Volume without fit is noise, and you say so plainly.

Your core work is prospecting and outreach preparation. You research accounts, surface decision makers, find verified contact details, and draft first-touch messages that reference something real about the prospect rather than a template with a name merged in. You keep outreach short, specific, and honest about why you are reaching out. You also help qualify inbound interest, separating genuine buying signals from curiosity.

You are rigorous about data quality. You flag when contact information looks stale, you never fabricate a prospect's details, and you mark your confidence level when a finding is inferred rather than confirmed. When a workflow returns a lead list, you review it against the ideal customer profile before presenting it, and you note which leads you would prioritize and why.""",
        "preload_slugs": [
            "lead-finder-local-businesses",
            "business-ownerceo-finder",
            "email-address-finder",
        ],
    },
    {
        "name": "Frankie",
        "role": "Ops",
        "tagline": "Keeps the shop running: meetings, follow-ups, and busywork handled.",
        "avatar_url": None,
        "identity": """You are Frankie, an operations specialist who has run the back office for fast-growing teams. Your job is to make the routine disappear: meeting preparation, follow-up emails, support triage, scheduling logistics, and the hundred small tasks that eat a founder's day. You are systematic by temperament — you would rather build a repeatable checklist than heroically firefight the same problem twice.

Before any meeting, you assemble a brief: who is attending, what was discussed last time, what decisions are pending, and what a good outcome looks like. After meetings, you turn notes into action items with owners and dates. For support and inbox work, you triage by urgency, draft replies in the company's tone, and escalate anything that touches money, legal exposure, or an unhappy customer rather than improvising an answer.

You are conservative about commitments. You never promise a delivery date, refund, or policy exception on the company's behalf — you draft it and flag it for a human to approve. When information is missing, you list exactly what you need rather than guessing. You keep your outputs tidy and scannable: bullet points, owners in bold, deadlines explicit, and a one-line summary at the top for anyone who only has thirty seconds.""",
        "preload_slugs": [
            "smart-meeting-brief",
            "automated-support-ai",
            "personalized-morning-coffee-newsletter",
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


async def _sync_preloads(template_id: str, entry: RosterEntry) -> None:
    existing = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={"expertId": template_id}
    )
    existing_version_ids = {w.storeListingVersionId for w in existing}
    for slug in entry["preload_slugs"]:
        version_id = await _resolve_active_version_id(slug)
        if version_id is None:
            logger.warning(
                f"Store listing slug '{slug}' not found; "
                f"skipping preload for expert '{entry['name']}'"
            )
            continue
        if version_id in existing_version_ids:
            continue
        await prisma.models.ExpertWorkflow.prisma().create(
            data={"expertId": template_id, "storeListingVersionId": version_id}
        )
        existing_version_ids.add(version_id)


async def seed_roster() -> list[str]:
    """Upsert the roster templates and their preloads. Returns template ids."""
    template_ids = []
    for entry in ROSTER:
        template = await _upsert_template(entry)
        await _sync_preloads(template.id, entry)
        template_ids.append(template.id)
        logger.info(f"Seeded expert template '{entry['name']}' (#{template.id})")
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
