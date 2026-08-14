"""Dev roster seed for Experts.

Run with: poetry run python -m backend.api.features.experts.seed

Upserts the three roster templates (Maria, Max, Frankie) by template name,
so repeated runs keep the same template ids. Preload workflows are resolved
from official store listing slugs; all listings are validated before any
template is mutated. Each upsert also refreshes the presentation fields
(avatar, bio, skills) on experts already hired from that template, so roster
changes reach existing users and not just new hires.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import TypedDict

import prisma.models

from backend.data import db as database
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

# StoreListing slugs are only unique per owner, so slug resolution must be
# scoped to the official creator — otherwise another creator publishing the
# same slug could get their listing preloaded. Matches the username the
# checked-in marketplace assets (backend/agents) publish under and the live
# marketplace creator of the roster listings.
OFFICIAL_CREATOR_USERNAME = "autogpt"


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
        "tagline": "Writes your LinkedIn posts, SEO articles, and webpage copy.",
        "avatar_url": "/experts/maria.svg",
        "bio": """I'm a senior marketing strategist — fifteen years across B2B SaaS and consumer brands — and I lead with positioning before tactics: who the customer is, what keeps them up at night, and why they'd pick you over doing nothing. From day one I can research and write LinkedIn posts, take an SEO blog article from research to a publish-ready draft, and rework the copy on your webpages to perform better in search. Everything ships in clear, confident prose with the jargon stripped out.""",
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
        "tagline": "Finds your leads, their decision-makers, and their contact details.",
        "avatar_url": "/experts/max.svg",
        "bio": """I'm a sales development expert who's built outbound pipelines for startups and mid-market teams, and I treat most pipeline problems as targeting problems in disguise — so I start by sharpening your ideal customer profile before I go hunting. From day one I can pull lists of businesses that fit that profile, surface the owner or decision-maker behind a company, and track down a contact's email address. Volume without fit is noise, and I say so plainly.""",
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
        "tagline": "Starts your day briefed: meeting prep, support email, and a morning digest.",
        "avatar_url": "/experts/frankie.svg",
        "bio": """I'm an operations specialist who's run the back office for fast-growing teams, and my job is to keep you ahead of the routine instead of buried in it. From day one I can brief you before your business meetings, draft replies to the support email flooding your inbox, and land a personalized morning digest on your desk at 7:40 every day. I'm conservative about commitments: I never promise a date, refund, or policy exception on your behalf — I draft it and flag it for you to approve.""",
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


# Cadences dropped when the roster consolidated on a single scheduled
# workflow (Frankie's daily digest). _sync_preloads only reaches template
# rows, so hires made before the change would keep firing these forever —
# every seed run retries this cleanup until no hired copy still carries the
# old template-managed cadence. Rows whose cron a user changed no longer
# match and are deliberately left alone.
REMOVED_TEMPLATE_CADENCES: list[tuple[str, str]] = [
    ("automated-blog-writer", "0 9 * * 1"),
    ("lead-finder-local-businesses", "0 8 * * 1"),
    ("smart-meeting-brief", "0 7 * * 1-5"),
]


async def _resolve_active_version_id(slug: str) -> str | None:
    # (owningUserId, slug) is the listing's uniqueness, and the username is
    # unique, so this match is deterministic: at most one listing.
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={
            "slug": slug,
            "isDeleted": False,
            "CreatorProfile": {"is": {"username": OFFICIAL_CREATOR_USERNAME}},
        }
    )
    if listing is None:
        return None
    return listing.activeVersionId


async def _clear_removed_cadences() -> int:
    """Migrate hired copies off cadences the roster no longer ships.

    Deletes each hired copy's scheduler job by owner + scheduleId and clears
    the row only once the job is confirmed gone (deleted now, or already
    absent) — a scheduler failure preserves scheduleId/scheduleCron so the
    next seed run retries, mirroring ``detach_expert_triggers``.
    """
    cleared = 0
    live_by_owner: dict[str, set[str]] = {}
    for slug, old_cron in REMOVED_TEMPLATE_CADENCES:
        rows = await prisma.models.ExpertWorkflow.prisma().find_many(
            where={
                "StoreListingVersion": {
                    "is": {
                        "StoreListing": {
                            "is": {
                                "slug": slug,
                                "isDeleted": False,
                                "CreatorProfile": {
                                    "is": {
                                        "username": OFFICIAL_CREATOR_USERNAME,
                                    }
                                },
                            }
                        }
                    }
                },
                "scheduleCron": old_cron,
                "Expert": {"is": {"isTemplate": False}},
            },
            include={"Expert": True},
        )
        for row in rows:
            owner = row.Expert.ownerUserId if row.Expert else None
            if owner is None:
                continue
            if row.scheduleId is not None and not await _delete_live_schedule(
                owner, row.scheduleId, live_by_owner
            ):
                continue
            await prisma.models.ExpertWorkflow.prisma().update(
                where={"id": row.id},
                data={"scheduleId": None, "scheduleCron": None},
            )
            cleared += 1
    if cleared:
        logger.info(f"Cleared removed roster cadences on {cleared} hired workflows")
    return cleared


async def _delete_live_schedule(
    owner_id: str, schedule_id: str, live_by_owner: dict[str, set[str]]
) -> bool:
    """Delete *schedule_id* if the scheduler still has it for *owner_id*.

    Returns True when the job is confirmed gone; False on any scheduler
    failure so the caller keeps the row for a later retry.
    """
    try:
        if owner_id not in live_by_owner:
            schedules = await get_scheduler_client().get_execution_schedules(
                user_id=owner_id, kind="graph"
            )
            live_by_owner[owner_id] = {s.id for s in schedules}
        if schedule_id not in live_by_owner[owner_id]:
            return True
        await get_scheduler_client().delete_schedule(schedule_id, user_id=owner_id)
        live_by_owner[owner_id].discard(schedule_id)
        return True
    except Exception as e:
        logger.warning(
            f"Could not delete schedule #{schedule_id} for user #{owner_id}; "
            f"keeping cadence for retry: {type(e).__name__}: {e}"
        )
        return False


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
    avatar/tagline/bio/skills forever. ``name`` is deliberately excluded —
    users may have renamed their hire — as are ``role``/``identity``, which
    drive live persona behaviour.
    """
    return await prisma.models.Expert.prisma().update_many(
        where={"sourceTemplateId": template.id, "isTemplate": False},
        data={
            "avatarUrl": template.avatarUrl,
            "tagline": template.tagline,
            "bio": template.bio,
            "skills": template.skills,
        },
    )


async def _sync_preloads(
    template_id: str,
    entry: RosterEntry,
    resolved_versions: Mapping[str, str] | None = None,
) -> None:
    existing = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={"expertId": template_id}
    )
    existing_by_version = {w.storeListingVersionId: w for w in existing}
    for preload in entry["preloads"]:
        version_id = (
            resolved_versions.get(preload["slug"])
            if resolved_versions is not None
            else await _resolve_active_version_id(preload["slug"])
        )
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


async def _resolve_roster_preloads() -> dict[str, str]:
    slugs = {preload["slug"] for entry in ROSTER for preload in entry["preloads"]}
    resolved = {
        slug: version_id
        for slug in sorted(slugs)
        if (version_id := await _resolve_active_version_id(slug)) is not None
    }
    missing = sorted(slugs - resolved.keys())
    if missing:
        raise RuntimeError(
            f"Official creator '{OFFICIAL_CREATOR_USERNAME}' is missing roster "
            f"listings for: {', '.join(missing)}. Load marketplace store assets "
            "before seeding the expert roster."
        )
    return resolved


async def seed_roster() -> list[str]:
    """Upsert the roster templates and their preloads. Returns template ids."""
    resolved_versions = await _resolve_roster_preloads()
    template_ids = []
    for entry in ROSTER:
        template = await _upsert_template(entry)
        await _sync_preloads(template.id, entry, resolved_versions)
        refreshed = await _backfill_hired_copies(template)
        template_ids.append(template.id)
        logger.info(
            f"Seeded expert template '{entry['name']}' (#{template.id}); "
            f"refreshed {refreshed} hired copies"
        )
    await _clear_removed_cadences()
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
