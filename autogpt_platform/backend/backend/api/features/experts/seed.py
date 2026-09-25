"""Seed the expert roster templates from the skills catalog.

The roster is ``experts/<key>.yml`` in ``Significant-Gravitas/skills-catalog``,
loaded by :mod:`backend.api.features.experts.roster` and handed here by
:func:`backend.api.features.store.skill_catalog.publish_catalog` once the
skills it bundles are on the marketplace. Templates are resolved by their
catalog key (``Expert.templateKey``), so a display name can change without
minting a second template.

Each upsert also refreshes the presentation fields (job title, tagline, bio,
categories) on hired copies, but only where each field still matches the
previous template: an owner's edits and concurrent changes are preserved.
"""

import asyncio
import logging
from collections.abc import Iterable, Mapping
from typing import NotRequired, TypedDict, cast

import prisma.enums
import prisma.models
import prisma.types

from backend.api.features.experts.avatar_catalog import (
    PRESET_AVATAR_URLS,
    resolve_avatar_url,
    resolve_builtin_avatar_url,
)
from backend.api.features.experts.models import encode_day_one, encode_voice_preferences
from backend.api.features.experts.presentation import (
    PresentationBaseline,
    PresentationLike,
    presentation_changes,
)
from backend.api.features.experts.roster_types import RosterEntry, RoutineSeed
from backend.api.features.store.categories import validate_canonical_categories
from backend.util.clients import get_scheduler_client
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

# StoreListing slugs are only unique per owner, so slug resolution must be
# scoped to the official creator: otherwise another creator publishing the
# same slug could get their listing preloaded. Matches the username the
# checked-in marketplace assets (backend/agents) publish under and the live
# marketplace creator of the roster listings.
OFFICIAL_CREATOR_USERNAME = "autogpt"


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


# Templates that shipped and were then folded into another entry. A seed run
# never deletes a template row — hires point at it through sourceTemplateId —
# so a retired one is archived instead: it leaves the Team page while the
# copies people already hired keep working. Safe to delete once every
# environment has been seeded past it.
RETIRED_TEMPLATES: list[str] = [
    # The senior sales package shipped as Blake for a day, then was folded into
    # Max, whose template it had been built to replace.
    "Blake",
]


async def _archive_retired_templates(keys: list[str], *, dry_run: bool) -> list[str]:
    """Archive the templates the catalog retires. Hires keep working: a hire
    points at its template through sourceTemplateId, so the row stays and only
    leaves the Team page. Returns the keys (or legacy names) archived."""
    rows = await prisma.models.Expert.prisma().find_many(
        where={
            "isTemplate": True,
            "isArchived": False,
            "ownerUserId": None,
            "OR": [
                {"templateKey": {"in": keys}},
                {"name": {"in": RETIRED_TEMPLATES}},
            ],
        }
    )
    if rows and not dry_run:
        await prisma.models.Expert.prisma().update_many(
            where={"id": {"in": [row.id for row in rows]}},
            data={"isArchived": True},
        )
    return sorted(row.templateKey or row.name for row in rows)


class RescopedTemplate(TypedDict):
    name: str
    # The role and identity the template shipped with before it was rescoped.
    # A hired copy still carrying both verbatim has never been edited by its
    # owner, so it is safe to move onto the new persona.
    old_role: str
    old_identity: str
    old_presentation: NotRequired[PresentationBaseline]


# One-off migrations for personas whose scope changed, not just their copy.
#
# ``_backfill_hired_copies`` deliberately leaves ``role`` and ``identity``
# alone: they drive live behaviour, and owners edit them through the Soul
# tools. That is right for a cosmetic roster edit, but a rescope would leave
# existing hires advertising the new bio while still behaving like the old
# persona — worse than either consistent outcome.
#
# So on a rescoped template the backfill moves the presentation and the
# persona together, and only for hires that are still recognisably the
# template's: see ``_backfill_hired_copies``. An owner who edited role or
# identity keeps the whole of the old persona, bio included, rather than half
# of each. Same shape as REMOVED_TEMPLATE_CADENCES, and just as safe to delete
# once every environment has been seeded past it.
RESCOPED_TEMPLATES: list[RescopedTemplate] = [
    {
        "name": "Max",
        "old_presentation": PresentationBaseline(
            jobTitle="Sales Development Rep",
            tagline="Finds your leads, their decision-makers, and their contact details.",
            bio="I'm a sales development expert who's built outbound pipelines for startups and mid-market teams, and I treat most pipeline problems as targeting problems in disguise — so I start by sharpening your ideal customer profile before I go hunting. From day one I can pull lists of businesses that fit that profile, surface the owner or decision-maker behind a company, and track down a contact's email address. Volume without fit is noise, and I say so plainly.",
            categories=["sales"],
        ),
        "old_role": "Sales",
        "old_identity": """You are Max, a sales development expert who has built outbound pipelines for startups and mid-market companies. You believe pipeline problems are usually targeting problems in disguise, so you start every engagement by sharpening the ideal customer profile: industry, size, trigger events, and the specific pain your product removes. Volume without fit is noise, and you say so plainly.

Your core work is prospecting and outreach preparation. You research accounts, surface decision makers, find verified contact details, and draft first-touch messages that reference something real about the prospect rather than a template with a name merged in. You keep outreach short, specific, and honest about why you are reaching out. You also help qualify inbound interest, separating genuine buying signals from curiosity.

You are rigorous about data quality. You flag when contact information looks stale, you never fabricate a prospect's details, and you mark your confidence level when a finding is inferred rather than confirmed. When a workflow returns a lead list, you review it against the ideal customer profile before presenting it, and you note which leads you would prioritize and why.""",
    },
    {
        "name": "Maria",
        "old_presentation": PresentationBaseline(
            jobTitle=None,
            tagline="Writes your LinkedIn posts, SEO articles, and webpage copy.",
            bio="I'm a senior marketing strategist — fifteen years across B2B SaaS and consumer brands — and I lead with positioning before tactics: who the customer is, what keeps them up at night, and why they'd pick you over doing nothing. From day one I can research and write LinkedIn posts, take an SEO blog article from research to a publish-ready draft, and rework the copy on your webpages to perform better in search. Everything ships in clear, confident prose with the jargon stripped out.",
            categories=["marketing", "content"],
        ),
        "old_role": "Marketing",
        "old_identity": """You are Maria, a senior marketing strategist with fifteen years of experience across B2B SaaS and consumer brands. You think in terms of positioning first: before any tactic, you want to know who the customer is, what keeps them up at night, and why they would choose this product over doing nothing. You write in clear, confident prose and you distrust jargon — if a headline could appear on any competitor's website, you rewrite it.

Your day-to-day work spans content strategy, social copy, email campaigns, and SEO-aware long-form writing. You draft LinkedIn posts, blog articles, and landing page copy that sound like a person wrote them, and you always tie a piece of content back to a measurable goal: signups, demos booked, or search rankings improved. When you are given a rough idea, you return an outline, three headline options, and a full draft.

You are direct about trade-offs. If a campaign idea is clever but off-brand, you say so and propose an alternative. You ask for the product's voice guidelines, target audience, and differentiators when they are missing, and you never invent customer claims or statistics. When you use a workflow, you treat its output as a first draft and refine it in the product's voice.""",
    },
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
                                # No isDeleted filter: soft-deleting the listing
                                # does not stop the hired copy's schedule, so
                                # the migration must still reach those rows.
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
        scheduler = get_scheduler_client()
        if owner_id not in live_by_owner:
            schedules = await scheduler.get_execution_schedules(
                user_id=owner_id, kind="graph"
            )
            live_by_owner[owner_id] = {s.id for s in schedules}
        if schedule_id not in live_by_owner[owner_id]:
            return True
        await scheduler.delete_schedule(schedule_id, user_id=owner_id)
        live_by_owner[owner_id].discard(schedule_id)
        return True
    except Exception as e:
        logger.warning(
            f"Could not delete schedule #{schedule_id} for user #{owner_id}; "
            f"keeping cadence for retry: {type(e).__name__}: {e}"
        )
        return False


async def _find_template(entry: RosterEntry) -> prisma.models.Expert | None:
    """The template row for a roster entry: by its catalog key, or, for a row
    seeded before keys existed and not yet keyed, by its display name."""
    template = await prisma.models.Expert.prisma().find_first(
        where={"isTemplate": True, "templateKey": entry["key"]}
    )
    if template is not None:
        return template
    return await prisma.models.Expert.prisma().find_first(
        where={
            "isTemplate": True,
            "templateKey": None,
            "ownerUserId": None,
            "name": entry["name"],
        },
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )


async def _upsert_template(
    entry: RosterEntry, previous: prisma.models.Expert | None
) -> prisma.models.Expert:
    fields = {
        "templateKey": entry["key"],
        "name": entry["name"],
        "role": entry["role"],
        "jobTitle": entry["job_title"],
        "tagline": entry["tagline"],
        "avatarUrl": resolve_avatar_url(entry["avatar_url"]),
        "identity": entry["identity"],
        "voicePreferences": encode_voice_preferences(
            entry["voice_preferences"], entry.get("voice_samples") or []
        ),
        "boundaries": entry["boundaries"],
        "bio": entry["bio"],
        "categories": validate_canonical_categories(entry["categories"]),
        "dayOne": SafeJson(encode_day_one(entry["day_one"])),
        "isArchived": False,
    }
    if previous is None:
        return await prisma.models.Expert.prisma().create(
            data={"isTemplate": True, **fields}
        )
    updated = await prisma.models.Expert.prisma().update(
        where={"id": previous.id}, data=fields
    )
    if updated is None:
        raise RuntimeError(f"Failed to update expert template '{entry['key']}'")
    return updated


_PRESENTATION_BACKFILL_BATCH_SIZE = 100


async def _backfill_hired_copies(
    template: prisma.models.Expert,
    previous: prisma.models.Expert | None = None,
) -> int:
    """Copy unchanged defaults with a concurrency guard; preserve owner edits.

    Rescope retries use recorded legacy defaults after the template advances,
    so presentation and behavior move together without overwriting owner edits.
    Names, skills and other behavioral settings are never cosmetic defaults.
    """
    if previous is None:
        return 0
    changed = 0
    rescope = next((r for r in RESCOPED_TEMPLATES if r["name"] == template.name), None)
    last_id: str | None = None
    while True:
        where: prisma.types.ExpertWhereInput = {
            "sourceTemplateId": template.id,
            "isTemplate": False,
        }
        if last_id is not None:
            where["id"] = {"gt": last_id}
        hires = await prisma.models.Expert.prisma().find_many(
            where=where,
            order={"id": "asc"},
            take=_PRESENTATION_BACKFILL_BATCH_SIZE,
        )
        for hire in hires:
            baseline: PresentationLike = previous
            if rescope and (hire.role, hire.identity) not in (
                (rescope["old_role"], rescope["old_identity"]),
                (previous.role, previous.identity),
                (template.role, template.identity),
            ):
                continue
            if (
                rescope
                and (hire.role, hire.identity)
                == (rescope["old_role"], rescope["old_identity"])
                and (previous.role, previous.identity) != (hire.role, hire.identity)
            ):
                legacy = rescope.get("old_presentation")
                if legacy is None:
                    continue
                baseline = legacy
            data = presentation_changes(hire, baseline, template)
            # A hire sitting on a picker preset may have chosen it, and the
            # row cannot say which. Leaving an older catalog image in place
            # costs less than overwriting a choice its owner made, so only
            # URLs the picker cannot produce get migrated. Templates keep the
            # broad match, since nobody edits those.
            if hire.avatarUrl not in PRESET_AVATAR_URLS:
                avatar_url = resolve_builtin_avatar_url(template.name, hire.avatarUrl)
                if avatar_url != hire.avatarUrl:
                    data["avatarUrl"] = avatar_url
            if rescope and (hire.role, hire.identity) != (
                template.role,
                template.identity,
            ):
                data.update(role=template.role, identity=template.identity)
            if not data:
                continue
            changed += await prisma.models.Expert.prisma().update_many(
                where={
                    "id": hire.id,
                    "sourceTemplateId": template.id,
                    "isTemplate": False,
                    "updatedAt": hire.updatedAt,
                },
                data=cast(prisma.types.ExpertUpdateManyMutationInput, data),
            )
        if len(hires) < _PRESENTATION_BACKFILL_BATCH_SIZE:
            return changed
        last_id = hires[-1].id


async def _sync_preloads(
    template_id: str,
    entry: RosterEntry,
    resolved_versions: Mapping[str, str] | None = None,
) -> None:
    existing = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={"expertId": template_id}
    )
    existing_by_version = {w.storeListingVersionId: w for w in existing}
    wanted: set[str] = set()
    unresolved = False
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
            unresolved = True
            continue
        wanted.add(version_id)
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
    await _prune_preloads(template_id, entry, existing, wanted, unresolved)


async def _prune_preloads(
    template_id: str,
    entry: RosterEntry,
    existing: list[prisma.models.ExpertWorkflow],
    wanted: set[str],
    unresolved: bool,
) -> None:
    """Drop template rows for workflows the roster no longer assigns.

    Without this the sync is create-only, so moving a workflow from one
    persona to another leaves it on both: the losing template keeps its row
    and every later hire still installs it.

    Template rows only — a hired copy is the user's, and deleting it would
    take a workflow out of someone's team. Existing hires therefore keep the
    workflow they were hired with, the same way ``_backfill_hired_copies``
    leaves hire-owned fields alone. Template rows also carry no schedule
    (``_install_preloads`` creates those per hire), so there is no live job
    to detach first.

    A slug that failed to resolve makes ``wanted`` incomplete, and pruning
    against it would delete a row that is still assigned. Skip the pass
    entirely in that case; ``_resolve_roster_preloads`` already fails the
    whole seed before any template is touched, so this only guards the
    ``resolved_versions=None`` path.
    """
    if unresolved:
        logger.warning(
            f"Skipping preload prune for expert '{entry['name']}': "
            "at least one roster slug did not resolve"
        )
        return
    stale = [w.id for w in existing if w.storeListingVersionId not in wanted]
    if not stale:
        return
    # Scoped by expertId as well as id: the ids came from a query already
    # filtered to this template, so the clause is redundant today, but it
    # keeps the only delete_many in this module from being able to reach
    # another persona's rows if the caller's `existing` ever widens.
    await prisma.models.ExpertWorkflow.prisma().delete_many(
        where={"id": {"in": stale}, "expertId": template_id}
    )
    logger.info(
        f"Removed {len(stale)} stale template preload(s) from '{entry['name']}'"
    )


async def _sync_routines(template_id: str, entry: RosterEntry) -> None:
    """Push the roster's routine proposals onto the template, keyed by slug.

    Template rows only. A hire's rows are handled by ``_sync_hired_routines``,
    which is far more cautious, because a routine on a hire may already be
    running on somebody's account.
    """
    existing = await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": template_id}
    )
    by_key = {row.key: row for row in existing if row.key is not None}
    wanted = {routine["key"] for routine in entry["routines"]}
    for routine in entry["routines"]:
        fields = _routine_fields(routine)
        current = by_key.get(routine["key"])
        if current is None:
            await prisma.models.ExpertRoutine.prisma().create(
                data=prisma.types.ExpertRoutineCreateInput(
                    expertId=template_id, key=routine["key"], **fields
                )
            )
        else:
            await prisma.models.ExpertRoutine.prisma().update(
                where={"id": current.id}, data=fields
            )
    stale = [row.id for row in existing if row.key not in wanted]
    if not stale:
        return
    await prisma.models.ExpertRoutine.prisma().delete_many(
        where={"id": {"in": stale}, "expertId": template_id}
    )
    logger.info(
        f"Removed {len(stale)} stale template routine(s) from '{entry['name']}'"
    )


def _routine_fields(
    routine: RoutineSeed,
) -> prisma.types.ExpertRoutineUpdateManyMutationInput:
    """The columns a roster entry owns on a template row.

    ``grantsCredentials`` is absent on purpose: it is never roster-declared, so
    a template row keeps the schema default of False and no roster edit can
    hand a seeded routine the keys to somebody's inbox.

    ``source`` is written rather than defaulted, because it is what decides
    that a hired copy of this row reaches nothing until its owner says so.
    """
    return {
        "title": routine["title"],
        "prompt": routine["prompt"],
        "crons": routine["crons"],
        "asks": routine["asks"],
        "sessionMode": prisma.enums.ExpertRoutineSession(routine["session_mode"]),
        "source": prisma.enums.ExpertRoutineSource.TEMPLATE,
    }


async def _sync_hired_routines(template_id: str, entry: RosterEntry) -> int:
    """Refresh routine proposals on hires — but only the untouched ones.

    A routine nobody has switched on and nobody has edited is still just an
    offer, so re-wording it or fixing its suggested hour is safe and reaches
    people who hired last month. Everything else is off limits: once a routine
    is running, or once its owner has changed a single thing about it, what it
    does is theirs and a roster edit must never silently rewrite it.

    New roster routines are not added to existing hires either. A hire's
    routine list is what that expert arrived with; growing it behind the
    owner's back would put unasked-for standing work on their team page.
    """
    if not entry["routines"]:
        return 0
    refreshed = 0
    for routine in entry["routines"]:
        refreshed += await prisma.models.ExpertRoutine.prisma().update_many(
            where={
                "key": routine["key"],
                "enabledAt": None,
                "customizedAt": None,
                "Expert": {"is": {"sourceTemplateId": template_id}},
            },
            data=_routine_fields(routine),
        )
    return refreshed


async def _resolve_roster_preloads(roster: list[RosterEntry]) -> dict[str, str]:
    slugs = {preload["slug"] for entry in roster for preload in entry["preloads"]}
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
            "before publishing the expert roster."
        )
    return resolved


async def _resolve_roster_skills(
    roster: list[RosterEntry], *, dry_run: bool = False
) -> dict[str, str]:
    """Listing id per bundled slug. A dry run tolerates slugs with no listing
    yet: the release that carries this roster also carries those packages,
    and a dry run of it has created nothing."""
    slugs = {slug for entry in roster for slug in entry["bundled_skills"]}
    if not slugs:
        return {}
    listings = await prisma.models.SkillListing.prisma().find_many(
        where={"slug": {"in": sorted(slugs)}, "isDeleted": False}
    )
    resolved = {listing.slug: listing.id for listing in listings}
    missing = sorted(slugs - resolved.keys())
    if missing and not dry_run:
        raise RuntimeError(
            f"Skills Hub is missing roster listings for: {', '.join(missing)}. "
            "Publish the skills catalog before the expert roster."
        )
    return resolved


async def _sync_bundled_skills(template_id: str, listing_ids: list[str]) -> None:
    await prisma.models.ExpertSkillListing.prisma().delete_many(
        where={"expertId": template_id, "skillListingId": {"not_in": listing_ids}}
    )
    for position, listing_id in enumerate(listing_ids):
        await prisma.models.ExpertSkillListing.prisma().upsert(
            where={
                "expertId_skillListingId": {
                    "expertId": template_id,
                    "skillListingId": listing_id,
                }
            },
            data={
                "create": {
                    "expertId": template_id,
                    "skillListingId": listing_id,
                    "position": position,
                },
                "update": {"position": position},
            },
        )


async def seed_roster(
    roster: list[RosterEntry],
    *,
    retired_keys: Iterable[str] = (),
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Upsert the roster templates with their preloads, routines and bundled
    skills, keyed by catalog key. Returns what changed, by key.

    Every preload and skill slug is resolved before any template is touched,
    so a bad roster fails whole. A dry run resolves and reports without
    writing.
    """
    resolved_versions = await _resolve_roster_preloads(roster)
    resolved_skills = await _resolve_roster_skills(roster, dry_run=dry_run)
    summary: dict[str, list[str]] = {"created": [], "updated": [], "retired": []}
    for entry in roster:
        previous = await _find_template(entry)
        summary["created" if previous is None else "updated"].append(entry["key"])
        if dry_run:
            continue
        template = await _upsert_template(entry, previous)
        await _sync_preloads(template.id, entry, resolved_versions)
        await _sync_routines(template.id, entry)
        await _sync_bundled_skills(
            template.id, [resolved_skills[slug] for slug in entry["bundled_skills"]]
        )
        refreshed = await _backfill_hired_copies(template, previous)
        routines = await _sync_hired_routines(template.id, entry)
        logger.info(
            f"Seeded expert template '{entry['key']}' (#{template.id}); "
            f"refreshed {refreshed} hired copies and {routines} untouched routine(s)"
        )
    summary["retired"] = await _archive_retired_templates(
        sorted(retired_keys), dry_run=dry_run
    )
    if summary["retired"]:
        logger.info(f"Archived retired template(s): {summary['retired']}")
    if not dry_run:
        await _clear_removed_cadences_bounded()
    return summary


# The cadence cleanup talks to the scheduler service, which the deploy job
# running the publisher cannot reach; its client would otherwise retry for
# far longer than a deploy should wait. Nothing else in the seed leaves the
# database.
_CADENCE_CLEANUP_TIMEOUT_S = 60


async def _clear_removed_cadences_bounded() -> None:
    try:
        await asyncio.wait_for(
            _clear_removed_cadences(), timeout=_CADENCE_CLEANUP_TIMEOUT_S
        )
    except (asyncio.TimeoutError, Exception):
        logger.warning(
            "Removed-cadence cleanup skipped this run (scheduler unreachable); "
            "the next publish retries it",
            exc_info=True,
        )
