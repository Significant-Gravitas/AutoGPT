"""Turning a stored expert into a downloadable package.

What leaves is decided here rather than by the manifest: the schema in
:mod:`package_model` has no field for memory, chats or workspace files, and this
module never looks for them. What it does strip is everything that only means
something inside this installation — the owner, the org and team the row is
tenanted to, the credential grants, the weekly budget, the library and schedule
ids a workflow was wired up with. A cron survives, because the cadence is the
creator's intent rather than a local id.

Where the avatar's bytes may come from is :mod:`package_avatar`'s problem.
"""

import logging

import prisma.models

from backend.api.features.experts.models import decode_day_one, decode_voice_preferences
from backend.api.features.experts.package_avatar import packaged_avatar
from backend.api.features.experts.package_model import (
    AVATAR_EXTENSIONS,
    MAX_MANIFEST_BYTES,
    MAX_PACKAGE_SKILLS,
    MAX_PACKAGE_WORKFLOWS,
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
    PackagedIdentity,
    PackagedSkill,
    PackagedSoul,
    PackagedWorkflow,
    expert_slug,
    manifest_json,
)
from backend.copilot.tools.skills import (
    SkillPackage,
    list_user_skills,
    read_user_skill_package,
    skill_slug,
)
from backend.data import graph as graph_db
from backend.data.graph import Graph

logger = logging.getLogger(__name__)

# Ownership and tenancy are the importer's to assign; a graph arriving with the
# exporter's still on it would claim the wrong owner on the way in.
_GRAPH_LOCAL_FIELDS = {"user_id", "organization_id", "team_id", "created_at"}


def package_filename(name: str) -> str:
    """The name a download is offered under."""
    return f"{expert_slug(name)}.expert.zip"


async def build_expert_package(row: prisma.models.Expert) -> ExpertPackage:
    """The whole expert as a package, from a row loaded with
    :data:`EXPORT_INCLUDE`."""
    description, samples = decode_voice_preferences(row.voicePreferences)
    packages, cards = await _skills(row)
    avatar, avatar_bytes = await packaged_avatar(row.avatarUrl)
    manifest = ExpertManifest(
        identity=PackagedIdentity(
            name=row.name,
            role=row.role,
            tagline=row.tagline,
            bio=row.bio,
            color=row.color,
            categories=list(row.categories),
        ),
        soul=PackagedSoul(
            identity=row.identity,
            voice_preferences=description,
            boundaries=row.boundaries,
            voice_samples=samples,
        ),
        day_one=decode_day_one(row.dayOne),
        skills=cards,
        workflows=await _workflows(row),
        avatar=avatar,
        tool_profile=row.toolProfile,
    )
    _within_the_manifest_cap(manifest)
    return ExpertPackage(
        manifest=manifest,
        skills=packages,
        avatar_bytes=avatar_bytes,
        avatar_mime=(
            AVATAR_EXTENSIONS[avatar.path.rsplit(".", 1)[1]]
            if avatar and avatar.path
            else None
        ),
    )


def _within_the_manifest_cap(manifest: ExpertManifest) -> None:
    """Refuse here rather than hand out a file our own reader would reject —
    the route owes a 413, not a download that fails on re-import."""
    size = len(manifest_json(manifest))
    if size > MAX_MANIFEST_BYTES:
        raise ExpertPackageError(
            f"expert.json would be {size} bytes; the limit is {MAX_MANIFEST_BYTES}",
            over_limit=True,
        )


async def _skills(
    row: prisma.models.Expert,
) -> tuple[dict[str, SkillPackage], list[PackagedSkill]]:
    """The expert's own skill folders. A card is only written for a skill whose
    files were actually read, because the reader refuses a manifest and a tree
    that disagree."""
    if not row.ownerUserId:
        # A roster template's skills live in a Skills Hub listing, not in any
        # user's workspace; serving those from the listing snapshot is B3.
        return {}, []
    packages: dict[str, SkillPackage] = {}
    cards: list[PackagedSkill] = []
    for skill in (await list_user_skills(row.ownerUserId, expert_id=row.id))[
        :MAX_PACKAGE_SKILLS
    ]:
        slug = skill_slug(skill.name)
        package = await read_user_skill_package(row.ownerUserId, slug, expert_id=row.id)
        if package is None:
            logger.info("Expert %s skill '%s' has no package to export", row.id, slug)
            continue
        packages[slug] = package
        cards.append(
            PackagedSkill(slug=slug, name=skill.name, description=skill.description)
        )
    return packages, cards


async def _workflows(row: prisma.models.Expert) -> list[PackagedWorkflow]:
    workflows = []
    for workflow in (row.Workflows or [])[:MAX_PACKAGE_WORKFLOWS]:
        packaged = await _workflow(workflow, row.ownerUserId)
        if packaged is None:
            logger.info(
                "Expert %s workflow %s has no source to export", row.id, workflow.id
            )
            continue
        workflows.append(packaged)
    return workflows


async def _workflow(
    row: prisma.models.ExpertWorkflow, owner_user_id: str | None
) -> PackagedWorkflow | None:
    """Both halves of a workflow's provenance, whenever the row has both.

    An import prefers the marketplace reference: it keeps the new owner on the
    published agent, with its updates and its creator's attribution. The graph
    travels alongside it anyway, so the file still restores in an installation
    where that listing does not exist. A roster template has no LibraryAgent,
    so it is the one case that carries the reference alone.
    """
    version = row.StoreListingVersion
    graph = await _exported_graph(row.LibraryAgent, owner_user_id)
    if version is None and graph is None:
        return None
    agent, listing = row.LibraryAgent, version.StoreListing if version else None
    profile = listing.CreatorProfile if listing else None
    return PackagedWorkflow(
        name=_first(
            version.name if version else None,
            agent.name if agent else None,
            graph.name if graph else None,
        ),
        description=_first(
            version.subHeading if version else None,
            agent.description if agent else None,
            graph.description if graph else None,
        ),
        store_listing_version_id=row.storeListingVersionId,
        store_listing_slug=listing.slug if listing else None,
        creator_username=profile.username if profile else None,
        graph=graph,
        schedule_cron=row.scheduleCron,
    )


def _first(*values: str | None) -> str:
    return next((value for value in values if value), "")


async def _exported_graph(
    agent: prisma.models.LibraryAgent | None, owner_user_id: str | None
) -> Graph | None:
    """The graph as an importer may have it: no credentials, no webhooks, and
    none of the ownership or tenancy that only means something here."""
    if agent is None or not owner_user_id:
        return None
    stored = await graph_db.get_graph(
        agent.agentGraphId,
        agent.agentGraphVersion,
        user_id=owner_user_id,
        for_export=True,
        include_subgraphs=True,
    )
    if stored is None:
        return None
    return Graph.model_validate(stored.model_dump(exclude=_GRAPH_LOCAL_FIELDS))
