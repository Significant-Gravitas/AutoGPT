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

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import decode_day_one, decode_voice_preferences
from backend.api.features.experts.package_avatar import packaged_avatar
from backend.api.features.experts.package_model import (
    AVATAR_EXTENSIONS,
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
    validate_expert_package,
    validate_packaged_skill,
)
from backend.api.features.store import skill_db, skill_model
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    SkillPackage,
    SkillPackageError,
    find_user_skill_slugs,
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

_Skills = tuple[dict[str, SkillPackage], list[PackagedSkill]]


class _Roster:
    """The skills admitted to a package so far.

    Each is held to its own caps and to the running count and size as it
    arrives, so an expert whose skills alone cannot fit is refused before the
    next package — or the avatar, or a graph — is read. Fifty stored skills
    that each fit could otherwise be read whole just to answer 413.
    """

    def __init__(self) -> None:
        self.packages: dict[str, SkillPackage] = {}
        self.cards: list[PackagedSkill] = []
        self.size_bytes = 0

    def admit(
        self, slug: str, name: str, description: str, package: SkillPackage
    ) -> None:
        validate_packaged_skill(slug, package)
        if len(self.packages) >= MAX_PACKAGE_SKILLS:
            # The folder is capped separately from the package, and a listing
            # can run over it; a package quietly missing skills is not a backup.
            raise ExpertPackageError(
                f"expert has more than {MAX_PACKAGE_SKILLS} skills; the limit is "
                f"{MAX_PACKAGE_SKILLS}",
                over_limit=True,
            )
        self.size_bytes += package.size_bytes
        if self.size_bytes > MAX_PACKAGE_BYTES:
            raise ExpertPackageError(
                f"package unpacks to at least {self.size_bytes} bytes in skills "
                f"alone; the limit is {MAX_PACKAGE_BYTES}",
                over_limit=True,
            )
        self.packages[slug] = package
        self.cards.append(PackagedSkill(slug=slug, name=name, description=description))


def package_filename(name: str) -> str:
    """The name a download is offered under."""
    return f"{expert_slug(name)}.expert.zip"


async def build_expert_package(
    row: prisma.models.Expert, *, user_id: str
) -> ExpertPackage:
    """The whole expert as a package, from a row loaded with
    :data:`EXPORT_INCLUDE`.

    *user_id* is the caller downloading it. For a roster template that decides
    which bundled Skills Hub listings travel, exactly as it decides which ones a
    hire installs.

    Refuses, with ``over_limit`` set, an expert that would not fit the package
    format's caps — the route owes a 413, not a download that fails on
    re-import.
    """
    description, samples = decode_voice_preferences(row.voicePreferences)
    packages, cards = await _skills(row, user_id)
    avatar, avatar_bytes = await packaged_avatar(row.avatarUrl, row.ownerUserId)
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
    package = ExpertPackage(
        manifest=manifest,
        skills=packages,
        avatar_bytes=avatar_bytes,
        avatar_mime=(
            AVATAR_EXTENSIONS[avatar.path.rsplit(".", 1)[1]]
            if avatar and avatar.path
            else None
        ),
    )
    validate_expert_package(package)
    return package


async def _skills(row: prisma.models.Expert, user_id: str) -> _Skills:
    """The expert's skills as whole packages, with a card for each.

    A hired expert's live in its own skill folder; a roster template's are the
    Skills Hub listings it bundles. Either way a card is only written for a
    skill whose package was actually produced, because the reader refuses a
    manifest and a tree that disagree.
    """
    roster = _Roster()
    if row.isTemplate:
        await _bundled_skills(row, user_id, roster)
    elif row.ownerUserId:
        await _owned_skills(row, row.ownerUserId, roster)
    return roster.packages, roster.cards


async def _owned_skills(
    row: prisma.models.Expert, owner_user_id: str, roster: _Roster
) -> None:
    """Every skill the expert owns, read from the folder it is actually stored
    under.

    The listing reports a skill by its frontmatter name, which a hand-written
    or legacy ``SKILL.md`` is free to spell differently from its folder — a
    skill named ``Deep Research`` under ``deep-research``. Deriving the folder
    from that name reads a path nothing was ever written to, so the skill would
    be dropped from the export without a word. The stored slug is resolved from
    the folder listing instead, and the derived name is only the fallback.
    """
    skills = await list_user_skills(owner_user_id, expert_id=row.id)
    stored = await find_user_skill_slugs(
        owner_user_id, [skill.name for skill in skills], expert_id=row.id
    )
    for skill in skills:
        slug = stored.get(skill.name.strip().lower()) or skill_slug(skill.name)
        try:
            package = await read_user_skill_package(
                owner_user_id, slug, expert_id=row.id
            )
        except SkillPackageError as exc:
            # A stored tree the skill download would refuse to serve whole.
            raise ExpertPackageError(
                f"skill '{slug[:120]}': {exc}", over_limit=exc.over_limit
            )
        if package is None:
            logger.info("Expert %s skill '%s' has no package to export", row.id, slug)
            continue
        roster.admit(slug, skill.name, skill.description, package)


async def _bundled_skills(
    row: prisma.models.Expert, user_id: str, roster: _Roster
) -> None:
    """A roster template's skills, read the way a hire installs them: only the
    live listings, only behind the Skills Hub flag for *user_id*, and rendered
    as the install would store them — so a listing a hire would skip is left
    out of the package too."""
    for listing in await experts_db.bundled_skill_listings(user_id, row.id):
        try:
            skill, package = skill_db.installable_skill(listing)
        except ValueError as exc:
            logger.info(
                "Template %s bundled skill '%s' cannot be packaged: %s",
                row.id,
                listing.slug[:120],
                exc,
            )
            continue
        if skill.name in roster.packages:
            continue
        # The install names the stored skill after the listing's slug, so that
        # is the folder; the card carries the listing's own name so a reviewer
        # reads "Web Scraper" rather than "web-scraper".
        display_name = skill_model.active_version(listing).name
        roster.admit(skill.name, display_name, skill.description, package)


async def _workflows(row: prisma.models.Expert) -> list[PackagedWorkflow]:
    workflows = []
    for workflow in row.Workflows or []:
        packaged = await _workflow(workflow, row.ownerUserId)
        if packaged is None:
            logger.info(
                "Expert %s workflow %s has no source to export", row.id, workflow.id
            )
            continue
        workflows.append(packaged)
        if len(workflows) > MAX_PACKAGE_WORKFLOWS:
            # Installing has no such cap, so a valid expert can be over it; a
            # package quietly missing runnable workflows is not a backup, and
            # the graphs of the rest are not worth reading to say so.
            raise ExpertPackageError(
                f"expert has more than {MAX_PACKAGE_WORKFLOWS} exportable "
                f"workflows; the limit is {MAX_PACKAGE_WORKFLOWS}",
                over_limit=True,
            )
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
