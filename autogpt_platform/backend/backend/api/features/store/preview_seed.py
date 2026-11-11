"""Publish the synthetic marketplace in :mod:`.preview_seed_content`.

Preview environments start from an empty database, so their store has nothing
to browse. This puts a small marketplace there the way real listings arrive:
each creator gets a profile and a saved workflow, submits it, and an admin
approves it; then reviews land and the store's materialized views refresh.
Skills go through the same submit-and-approve path.

Identities come from the content (uuid5), so every run addresses the same
rows. A run creates what is missing, approves a submission an earlier run left
pending, and leaves everything else as it is: a listing someone in the preview
has since rejected or deleted stays rejected or deleted.

It needs only the database. Approval skips the creator's verdict email, which
would need the message queue and has nobody to reach; the listing's search
embedding is generated if an embedding provider is configured, as on any
approval, and skipped otherwise.
"""

import logging
import re
import uuid
from collections.abc import Iterable

import prisma.enums
import prisma.models
from pydantic import BaseModel, ConfigDict

from backend.api.features.orgs.db import get_user_default_team
from backend.blocks.io import (
    AgentLongTextInputBlock,
    AgentOutputBlock,
    AgentShortTextInputBlock,
)
from backend.blocks.maths import CalculatorBlock, CountItemsBlock
from backend.blocks.text import FillTextTemplateBlock, TextReplaceBlock, TextSplitBlock
from backend.copilot.tools.skills import (
    ParsedSkill,
    canonicalize_skill,
    parse_skill_markdown,
    validate_skill_content,
)
from backend.data.block import initialize_blocks
from backend.data.db import execute_raw_with_schema
from backend.data.graph import Graph, Link, Node, create_graph, make_graph_model
from backend.data.org_migration import ensure_personal_org
from backend.util.settings import AppEnvironment, Settings

from . import db as store_db
from . import model as store_model
from . import skill_model, skill_submission_db
from .preview_seed_content import (
    ADMIN,
    CREATORS,
    LISTINGS,
    REVIEW_COMMENTS,
    REVIEWERS,
    SKILLS,
    CalculatorWorkflow,
    CountWorkflow,
    ReplaceWorkflow,
    SeedListing,
    SeedSkill,
    SeedUser,
    TemplateWorkflow,
    Workflow,
)

logger = logging.getLogger(__name__)

# Fixed forever: every id the seed writes is derived from it.
SEED_NAMESPACE = uuid.UUID("6f1c2d8e-3b4a-4c5d-9e7f-a1b2c3d4e5f6")

WORKFLOW_BLOCKS = (
    AgentShortTextInputBlock,
    AgentLongTextInputBlock,
    AgentOutputBlock,
    FillTextTemplateBlock,
    CalculatorBlock,
    TextReplaceBlock,
    TextSplitBlock,
    CountItemsBlock,
)

APPROVAL_COMMENT = "Approved for the preview marketplace."
SEED_NOTE = "Seeded by seed-preview-marketplace."


class SeedRefused(RuntimeError):
    """Synthetic data must never reach this environment."""


class SeedSummary(BaseModel):
    users: int = 0
    # Per listing or skill slug: "created", "approved" (a submission an earlier
    # run left pending) or "existing" (left as it was).
    listings: dict[str, str] = {}
    skills: dict[str, str] = {}
    # Reviews this run added; one that already exists is never rewritten.
    reviews: int = 0
    failed: dict[str, str] = {}


class _SeededUser(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    org_id: str | None
    team_id: str | None


def seed_id(*parts: str) -> str:
    return str(uuid.uuid5(SEED_NAMESPACE, "/".join(parts)))


def seed_user_id(user: SeedUser) -> str:
    return seed_id("user", user.key)


def seed_graph_id(listing: SeedListing) -> str:
    return seed_id("graph", listing.slug)


def all_users() -> tuple[SeedUser, ...]:
    return (*CREATORS, *REVIEWERS, ADMIN)


def reviewer_keys(listing: SeedListing) -> list[str]:
    """Who reviews a listing, in order: the reviewers, then the other creators."""
    return [user.key for user in REVIEWERS] + [
        user.key for user in CREATORS if user.key != listing.creator
    ]


def ensure_seedable() -> None:
    """Raise :class:`SeedRefused` unless ``APP_ENV`` is ``local`` or this is a
    PR preview.

    A preview runs with ``APP_ENV=dev``, like the shared dev deployment and
    self-hosted installs; its Sentry environment, ``pr-<n>``, is what sets it
    apart. ``local`` is taken at its word: it is the developer's setting, not
    a check of where ``DATABASE_URL`` points.
    """
    config = Settings().config
    if config.app_env == AppEnvironment.LOCAL:
        return
    if config.app_env != AppEnvironment.PRODUCTION and re.fullmatch(
        r"pr-\d+", config.sentry_environment
    ):
        return
    raise SeedRefused(
        f"refusing to seed synthetic data with APP_ENV={config.app_env.value}: "
        "only APP_ENV=local and pr-<n> previews are seeded"
    )


async def seed_marketplace() -> SeedSummary:
    ensure_seedable()
    summary = SeedSummary()
    await _ensure_workflow_blocks()

    users = {user.key: await _ensure_user(user) for user in all_users()}
    summary.users = len(users)
    admin_id = users[ADMIN.key].id

    for listing in LISTINGS:
        try:
            outcome, live_version_id = await _ensure_listing(
                listing, users[listing.creator], admin_id
            )
            summary.listings[listing.slug] = outcome
            if live_version_id:
                summary.reviews += await _ensure_reviews(
                    listing,
                    live_version_id,
                    [users[key].id for key in reviewer_keys(listing)],
                )
        except Exception as e:
            logger.exception(f"Could not seed listing '{listing.slug}'")
            summary.failed[f"listing:{listing.slug}"] = str(e)

    for skill in SKILLS:
        try:
            summary.skills[skill.slug] = await _ensure_skill(
                skill, users[skill.creator].id, admin_id
            )
        except Exception as e:
            logger.exception(f"Could not seed skill '{skill.slug}'")
            summary.failed[f"skill:{skill.slug}"] = str(e)

    # Ratings and run counts come from these views; pg_cron refreshes them
    # every 15 minutes where it is installed, and nowhere else.
    await execute_raw_with_schema(
        "SELECT {schema_prefix}refresh_store_materialized_views()"
    )
    return summary


async def _ensure_workflow_blocks() -> None:
    """Workflow nodes reference ``AgentBlock`` rows, which the API server
    writes when it starts. The seed can run before it ever has."""
    block_ids = [block_cls().id for block_cls in WORKFLOW_BLOCKS]
    stored = await prisma.models.AgentBlock.prisma().count(
        where={"id": {"in": block_ids}}
    )
    if stored < len(block_ids):
        await initialize_blocks()


async def _ensure_user(user: SeedUser) -> _SeededUser:
    user_id = seed_user_id(user)
    await prisma.models.User.prisma().upsert(
        where={"id": user_id},
        data={
            "create": {
                "id": user_id,
                "email": user.email,
                "name": user.name,
                # Synthetic addresses: nothing should try to mail them.
                "briefingFrequency": prisma.enums.BriefingFrequency.OFF,
                "alertsEnabled": False,
                "notifyOnStoreVerdict": False,
                "maxEmailsPerDay": 0,
            },
            "update": {},
        },
    )
    profile = await prisma.models.Profile.prisma().find_unique(
        where={"userId": user_id}
    )
    if profile is None:
        await store_db.update_profile(
            user_id,
            store_model.ProfileUpdateRequest(
                username=user.username,
                name=user.name,
                description=user.description,
                links=list(user.links),
            ),
        )
    # update_profile leaves a new profile's avatar NULL, but the Creator view's
    # avatar_url is non-null, so one NULL fails the whole creators list. One
    # write finishes the profile: a run that stopped before it finishes it next
    # time, and a profile it already finished keeps any later edit.
    await prisma.models.Profile.prisma().update_many(
        where={"userId": user_id, "avatarUrl": None},
        data={"avatarUrl": "", "isFeatured": user.featured},
    )
    await ensure_personal_org(user_id)
    org_id, team_id = await get_user_default_team(user_id)
    return _SeededUser(id=user_id, org_id=org_id, team_id=team_id)


# (role, block id, input defaults), laid out left to right.
_Step = tuple[str, str, dict]
# (source role, source pin, sink role, sink pin)
_Wire = tuple[str, str, str, str]


def build_graph(listing: SeedListing) -> Graph:
    """The listing's workflow: input, one to three processing nodes, output."""

    def node_id(role: str) -> str:
        return seed_id("node", listing.slug, role)

    workflow = listing.workflow
    if isinstance(workflow, (ReplaceWorkflow, CountWorkflow)) or (
        isinstance(workflow, TemplateWorkflow) and workflow.long_input
    ):
        input_block = AgentLongTextInputBlock()
    else:
        input_block = AgentShortTextInputBlock()
    steps, wiring = _processing(workflow)
    layout: list[_Step] = [
        (
            "input",
            input_block.id,
            {
                "name": workflow.input_name,
                "title": workflow.input_title,
                "value": workflow.example,
            },
        ),
        *steps,
        ("output", AgentOutputBlock().id, {"name": workflow.output_name}),
    ]
    return Graph(
        id=seed_graph_id(listing),
        version=1,
        is_active=True,
        name=listing.name,
        description=listing.description,
        instructions=listing.instructions,
        recommended_schedule_cron=listing.recommended_schedule_cron,
        nodes=[
            Node(
                id=node_id(role),
                block_id=block_id,
                input_default=input_default,
                metadata={"position": {"x": column * 450, "y": 0}},
            )
            for column, (role, block_id, input_default) in enumerate(layout)
        ],
        links=[
            Link(
                id=seed_id("link", listing.slug, source, sink, sink_pin),
                source_id=node_id(source),
                source_name=source_pin,
                sink_id=node_id(sink),
                sink_name=sink_pin,
                # Input blocks emit a static value; the save path marks their
                # links static whatever the caller sends.
                is_static=source == "input",
            )
            for source, source_pin, sink, sink_pin in wiring
        ],
    )


def _processing(workflow: Workflow) -> tuple[list[_Step], list[_Wire]]:
    """The nodes between a workflow's input and output, and every link."""
    if isinstance(workflow, TemplateWorkflow):
        return [
            (
                "step",
                FillTextTemplateBlock().id,
                {"format": workflow.template, "values": {}},
            )
        ], [
            ("input", "result", "step", f"values_#_{workflow.input_name}"),
            ("step", "output", "output", "value"),
        ]
    if isinstance(workflow, CalculatorWorkflow):
        return [
            (
                "step",
                CalculatorBlock().id,
                {"operation": workflow.operation, "b": workflow.operand},
            )
        ], [
            ("input", "result", "step", "a"),
            ("step", "result", "output", "value"),
        ]
    if isinstance(workflow, ReplaceWorkflow):
        return [
            ("step", TextReplaceBlock().id, {"old": workflow.old, "new": workflow.new})
        ], [
            ("input", "result", "step", "text"),
            ("step", "output", "output", "value"),
        ]
    # Drop blank lines first, so a trailing newline isn't an item.
    lines = f"{workflow.input_name}.splitlines() | map('trim') | select"
    return [
        (
            "tidy",
            FillTextTemplateBlock().id,
            {"format": f"{{{{ {lines} | join('\\n') }}}}", "values": {}},
        ),
        ("split", TextSplitBlock().id, {"delimiter": "\n", "strip": True}),
        ("step", CountItemsBlock().id, {}),
    ], [
        ("input", "result", "tidy", f"values_#_{workflow.input_name}"),
        ("tidy", "output", "split", "text"),
        ("split", "texts", "step", "collection"),
        ("step", "count", "output", "value"),
    ]


async def _ensure_graph(listing: SeedListing, owner: _SeededUser) -> tuple[str, int]:
    # The seed's own version: a later one is someone's edit, which the seed
    # must not submit and approve unreviewed.
    seeded = await prisma.models.AgentGraph.prisma().find_unique(
        where={"graphVersionId": {"id": seed_graph_id(listing), "version": 1}}
    )
    if seeded is not None:
        return seeded.id, seeded.version
    graph = make_graph_model(build_graph(listing), owner.id)
    graph.validate_graph(for_run=False)
    created = await create_graph(
        graph, owner.id, organization_id=owner.org_id, team_id=owner.team_id
    )
    return created.id, created.version


async def _ensure_listing(
    listing: SeedListing, owner: _SeededUser, admin_id: str
) -> tuple[str, str | None]:
    """Returns the outcome and the version reviews belong on, if any."""
    graph_id, graph_version = await _ensure_graph(listing, owner)
    existing = await prisma.models.StoreListing.prisma().find_unique(
        where={"agentGraphId": graph_id}, include={"Versions": True}
    )
    if existing is not None:
        # Only a submission of the seed's own workflow version: one of a later
        # version is someone's edit, even when its form kept the seed's note.
        pending = _unreviewed_pending(
            v for v in existing.Versions or [] if v.agentGraphVersion == graph_version
        )
        # An earlier run stopped after submitting, or partway through the
        # approval: it points the listing at the version before marking the
        # version approved. Either way the version is still pending.
        if (
            existing.isDeleted
            or pending is None
            or (existing.hasApprovedVersion and existing.activeVersionId != pending.id)
        ):
            active = next(
                (
                    v
                    for v in existing.Versions or []
                    if v.id == existing.activeVersionId
                ),
                None,
            )
            # Reviews belong only on a live version the seed itself approved.
            seeded_live = (
                not existing.isDeleted
                and existing.hasApprovedVersion
                and active is not None
                and active.reviewerId == admin_id
            )
            return "existing", active.id if seeded_live and active else None
        version_id, outcome = pending.id, "approved"
    else:
        request = store_model.StoreSubmissionRequest(
            graph_id=graph_id,
            graph_version=graph_version,
            slug=listing.slug,
            name=listing.name,
            sub_heading=listing.sub_heading,
            description=listing.description,
            instructions=listing.instructions,
            categories=list(listing.categories),
            recommended_schedule_cron=listing.recommended_schedule_cron,
            changes_summary=SEED_NOTE,
        )
        submission = await store_db.create_store_submission(
            user_id=owner.id,
            organization_id=owner.org_id,
            **request.model_dump(exclude_none=True),
        )
        if submission.listing_version_id is None:
            raise ValueError(f"Submission of '{listing.slug}' returned no version")
        version_id, outcome = submission.listing_version_id, "created"

    # Featured while still pending, so a run that stops after approving has
    # nothing left to set, and a version already live keeps an admin's choice.
    if listing.featured:
        await prisma.models.StoreListingVersion.prisma().update(
            where={"id": version_id}, data={"isFeatured": True}
        )
    await store_db.review_store_submission(
        store_listing_version_id=version_id,
        is_approved=True,
        external_comments=APPROVAL_COMMENT,
        internal_comments=SEED_NOTE,
        reviewer_id=admin_id,
        notify_creator=False,
    )
    return outcome, version_id


def _unreviewed_pending(
    versions: Iterable[
        prisma.models.StoreListingVersion | prisma.models.SkillListingVersion
    ],
):
    """A pending version the seed submitted and nobody has reviewed: what a
    run that stopped between submitting and approving leaves behind. Anyone
    else's submission waits for a real review."""
    return next(
        (
            version
            for version in versions
            if version.submissionStatus == prisma.enums.SubmissionStatus.PENDING
            and not version.isDeleted
            and version.reviewerId is None
            and version.changesSummary == SEED_NOTE
        ),
        None,
    )


async def _ensure_reviews(
    listing: SeedListing, version_id: str, reviewer_ids: list[str]
) -> int:
    """Adds the reviews the version is missing; returns how many."""
    if len(listing.reviews) > len(reviewer_ids):
        raise ValueError(f"'{listing.slug}' has more reviews than reviewers")
    added = 0
    for score, reviewer_id in zip(listing.reviews, reviewer_ids):
        existing = await prisma.models.StoreListingReview.prisma().find_unique(
            where={
                "storeListingVersionId_reviewByUserId": {
                    "storeListingVersionId": version_id,
                    "reviewByUserId": reviewer_id,
                }
            }
        )
        if existing is not None:
            continue
        review = store_model.StoreReviewCreate(
            store_listing_version_id=version_id,
            score=score,
            comments=REVIEW_COMMENTS[score],
        )
        await store_db.create_store_review(
            user_id=reviewer_id,
            store_listing_version_id=review.store_listing_version_id,
            score=review.score,
            comments=review.comments,
        )
        added += 1
    return added


def parse_seed_skill(skill: SeedSkill) -> ParsedSkill:
    """The skill as a library would hold it: parsed, normalised and checked
    against the limits every saved skill meets."""
    parsed = parse_skill_markdown(skill.skill_md, fallback_name=skill.slug)
    if parsed is None:
        raise ValueError(f"'{skill.slug}' is not a valid SKILL.md")
    parsed = canonicalize_skill(parsed)
    validate_skill_content(
        parsed.description, parsed.body, parsed.triggers, name=parsed.name
    )
    if parsed.name != skill.slug:
        raise ValueError(f"'{skill.slug}' is named '{parsed.name}' in its SKILL.md")
    return parsed


async def _ensure_skill(skill: SeedSkill, owner_id: str, admin_id: str) -> str:
    existing = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": skill.slug}, include={"Versions": True}
    )
    if existing is None:
        submission = await skill_submission_db.submit_skill_package(
            owner_id,
            skill_model.SkillSubmissionRequest(
                skill_name=skill.slug,
                categories=list(skill.categories),
                changes_summary=SEED_NOTE,
            ),
            parse_seed_skill(skill),
            files=[],
        )
        version_id, outcome = submission.skill_listing_version_id, "created"
    else:
        pending = _unreviewed_pending(existing.Versions or [])
        if (
            existing.owningUserId != owner_id
            or existing.isDeleted
            or existing.hasApprovedVersion
            or pending is None
        ):
            return "existing"
        version_id, outcome = pending.id, "approved"

    await skill_submission_db.review_skill_submission(
        version_id,
        is_approved=True,
        reviewer_id=admin_id,
        comments=APPROVAL_COMMENT,
        internal_comments=SEED_NOTE,
    )
    return outcome
