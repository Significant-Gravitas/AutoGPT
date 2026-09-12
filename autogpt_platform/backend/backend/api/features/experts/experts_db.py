import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from typing import Literal, cast
from zoneinfo import ZoneInfo

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types
from prisma.enums import ResourceVisibility
from pydantic import BaseModel, JsonValue, ValidationError

from backend.api.features.experts import raise_attachments, scheduling

# Re-exported so `db_accessors.experts_db()` resolves the same attribute name
# on both branches: the module here, and the RPC client stub in db_manager.
from backend.api.features.experts.credential_counts import expert_credential_providers
from backend.api.features.experts.credentials import (
    expert_allowed_credential_ids as expert_allowed_credential_ids,
)
from backend.api.features.experts.errors import (
    ACTIVE_EXPERT_LIMIT,
    LIFETIME_RAISED_EXPERT_LIMIT,
    ExpertHireUnavailableError,
    ExpertLimitExceededError,
    ExpertPodLimitReachedError,
    ExpertPodNameTakenError,
    ExpertPodNotFoundError,
    ExpertTemplateNotFoundError,
    RaisedExpertLifetimeLimitExceededError,
)
from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertActivity,
    ExpertActivityDay,
    ExpertBundledSkill,
    ExpertIdentity,
    ExpertPod,
    ExpertRun,
    ExpertRunSource,
    ExpertRunStatus,
    ExpertSoulFieldsPatch,
    ExpertSoulUpdate,
    ExpertTemplate,
    ExpertWorkflowRef,
    HireResult,
    RaiseAttachment,
    RaiseResult,
    decode_day_one,
    decode_voice_preferences,
)
from backend.api.features.experts.workflow_chain import (
    build_workflow_chain,
    integration_providers,
)
from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.features.orgs.db import get_user_default_team
from backend.api.features.store import skill_db
from backend.api.features.store.categories import category_match_values
from backend.blocks import get_output_block_ids
from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME, run_link
from backend.copilot.tools.skills import (
    BuiltInSkillError,
    SkillNotFoundError,
    copy_skill_to_expert,
    delete_user_skill,
    find_user_skill_slugs,
    get_default_skill_with_body,
    skill_name_key,
)
from backend.data.db import prisma as db_client
from backend.data.db import query_raw_with_schema, transaction
from backend.data.expert_attribution import (
    resolve_attributable_expert as resolve_attributable_expert_row,
)
from backend.data.expert_run_output import (
    OutputType,
    classify_run_output,
    reconstruct_run_outputs,
)
from backend.data.expert_spend import get_weekly_spend
from backend.data.model import NodeExecutionStats
from backend.data.user import get_user_by_id
from backend.util import type as type_utils
from backend.util.exceptions import (
    ConflictError,
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
    ExpertWriteNotReadableError,
    NotFoundError,
)
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)


def _raised_identity(name: str) -> str:
    # f-string, not str.format on a template: user names may contain { or },
    # which str.format would choke on.
    return f"I'm {name}, raised by you. I learn how you work and grow with you."


# Postgres promises no row order without this, so the profile's workflow grid
# could reshuffle between loads; createdAt keeps the roster's authored order.
_WORKFLOW_ORDER = [{"createdAt": "asc"}, {"id": "asc"}]

_WORKFLOW_ROW_INCLUDE: prisma.types.ExpertWorkflowInclude = {
    # AgentGraph carries the name/description of a user-created library agent;
    # LibraryAgent.name is only populated from a marketplace snapshot.
    "LibraryAgent": {"include": {"AgentGraph": {"include": {"Nodes": True}}}},
    "StoreListingVersion": True,
}
_WORKFLOW_INCLUDE = {
    "Workflows": {"include": _WORKFLOW_ROW_INCLUDE, "order_by": _WORKFLOW_ORDER}
}
_ROSTER_WORKFLOW_INCLUDE: prisma.types.ExpertInclude = {
    "Workflows": {
        "include": {
            "LibraryAgent": {"include": {"AgentGraph": True}},
            "StoreListingVersion": True,
        },
        "order_by": _WORKFLOW_ORDER,
    }
}
# A template workflow has no LibraryAgent — that row is created at hire time —
# so its chain has to come from the listing's own graph. Without it the
# marketplace profile cannot say what a workflow connects to.
_TEMPLATE_WORKFLOW_INCLUDE: prisma.types.ExpertInclude = {
    "Workflows": {
        "include": {
            "LibraryAgent": {"include": {"AgentGraph": True}},
            "StoreListingVersion": {
                "include": {"AgentGraph": {"include": {"Nodes": True}}}
            },
        },
        "order_by": _WORKFLOW_ORDER,
    }
}
_MAX_EXPERT_RUNS = 20
# One year: the window the at-a-glance activity graph draws.
EXPERT_ACTIVITY_DAYS = 365

FirstJobUnavailableError = raise_attachments.RaiseAttachmentUnavailableError


def _to_workflow_ref(row: prisma.models.ExpertWorkflow) -> ExpertWorkflowRef:
    listing = row.StoreListingVersion
    library_agent = row.LibraryAgent
    # A listing always carries both name and description (non-null columns), so
    # the pair is taken from one source or the other — never mixed, which would
    # pair a published title with the creator's private description.
    if listing is not None:
        name, description = listing.name, listing.description
    elif library_agent is not None:
        name, description = _library_agent_labels(library_agent)
    else:
        name, description = None, None
    nodes = _chain_nodes(row)
    return ExpertWorkflowRef(
        id=row.id,
        store_listing_version_id=row.storeListingVersionId,
        library_agent_id=row.libraryAgentId,
        graph_id=library_agent.agentGraphId if library_agent else None,
        name=name,
        description=description,
        schedule_cron=row.scheduleCron,
        schedule_id=row.scheduleId,
        chain=build_workflow_chain(nodes),
        integration_providers=integration_providers(nodes),
    )


def _chain_nodes(
    row: prisma.models.ExpertWorkflow,
) -> Sequence[prisma.models.AgentNode]:
    """The graph whose blocks the chain summarises: the hire's own library
    agent, or the marketplace listing behind a template that has none yet."""
    library_graph = row.LibraryAgent.AgentGraph if row.LibraryAgent else None
    if library_graph and library_graph.Nodes:
        return library_graph.Nodes
    listing_graph = (
        row.StoreListingVersion.AgentGraph if row.StoreListingVersion else None
    )
    if listing_graph and listing_graph.Nodes:
        return listing_graph.Nodes
    return []


def _library_agent_labels(
    row: prisma.models.LibraryAgent,
) -> tuple[str | None, str | None]:
    """Name/description of a library agent, mirroring ``LibraryAgent.from_db``.

    The columns hold a marketplace snapshot and are NULL on a user's own agent,
    whose real title lives on the graph.
    """
    graph = row.AgentGraph
    return (
        row.name if row.name is not None else (graph.name if graph else None),
        (
            row.description
            if row.description is not None
            else (graph.description if graph else None)
        ),
    )


def _to_model(
    row: prisma.models.Expert,
    latest_run: prisma.models.AgentGraphExecution | None = None,
    weekly_spend: int = 0,
) -> Expert:
    """Translate the overloaded ``voicePreferences`` column safely.

    Template rows store an internal ``{description, samples}`` JSON envelope
    so the hire flow can present choices. Hired rows must store only the final
    plain-text preference that is safe to render in prompts. Keep this branch
    on ``isTemplate`` until those representations have separate columns.
    """
    if row.isTemplate:
        voice_preferences, voice_samples = decode_voice_preferences(
            row.voicePreferences
        )
    else:
        voice_preferences, voice_samples = row.voicePreferences, []
    return Expert(
        id=row.id,
        name=row.name,
        avatar_url=row.avatarUrl,
        color=row.color,
        role=row.role,
        tagline=row.tagline,
        bio=row.bio,
        skills=row.skills or [],
        categories=row.categories or [],
        identity=row.identity,
        voice_preferences=voice_preferences,
        voice_samples=voice_samples,
        day_one=decode_day_one(row.dayOne),
        boundaries=row.boundaries,
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=row.isTemplate,
        source_template_id=row.sourceTemplateId,
        is_archived=row.isArchived,
        workflows=[_to_workflow_ref(w) for w in row.Workflows or []],
        last_run_at=latest_run.createdAt if latest_run else None,
        last_run_status=(str(latest_run.executionStatus) if latest_run else None),
        weekly_budget=scheduling.effective_weekly_budget(row),
        weekly_spend=weekly_spend,
        schedules_paused_at=row.schedulesPausedAt,
        pod_id=row.podId,
    )


async def _latest_runs(
    expert_ids: list[str],
) -> dict[str, prisma.models.AgentGraphExecution]:
    """Latest execution per expert, one indexed query via Prisma distinct."""
    if not expert_ids:
        return {}
    rows = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={"expertId": {"in": expert_ids}, "isDeleted": False},
        order=[{"expertId": "asc"}, {"createdAt": "desc"}],
        distinct=["expertId"],
    )
    return {row.expertId: row for row in rows if row.expertId is not None}


async def list_templates(
    search_query: str | None = None,
    category: str | None = None,
) -> list[Expert]:
    rows = await prisma.models.Expert.prisma().find_many(
        where=_template_where(search_query, category),
        include=_TEMPLATE_WORKFLOW_INCLUDE,
    )
    return [_to_model(row) for row in rows]


def _template_where(
    search_query: str | None, category: str | None
) -> prisma.types.ExpertWhereInput:
    where: prisma.types.ExpertWhereInput = {"isTemplate": True, "isArchived": False}
    if category:
        # Not `category_filter_values`: with the canonical-category setting
        # on, that one hides uncategorised experts from the unfiltered roster.
        where["categories"] = {"has_some": category_match_values(category)}
    if search_query and (needle := search_query.strip()):
        where["OR"] = [
            {"name": {"contains": needle, "mode": "insensitive"}},
            {"role": {"contains": needle, "mode": "insensitive"}},
            {"tagline": {"contains": needle, "mode": "insensitive"}},
            {"bio": {"contains": needle, "mode": "insensitive"}},
        ]
    return where


async def with_bundled_skills(
    templates: list[Expert], user_id: str | None
) -> list[ExpertTemplate]:
    """Attach to each template the live Skills Hub listings it bundles —
    exactly what ``hire_expert`` installs."""
    bundled = await _live_bundled_skills(user_id, [t.id for t in templates])
    return [
        ExpertTemplate(
            **template.model_dump(), bundled_skills=bundled.get(template.id, [])
        )
        for template in templates
    ]


async def _live_bundled_skills(
    user_id: str | None, template_ids: list[str]
) -> dict[str, list[ExpertBundledSkill]]:
    """Per template id, the live Hub listings it bundles, in roster order."""
    # The Hub routes' key, so nothing is linked or installed that would 404.
    if not await is_feature_enabled(Flag.SKILLS_HUB, user_id or "anonymous"):
        return {}
    rows = await prisma.models.ExpertSkillListing.prisma().find_many(
        where={"expertId": {"in": template_ids}}, order={"position": "asc"}
    )
    live = await skill_db.get_live_skills(sorted({r.skillListingId for r in rows}))
    return {
        template_id: [
            ExpertBundledSkill(
                id=row.skillListingId,
                slug=skill.slug,
                name=skill.name,
                description=skill.description,
            )
            for row in rows
            if row.expertId == template_id and (skill := live.get(row.skillListingId))
        ]
        for template_id in template_ids
    }


# Ceiling on in-flight Redis reads inside ``_weekly_spends``. The roster is
# user-controlled and unbounded, so an uncapped ``gather`` would ask the shared
# Redis pool for one connection per hired expert on every team-page load.
_WEEKLY_SPEND_READ_CONCURRENCY = 10


async def _weekly_spends(expert_ids: list[str]) -> dict[str, int]:
    """Weekly spend per expert, one Redis read each, run concurrently.

    A read that fails degrades that expert to 0 rather than failing the whole
    roster: the team page still renders, just without that spend figure.
    """
    semaphore = asyncio.Semaphore(_WEEKLY_SPEND_READ_CONCURRENCY)

    async def read(expert_id: str) -> tuple[str, int]:
        async with semaphore:
            try:
                return expert_id, await get_weekly_spend(expert_id)
            except Exception:
                logger.warning(
                    "Failed to read weekly spend for expert #%s",
                    expert_id,
                    exc_info=True,
                )
                return expert_id, 0

    return dict(await asyncio.gather(*(read(expert_id) for expert_id in expert_ids)))


async def list_experts(user_id: str, *, with_metrics: bool = True) -> list[Expert]:
    """List the user's hired roster, with workflow names always included.

    Set ``with_metrics=False`` to skip the ``AgentGraphExecution`` lookup and
    the per-expert Redis spend reads — callers that only render name/role/id
    and workflow names (e.g. the copilot team-context roster) would otherwise
    pay for ``latest_run``/``weekly_spend`` data they discard. Those fields
    come back as their unset defaults (``None`` / ``0``) in that case.
    """
    rows = await prisma.models.Expert.prisma().find_many(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        include=_ROSTER_WORKFLOW_INCLUDE,
    )
    if not with_metrics:
        return [_to_model(row) for row in rows]
    latest_runs = await _latest_runs([row.id for row in rows])
    weekly_spends = await _weekly_spends([row.id for row in rows])
    credential_providers = await _credential_providers(user_id, rows)
    return [
        _to_model(
            row, latest_runs.get(row.id), weekly_spends.get(row.id, 0)
        ).model_copy(update=_credential_fields(credential_providers.get(row.id, [])))
        for row in rows
    ]


async def _credential_providers(
    user_id: str, rows: list[prisma.models.Expert]
) -> dict[str, list[str]]:
    """Each expert's live grants, or nothing when the read fails.

    The logos are decoration on the roster and the expert page; a credential
    outage must not take the whole expert with it.
    """
    try:
        return await expert_credential_providers(user_id, rows)
    except Exception:
        logger.exception("Failed to read credential providers for experts")
        return {}


def _credential_fields(providers: list[str]) -> dict[str, object]:
    """The count is every grant; the logos show each provider once, in the
    order it was first granted."""
    return {
        "credential_count": len(providers),
        "credential_providers": list(dict.fromkeys(providers)),
    }


async def list_expert_identities(user_id: str) -> list[ExpertIdentity]:
    """Return the lifetime roster without hydrating team-page details.

    Raw SQL rather than a Prisma projection: ``find_many`` has no partial
    ``select`` in prisma-client-py, so the ORM path would hydrate every Expert
    column (including the Soul text this endpoint exists to avoid) on every
    copilot mount. Only ``{schema_prefix}`` is interpolated — a server-side
    constant from settings, never request data — and ``user_id`` is bound as
    ``$1``. ``experts_db_test.py`` asserts the selected columns so a
    ``schema.prisma`` rename fails in CI instead of at runtime.
    """
    return await query_raw_with_schema(
        """
        SELECT "id", "name", "avatarUrl" AS "avatar_url", "color", "role",
               "isArchived" AS "is_archived"
        FROM {schema_prefix}"Expert"
        WHERE "ownerUserId" = $1 AND "isTemplate" = false
        """,
        user_id,
        model=ExpertIdentity,
    )


async def owns_active_expert(user_id: str, expert_id: str) -> bool:
    """True iff *user_id* owns *expert_id* and that expert is still hireable.

    The ownership half is the point: callers use this to authorise writes, so
    a fired (archived), template, or someone else's expert must all answer
    False here rather than being distinguished by the caller.
    """
    return (
        await prisma.models.Expert.prisma().count(
            where={
                "id": expert_id,
                "ownerUserId": user_id,
                "isTemplate": False,
                "isArchived": False,
            }
        )
        > 0
    )


async def owns_private_active_expert(user_id: str, expert_id: str) -> bool:
    """True iff *user_id* owns *expert_id* as a live, private hire.

    Stricter than :func:`owns_active_expert` by the visibility filter, which
    is what the per-expert resource routes need: a TEAM/ORG expert has no
    sharing rules yet, so writing its folder would mutate an expert the chat
    side resolves to no grants at all.
    """
    return await _owned_active_expert(user_id, expert_id) is not None


async def get_expert(
    user_id: str,
    expert_id: str,
    *,
    include_workflows: bool = True,
    include_archived: bool = False,
    include_credentials: bool = False,
) -> Expert | None:
    """Fetch a hired expert owned by *user_id*.

    Set ``include_workflows=False`` to skip the ExpertWorkflow → LibraryAgent
    + StoreListingVersion joins when the caller only needs the expert's own
    columns. The returned model then always carries an empty ``workflows``
    list — never use that flag to decide whether workflows are installed.

    Set ``include_credentials=True`` to fill ``credential_count`` and
    ``credential_providers`` the way the roster does. Off by default because
    the read seeds the expert's allow-list on first touch, and most callers
    (hire, raise, the scheduler's scope gate) only need the expert's columns.

    Archived experts are hidden by default so product surfaces treat them as
    gone. Set ``include_archived=True`` when the caller must distinguish
    "archived" (reversible — re-hire revives) from "deleted": the scheduler's
    scope gate uses this to skip firings without destroying schedules that
    an un-archive should bring back.
    """
    where: prisma.types.ExpertWhereInput = {
        "id": expert_id,
        "ownerUserId": user_id,
        "isTemplate": False,
        "visibility": ResourceVisibility.PRIVATE,
    }
    if not include_archived:
        where["isArchived"] = False
    row = await prisma.models.Expert.prisma().find_first(
        where=where,
        include=_WORKFLOW_INCLUDE if include_workflows else None,
    )
    if row is None:
        return None
    latest_runs = await _latest_runs([row.id])
    expert = _to_model(row, latest_runs.get(row.id), await get_weekly_spend(row.id))
    if not include_credentials:
        return expert
    credential_providers = await _credential_providers(user_id, [row])
    return expert.model_copy(
        update=_credential_fields(credential_providers.get(row.id, []))
    )


async def list_expert_runs(
    user_id: str, expert_id: str, limit: int = _MAX_EXPERT_RUNS
) -> list[ExpertRun]:
    """Recent expert-attributed executions with a classified output type.

    Owner-scoped: the execution, review and workflow lookups all filter by
    *user_id*, so one user's Work surface can never surface another's runs.
    Raises :class:`ExpertNotFoundError` when the expert isn't a live hire of
    this user.
    """
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        include=_WORKFLOW_INCLUDE,
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    workflow_by_graph = {
        w.LibraryAgent.agentGraphId: w
        for w in expert.Workflows or []
        if w.LibraryAgent is not None
    }

    executions = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={"userId": user_id, "expertId": expert_id, "isDeleted": False},
        order={"createdAt": "desc"},
        take=limit,
        include={"AgentPreset": True},
    )
    if not executions:
        return []
    execution_ids = [execution.id for execution in executions]

    # Exact per-execution review state (WAITING reviews for exactly these
    # ids) — a page of the user's newest reviews could miss an older run
    # that is still genuinely blocked.
    waiting_reviews = await prisma.models.PendingHumanReview.prisma().find_many(
        where={
            "userId": user_id,
            "status": prisma.enums.ReviewStatus.WAITING,
            "graphExecId": {"in": execution_ids},
        }
    )
    reviewing = {review.graphExecId for review in waiting_reviews}
    classified = await _classify_run_outputs(execution_ids)

    return [
        _to_expert_run(
            execution,
            workflow_by_graph.get(execution.agentGraphId),
            *classified.get(execution.id, ("unknown", None)),
            needs_review=execution.id in reviewing,
        )
        for execution in executions
    ]


class _DayCount(BaseModel):
    day: date
    count: int


_ACTIVITY_TABLES: dict[str, str] = {
    "ChatSession": "",
    "AgentGraphExecution": 'AND "isDeleted" = false',
}


async def _count_expert_rows_by_day(
    table: Literal["ChatSession", "AgentGraphExecution"],
    user_id: str,
    expert_id: str,
    since: datetime,
    tz_name: str,
) -> dict[date, int]:
    """Rows of *table* stamped with this expert, bucketed by the owner's
    calendar day. ``createdAt`` is a UTC wall-clock ``timestamp``, so it is
    re-tagged as UTC before shifting into the owner's zone."""
    rows = await query_raw_with_schema(
        f"""
        SELECT (("createdAt" AT TIME ZONE 'UTC') AT TIME ZONE $3::text)::date AS day,
               COUNT(*)::int AS count
        FROM {{schema_prefix}}"{table}"
        WHERE "userId" = $1
          AND "expertId" = $2
          AND "createdAt" >= ($4::timestamptz AT TIME ZONE 'UTC')
          {_ACTIVITY_TABLES[table]}
        GROUP BY day
        """,
        user_id,
        expert_id,
        tz_name,
        since,
        model=_DayCount,
    )
    return {row.day: row.count for row in rows}


async def get_expert_activity(user_id: str, expert_id: str) -> ExpertActivity:
    """Per-day chat sessions and runs for the last :data:`EXPERT_ACTIVITY_DAYS`,
    on the owner's calendar so "today" matches what they see.

    Raises :class:`ExpertNotFoundError` when the expert isn't a live hire of
    this user.
    """
    if not await owns_active_expert(user_id, expert_id):
        raise ExpertNotFoundError(expert_id)

    user = await get_user_by_id(user_id)
    tz_name = get_user_timezone_or_utc(user.timezone if user else None)
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    first_day = today - timedelta(days=EXPERT_ACTIVITY_DAYS - 1)
    since = datetime.combine(first_day, time.min, tzinfo=tz)

    sessions, runs = await asyncio.gather(
        _count_expert_rows_by_day("ChatSession", user_id, expert_id, since, tz_name),
        _count_expert_rows_by_day(
            "AgentGraphExecution", user_id, expert_id, since, tz_name
        ),
    )
    days = [
        first_day + timedelta(days=offset) for offset in range(EXPERT_ACTIVITY_DAYS)
    ]
    return ExpertActivity(
        timezone=tz_name,
        days=[
            ExpertActivityDay(
                day=day, sessions=sessions.get(day, 0), runs=runs.get(day, 0)
            )
            for day in days
        ],
    )


async def _classify_run_outputs(
    execution_ids: list[str],
) -> dict[str, tuple[OutputType, str | None]]:
    """Batch-classify run outputs: one bounded query for the OUTPUT-block
    node executions (plus their small name/value input rows) of all listed
    executions, instead of a full ``get_graph_execution`` per run.

    ``execution_ids`` must already be user-scoped by the caller (they come
    from the owner-filtered executions query). Any per-execution parse
    failure degrades that run to ``("unknown", None)`` — one corrupt run
    must never 500 the whole Work tab.
    """
    node_execs = await prisma.models.AgentNodeExecution.prisma().find_many(
        where={
            "agentGraphExecutionId": {"in": execution_ids},
            "Node": {"is": {"agentBlockId": {"in": list(get_output_block_ids())}}},
            "executionStatus": {"not": prisma.enums.AgentExecutionStatus.INCOMPLETE},
        },
        include={"Input": True},
    )
    by_execution: dict[str, list[prisma.models.AgentNodeExecution]] = defaultdict(list)
    for node_exec in node_execs:
        by_execution[node_exec.agentGraphExecutionId].append(node_exec)

    classified: dict[str, tuple[OutputType, str | None]] = {}
    for execution_id in execution_ids:
        try:
            classified[execution_id] = classify_run_output(
                _outputs_from_node_execs(by_execution.get(execution_id, []))
            )
        except Exception as e:
            logger.warning(
                f"Failed to classify outputs for run #{execution_id}: "
                f"{type(e).__name__}: {e}"
            )
            classified[execution_id] = ("unknown", None)
    return classified


def _outputs_from_node_execs(
    node_execs: list[prisma.models.AgentNodeExecution],
) -> dict[str, list[JsonValue]]:
    return reconstruct_run_outputs(
        [
            (node_exec.queuedTime, node_exec.addedTime, _node_exec_inputs(node_exec))
            for node_exec in node_execs
        ]
    )


def _node_exec_inputs(
    node_exec: prisma.models.AgentNodeExecution,
) -> dict[str, JsonValue]:
    """Mirror ``NodeExecutionResult.from_db`` input precedence: moderation-cleared
    inputs win over the denormalized ``executionData`` blob, which wins over the
    Input rows. Skipping the cleared branch would drop the name/value pins of a
    moderated OUTPUT node and misclassify the run as ``unknown``.
    """
    try:
        stats = NodeExecutionStats.model_validate(node_exec.stats or {})
    except (ValueError, ValidationError):
        stats = NodeExecutionStats()

    if stats.cleared_inputs:
        return {
            name: (messages[-1] if messages else "")
            for name, messages in stats.cleared_inputs.items()
        }
    if node_exec.executionData is not None:
        return cast(
            dict[str, JsonValue],
            type_utils.convert(node_exec.executionData, dict),
        )
    return {
        row.name: type_utils.convert(row.data, JsonValue)
        for row in node_exec.Input or []
    }


def _run_source(execution: prisma.models.AgentGraphExecution) -> ExpertRunSource:
    preset = getattr(execution, "AgentPreset", None)
    if preset is None:
        return "manual"
    return "trigger" if preset.webhookId else "scheduled"


def _to_expert_run(
    execution: prisma.models.AgentGraphExecution,
    workflow: prisma.models.ExpertWorkflow | None,
    output_type: OutputType,
    output_key: str | None,
    *,
    needs_review: bool,
) -> ExpertRun:
    listing = workflow.StoreListingVersion if workflow else None
    library_agent = workflow.LibraryAgent if workflow else None
    library_agent_id = workflow.libraryAgentId if workflow else None
    # Library-only workflows carry no listing, so the name comes from the
    # user's own library agent rather than falling through to the placeholder.
    if listing is not None:
        agent_name = listing.name
    elif library_agent is not None:
        agent_name = _library_agent_labels(library_agent)[0] or DEFAULT_AGENT_NAME
    else:
        agent_name = DEFAULT_AGENT_NAME
    return ExpertRun(
        execution_id=execution.id,
        graph_id=execution.agentGraphId,
        agent_name=agent_name,
        library_agent_id=library_agent_id,
        status=cast(ExpertRunStatus, str(execution.executionStatus).lower()),
        output_type=output_type,
        output_key=output_key,
        needs_review=needs_review,
        source=_run_source(execution),
        started_at=execution.startedAt,
        ended_at=execution.endedAt,
        link=run_link(library_agent_id, execution.id),
    )


async def expert_row_exists(user_id: str, expert_id: str) -> bool:
    """Lenient existence check for a hired expert row owned by *user_id*.

    Unlike :func:`get_expert` this ignores visibility and archive state, so
    callers can tell "row exists but is not currently accessible" (archived /
    no-longer-private) apart from "row truly gone". The copilot-turn
    scheduler uses it to keep schedules registered for recovery instead of
    irreversibly self-deleting them.
    """
    count = await prisma.models.Expert.prisma().count(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
        }
    )
    return count > 0


async def resolve_private_expert_tenancy(
    user_id: str, expert_id: str
) -> tuple[str, str | None]:
    """Return the owner scope for an active, owner-only PRIVATE expert.

    TEAM and ORG experts are deliberately unsupported for now. Checking the
    visibility here before resolving or rewriting any child resource keeps
    those future scopes fail-closed instead of silently moving them into the
    owner's personal organization.
    """
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    organization_id, team_id = await get_user_default_team(user_id)
    if organization_id is None:
        raise ExpertPrivateTenancyNotFoundError(expert_id)
    return organization_id, team_id


async def hire_expert(user_id: str, template_id: str, name: str | None) -> HireResult:
    template = await prisma.models.Expert.prisma().find_first(
        where={"id": template_id, "isTemplate": True, "isArchived": False},
        include=_WORKFLOW_INCLUDE,
    )
    if template is None:
        raise ExpertTemplateNotFoundError(template_id)

    # Copy the plain description, never the template's sample envelope: a hire
    # that skips the voice pick must not leave raw JSON in the prompt, and the
    # pick (when made) overwrites this via the soul PATCH anyway.
    template_voice, _ = decode_voice_preferences(template.voicePreferences)
    create_data: prisma.types.ExpertCreateInput = {
        "ownerUserId": user_id,
        "name": name or template.name,
        "avatarUrl": template.avatarUrl,
        "color": template.color,
        "role": template.role,
        "tagline": template.tagline,
        "bio": template.bio,
        # The bundled installs below record each name, so the row lists only
        # skills the hire actually owns.
        "skills": [],
        "categories": template.categories or [],
        # No dayOne: it is the template's pre-hire promise, not the hire's.
        "identity": template.identity,
        "voicePreferences": template_voice,
        "boundaries": template.boundaries,
        "sourceTemplateId": template.id,
        "visibility": ResourceVisibility.PRIVATE,
    }
    if template.toolProfile is not None:
        create_data["toolProfile"] = template.toolProfile

    try:
        expert, state = await _reserve_hired_expert(user_id, template_id, create_data)
    except prisma.errors.UniqueViolationError:
        # A caller running older code may not participate in the advisory lock.
        # Retry after the failed transaction so its winning row is handled by
        # the same capacity-aware path.
        expert, state = await _reserve_hired_expert(user_id, template_id, create_data)

    if state == "existing":
        return HireResult(expert=_to_model(expert), failed_preloads=[])
    if state == "revived":
        expert = await _resume_revived_hire(expert)
        return HireResult(expert=_to_model(expert), failed_preloads=[])

    failed = await _install_preloads(expert.id, user_id, template.Workflows or [])
    await _install_bundled_skills(user_id, expert.id, template.id)

    hydrated = await prisma.models.Expert.prisma().find_unique(
        where={"id": expert.id}, include=_WORKFLOW_INCLUDE
    )
    if hydrated is None:
        raise ExpertNotFoundError(expert.id)
    return HireResult(expert=_to_model(hydrated), failed_preloads=failed)


async def _reserve_hired_expert(
    user_id: str,
    template_id: str,
    create_data: prisma.types.ExpertCreateInput,
) -> tuple[prisma.models.Expert, Literal["existing", "revived", "created"]]:
    """Atomically get, revive, or create one hired expert.

    Hires share the same per-user lock and active-team capacity check as
    raised experts. An idempotent retry of an already-active hire does not
    consume capacity, while reviving an archived hire does.
    """
    async with transaction() as tx:
        await _lock_expert_creation(tx, user_id)
        existing = await tx.expert.find_first(
            where={"ownerUserId": user_id, "sourceTemplateId": template_id},
            include=_WORKFLOW_INCLUDE,
        )
        if existing is not None:
            # Fail closed on a hire that would resolve to a non-PRIVATE row:
            # idempotent re-hire must never hand back an expert the rest of
            # the API hides (mirrors get_expert's visibility filter).
            if existing.visibility != ResourceVisibility.PRIVATE:
                raise ExpertNotFoundError(existing.id)
            if not existing.isArchived:
                return existing, "existing"
            await _ensure_active_expert_capacity(tx, user_id)
            revived = await tx.expert.update(
                where={"id": existing.id},
                data={"isArchived": False},
                include=_WORKFLOW_INCLUDE,
            )
            if revived is None:
                raise ExpertNotFoundError(existing.id)
            return revived, "revived"

        await _ensure_active_expert_capacity(tx, user_id)
        created = await tx.expert.create(
            data=create_data,
            include=_WORKFLOW_INCLUDE,
        )
        return created, "created"


async def _resume_revived_hire(row: prisma.models.Expert) -> prisma.models.Expert:
    if row.ownerUserId is None:
        return row
    owner_user_id = row.ownerUserId

    # Fail-closed revive: the personal workspace must exist before schedules
    # re-attach, and a failed reattach rolls the row back to archived so the
    # hire surfaces as retryable instead of returning an expert with dead
    # triggers.
    organization_id, _ = await get_user_default_team(owner_user_id)
    if organization_id is None:
        await _rollback_revive(owner_user_id, row.id)
        raise ExpertPrivateTenancyNotFoundError(row.id)

    try:
        await scheduling.resume_expert_schedules(owner_user_id, row.id)
        await scheduling.reattach_expert_triggers(owner_user_id, row.id)
    except Exception as e:
        logger.exception(f"Failed to reattach triggers while reviving expert #{row.id}")
        await _rollback_revive(owner_user_id, row.id)
        raise ExpertHireUnavailableError(row.id) from e

    # Resume/reattach mutated pause state and workflow scheduleIds after `row`
    # was read — reload so the result isn't stale.
    refreshed = await prisma.models.Expert.prisma().find_unique(
        where={"id": row.id}, include=_WORKFLOW_INCLUDE
    )
    return refreshed or row


async def _rollback_revive(owner_user_id: str, expert_id: str) -> None:
    """Best-effort restore of the archived state after a failed revive.

    Pause before re-archiving — ``pause_expert_schedules`` refuses archived
    rows (same ordering as ``archive_expert``).
    """
    try:
        await scheduling.pause_expert_schedules(
            owner_user_id, expert_id, reason="Expert re-hire did not complete"
        )
        await prisma.models.Expert.prisma().update(
            where={"id": expert_id},
            data={"isArchived": True},
        )
        await scheduling.detach_expert_triggers(owner_user_id, expert_id)
    except Exception:
        logger.exception(f"Failed to restore archived state for expert #{expert_id}")


async def _lock_expert_creation(tx: prisma.Prisma, user_id: str) -> None:
    # execute_raw, not query_raw: pg_advisory_xact_lock returns void,
    # which Prisma cannot deserialize as a result column.
    await tx.execute_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", user_id
    )


async def _ensure_active_expert_capacity(tx: prisma.Prisma, user_id: str) -> None:
    active_count = await tx.expert.count(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        }
    )
    if active_count >= ACTIVE_EXPERT_LIMIT:
        raise ExpertLimitExceededError(ACTIVE_EXPERT_LIMIT)


async def count_active_experts(user_id: str) -> int:
    """Active hired experts. Lock-free, for preview-time capacity checks only;
    the creation transaction re-enforces the cap."""
    return await prisma.models.Expert.prisma().count(
        where={"ownerUserId": user_id, "isTemplate": False, "isArchived": False}
    )


async def count_raised_experts(user_id: str) -> int:
    """Lifetime raised experts, archived included. Same preview-only caveat."""
    return await prisma.models.Expert.prisma().count(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "sourceTemplateId": None,
        }
    )


async def create_raised_expert(
    user_id: str,
    name: str,
    role: str | None,
    voice_preferences: str | None,
    *,
    avatar_url: str | None = None,
    color: str | None = None,
    tagline: str | None = None,
    about: str | None = None,
    boundaries: str | None = None,
    weekly_budget: int | None = None,
    attachments: list[RaiseAttachment] | None = None,
) -> RaiseResult:
    """Raise a blank expert owned by *user_id*.

    A raised expert has no source template, so ``sourceTemplateId`` stays
    NULL. Capacity checks and creation share a per-user advisory lock.
    Attachments are validated before creation. Workflow install failure
    remains non-fatal and is reported in the result.
    """
    resolved = await raise_attachments.resolve_attachments(user_id, attachments or [])
    expert = await _create_raised_expert_row(
        user_id,
        name,
        role,
        voice_preferences,
        avatar_url=avatar_url,
        color=color,
        tagline=tagline,
        about=about,
        boundaries=boundaries,
        weekly_budget=weekly_budget,
        skills=resolved.skill_names,
    )
    failed_skills = await _copy_library_skills(user_id, expert.id, resolved.skill_names)
    if failed_skills:
        expert = (
            await prisma.models.Expert.prisma().update(
                where={"id": expert.id},
                data={
                    "skills": [
                        s for s in (expert.skills or []) if s not in failed_skills
                    ]
                },
                include=_WORKFLOW_INCLUDE,
            )
            or expert
        )
    failed_attachments = await raise_attachments.install_workflows(
        user_id, expert.id, resolved.workflows
    )
    if resolved.workflows and len(failed_attachments) < len(resolved.workflows):
        hydrated = await get_expert(user_id, expert.id)
        if hydrated is None:
            raise ExpertNotFoundError(expert.id)
    else:
        hydrated = _to_model(expert)
    return RaiseResult(expert=hydrated, failed_attachments=failed_attachments)


async def _copy_library_skills(
    user_id: str, expert_id: str, names: list[str]
) -> list[str]:
    """Give a freshly raised expert its own copies of the Otto skills it
    was raised with. Defaults and marketplace names have nothing to copy.
    Returns the names whose copy failed so the caller can drop them from the
    expert's row rather than list a skill the expert cannot read."""
    candidates = [
        name
        for name in names
        if get_default_skill_with_body(name.strip().lower()) is None
    ]
    folders = await find_user_skill_slugs(user_id, candidates)
    failed: list[str] = []
    for name in candidates:
        folder = folders.get(name.strip().lower())
        if folder is None:
            # A marketplace attachment: no library folder to copy, and the
            # name is legitimate, so it stays on the row.
            continue
        try:
            if await copy_skill_to_expert(user_id, expert_id, folder) is None:
                failed.append(name)
        except Exception:
            logger.exception(f"Failed to copy skill {name!r} to expert #{expert_id}")
            failed.append(name)
    return failed


async def _create_raised_expert_row(
    user_id: str,
    name: str,
    role: str | None,
    voice_preferences: str | None,
    *,
    avatar_url: str | None,
    color: str | None,
    tagline: str | None = None,
    about: str | None,
    boundaries: str | None = None,
    weekly_budget: int | None = None,
    skills: list[str] | None = None,
) -> prisma.models.Expert:
    async with transaction() as tx:
        await _lock_expert_creation(tx, user_id)
        await _ensure_active_expert_capacity(tx, user_id)
        lifetime_raised_count = await tx.expert.count(
            where={
                "ownerUserId": user_id,
                "isTemplate": False,
                "sourceTemplateId": None,
            }
        )
        if lifetime_raised_count >= LIFETIME_RAISED_EXPERT_LIMIT:
            raise RaisedExpertLifetimeLimitExceededError(LIFETIME_RAISED_EXPERT_LIMIT)
        return await tx.expert.create(
            data={
                "ownerUserId": user_id,
                "name": name,
                "avatarUrl": avatar_url,
                "color": color or "",
                "role": role or "",
                "tagline": tagline,
                "identity": about or _raised_identity(name),
                "voicePreferences": voice_preferences or "",
                "boundaries": boundaries or "",
                "weeklyBudget": weekly_budget,
                "skills": skills or [],
            },
            include=_WORKFLOW_INCLUDE,
        )


async def _install_first_job(
    user_id: str,
    expert_id: str,
    store_listing_version_id: str,
) -> None:
    await raise_attachments.install_marketplace_workflow(
        user_id, expert_id, store_listing_version_id
    )


async def update_skills(
    user_id: str,
    expert_id: str,
    skills: list[str],
    marketplace_listing_ids: list[str] | None = None,
) -> Expert:
    """Replace an expert's skill list.

    Names the expert does not already carry must resolve to a library skill.
    A personal-Otto skill is copied into the expert's own folder so the
    expert owns it from then on; names dropped from the list delete the
    expert's copy. The stored name is the skill's canonical one so display
    and lookup agree."""
    row = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if row is None:
        raise ExpertNotFoundError(expert_id)

    # Resolve every name before any write, so a bad name rejects the whole
    # request instead of leaving a half-applied prefix of copies behind.
    current = {name.lower(): name for name in row.skills or []}
    # One library listing for the whole request: resolving per name turned a
    # single PUT into a storage scan per name.
    folders = await find_user_skill_slugs(
        user_id, [current.get(name.lower()) or name for name in skills]
    )
    plan = [_plan_skill(current.get(name.lower()), name, folders) for name in skills]
    marketplace = [
        await _resolve_marketplace_skill_name(listing_id)
        for listing_id in marketplace_listing_ids or []
    ]
    resolved: list[str] = []
    for canonical, folder in plan:
        if folder is not None:
            # Idempotent: also heals a name kept from before skills were
            # owned per expert, which had no copy in the expert's folder.
            if await copy_skill_to_expert(user_id, expert_id, folder) is None:
                # Resolved moments ago and gone now. Keeping the name would
                # leave the row listing a skill the expert cannot read, which
                # is the one state this whole path exists to prevent.
                logger.warning(
                    f"Skill {canonical!r} vanished before it could be copied "
                    f"to expert #{expert_id}; dropping it from the list"
                )
                continue
        resolved.append(canonical)
    for name in marketplace:
        if name.lower() not in {r.lower() for r in resolved}:
            resolved.append(name)
    kept = {r.lower() for r in resolved}
    for dropped in [name for name in current.values() if name.lower() not in kept]:
        # delete_user_skill drops the row name itself — except for a built-in,
        # where it raises first and _detach_expert_skill swallows that.
        await _detach_expert_skill(user_id, expert_id, dropped)
        await remove_expert_skill_name(user_id, expert_id, dropped)
    # Per-name writes that each re-read the row, not one set computed up front:
    # the expert can append to its own row with store_skill while the copies
    # above run, and an overwrite from the pre-copy read would silently drop it.
    for name in resolved:
        await add_expert_skill_name(user_id, expert_id, name)
    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def _resolve_marketplace_skill_name(store_listing_version_id: str) -> str:
    if not await library_db.is_store_listing_version_available_for_install(
        store_listing_version_id
    ):
        raise NotFoundError(f"Marketplace skill #{store_listing_version_id} not found")
    listing = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id}
    )
    if listing is None:
        raise NotFoundError(f"Marketplace skill #{store_listing_version_id} not found")
    return listing.name


def _plan_skill(
    kept_name: str | None, name: str, folders: dict[str, str]
) -> tuple[str, str | None]:
    """Decide the stored name and which Otto folder, if any, to copy.

    A name the expert already carries is kept as is; its folder is looked up
    so a legacy assignment without a copy gets one. A new name must be a
    default skill or one of Otto's skills, resolved to its folder (a
    hand-written skill may be listed under a frontmatter name that differs
    from the folder). Raises ``NotFoundError`` before anything is written.
    """
    slug = (kept_name or name).strip().lower()
    default = get_default_skill_with_body(slug)
    if default is not None:
        return default.name, None
    folder = folders.get(slug)
    if kept_name is not None:
        return kept_name, folder
    if folder is None:
        raise NotFoundError(f"Skill '{name}' is not in your library")
    return folder, folder


async def _detach_expert_skill(user_id: str, expert_id: str, name: str) -> None:
    try:
        await delete_user_skill(user_id, name, expert_id=expert_id)
    except (SkillNotFoundError, BuiltInSkillError, ValueError):
        return


async def add_expert_skill_name(user_id: str, expert_id: str, name: str) -> None:
    """Record a skill the expert now owns; idempotent, and a display name and
    its slug count as one name."""
    await _rewrite_skill_names(
        {
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        lambda names: (
            names
            if skill_name_key(name) in {skill_name_key(n) for n in names}
            else [*names, name]
        ),
    )


async def remove_expert_skill_name(user_id: str, expert_id: str, name: str) -> None:
    """Forget a skill the expert no longer owns, in any spelling of its name."""
    await _rewrite_skill_names(
        {"id": expert_id, "ownerUserId": user_id},
        lambda names: [n for n in names if skill_name_key(n) != skill_name_key(name)],
    )


_SKILL_NAME_WRITE_ATTEMPTS = 5


async def _rewrite_skill_names(
    where: prisma.types.ExpertWhereInput,
    rewrite: Callable[[list[str]], list[str]],
) -> None:
    """Compare-and-swap the row's skill list. store_skill can append to it
    while update_skills runs, so a write that lands after the read fails the
    ``equals`` guard and is re-read rather than overwritten."""
    for _ in range(_SKILL_NAME_WRITE_ATTEMPTS):
        row = await prisma.models.Expert.prisma().find_first(where=where)
        if row is None:
            return
        names = rewrite(row.skills)
        if names == row.skills:
            return
        if await prisma.models.Expert.prisma().update_many(
            where={**where, "skills": {"equals": row.skills}},
            data={"skills": {"set": names}},
        ):
            return
    raise ConflictError(
        "This expert's skills were changed by another update at the same time. "
        "Try again."
    )


async def _owned_active_expert(
    user_id: str, expert_id: str
) -> prisma.models.Expert | None:
    return await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )


async def update_soul(user_id: str, expert_id: str, soul: ExpertSoulUpdate) -> Expert:
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        data={
            "name": soul.name,
            "identity": soul.identity,
            "voicePreferences": soul.voice_preferences,
            "boundaries": soul.boundaries,
        },
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def update_avatar(user_id: str, expert_id: str, avatar_url: str | None) -> Expert:
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        data={"avatarUrl": avatar_url},
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def update_budget(
    user_id: str, expert_id: str, weekly_budget: int | None
) -> Expert:
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        data={"weeklyBudget": weekly_budget},
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def update_soul_if_current(
    user_id: str,
    expert_id: str,
    soul: ExpertSoulUpdate,
    *,
    expected_name: str,
    expected_identity: str,
    expected_voice_preferences: str,
    expected_boundaries: str,
) -> Expert | None:
    """Whole-soul :func:`update_soul` that only lands while the soul is unchanged.

    Backs the copilot ``update_expert`` confirm step, which rewrites every
    column from a preview taken minutes earlier — without the comparison a
    concurrent edit from the team UI would be reverted. ``None`` means the
    write was refused (the expert is gone or moved); the caller re-previews.

    The rowcount is the only success signal: once it is non-zero the edit is
    committed, so a read-back that comes up empty must not be reported as a
    refusal. Archived rows are included for that reason, and a row that is
    gone outright raises rather than returning ``None``.
    """
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
            "name": expected_name,
            "identity": expected_identity,
            "voicePreferences": expected_voice_preferences,
            "boundaries": expected_boundaries,
        },
        data={
            "name": soul.name,
            "identity": soul.identity,
            "voicePreferences": soul.voice_preferences,
            "boundaries": soul.boundaries,
        },
    )
    if updated == 0:
        return None
    expert = await get_expert(user_id, expert_id, include_archived=True)
    if expert is None:
        raise ExpertWriteNotReadableError(expert_id)
    return expert


def _soul_field_update_data(
    *,
    identity: str | None,
    voice_preferences: str | None,
    boundaries: str | None,
) -> prisma.types.ExpertUpdateManyMutationInput:
    patch = ExpertSoulFieldsPatch(
        identity=identity,
        voice_preferences=voice_preferences,
        boundaries=boundaries,
    )
    data: prisma.types.ExpertUpdateManyMutationInput = {}
    if patch.identity is not None:
        data["identity"] = patch.identity
    if patch.voice_preferences is not None:
        data["voicePreferences"] = patch.voice_preferences
    if patch.boundaries is not None:
        data["boundaries"] = patch.boundaries
    if not data:
        raise ValueError("At least one Soul field must be provided")
    return data


async def update_soul_fields(
    user_id: str,
    expert_id: str,
    *,
    identity: str | None = None,
    voice_preferences: str | None = None,
    boundaries: str | None = None,
) -> Expert:
    """Patch only the supplied Soul fields in one scoped write.

    Backs the copilot Soul-edit tools, which edit identity / voice /
    boundaries but never rename the expert. A single ``update_many`` writes
    only the supplied columns, so concurrent edits to disjoint fields cannot
    clobber each other; per-field validation mirrors ``update_soul`` via
    ``ExpertSoulFieldsPatch``.
    """
    data = _soul_field_update_data(
        identity=identity,
        voice_preferences=voice_preferences,
        boundaries=boundaries,
    )

    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        data=data,
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def update_soul_fields_if_current(
    user_id: str,
    expert_id: str,
    *,
    identity: str | None = None,
    voice_preferences: str | None = None,
    boundaries: str | None = None,
    expected_identity: str | None = None,
    expected_voice_preferences: str | None = None,
    expected_boundaries: str | None = None,
) -> bool:
    """Atomically patch Soul fields only when their previewed values still match."""
    data = _soul_field_update_data(
        identity=identity,
        voice_preferences=voice_preferences,
        boundaries=boundaries,
    )
    comparisons: dict[str, str] = {}
    for field, value, expected in (
        ("identity", identity, expected_identity),
        ("voicePreferences", voice_preferences, expected_voice_preferences),
        ("boundaries", boundaries, expected_boundaries),
    ):
        if value is None:
            continue
        if expected is None:
            raise ValueError(f"Expected value required for {field}")
        comparisons[field] = expected

    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            **comparisons,
        },
        data=data,
    )
    return updated == 1


async def _install_preloads(
    expert_id: str, user_id: str, preloads: list[prisma.models.ExpertWorkflow]
) -> list[str]:
    """Install template preloads into the hiring user's library.

    Honest partial hire: a failed preload is logged and reported, never
    fatal to the hire itself. Preloads with a roster cadence also get their
    schedule created here (see ``_schedule_preload``).
    """
    failed: list[str] = []
    user_timezone: str | None = None
    if any(p.scheduleCron for p in preloads):
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)
    # Rows first, schedules second: creating a schedule resolves credentials
    # scoped to the expert, which seeds its allow-list from the workflows
    # installed so far. Interleaving would freeze that list after the first one.
    installed: list[
        tuple[
            prisma.models.ExpertWorkflow,
            prisma.models.ExpertWorkflow,
            library_model.LibraryAgent,
        ]
    ] = []
    for preload in preloads:
        if preload.storeListingVersionId is None:
            continue
        try:
            library_agent = await library_db.add_store_agent_to_library(
                preload.storeListingVersionId, user_id
            )
            row = await prisma.models.ExpertWorkflow.prisma().create(
                data={
                    "expertId": expert_id,
                    "storeListingVersionId": preload.storeListingVersionId,
                    "libraryAgentId": library_agent.id,
                    "scheduleCron": preload.scheduleCron,
                }
            )
        except Exception:
            logger.exception(
                f"Failed to install preload {preload.storeListingVersionId} "
                f"on expert #{expert_id} for user #{user_id}"
            )
            failed.append(
                preload.StoreListingVersion.name
                if preload.StoreListingVersion
                else preload.storeListingVersionId
            )
            continue
        installed.append((row, preload, library_agent))
    for row, preload, library_agent in installed:
        if not preload.scheduleCron:
            continue
        listing = preload.StoreListingVersion
        await scheduling.create_workflow_schedule(
            workflow_row_id=row.id,
            expert_id=expert_id,
            user_id=user_id,
            cron=preload.scheduleCron,
            graph_id=library_agent.graph_id,
            graph_version=library_agent.graph_version,
            name=listing.name if listing else "Expert workflow",
            user_timezone=user_timezone or "UTC",
        )
    return failed


async def _install_bundled_skills(
    user_id: str, expert_id: str, template_id: str
) -> None:
    """Install the Hub skills the template bundles into the new expert's folder.

    Each install records its name on the row; a failed one is logged and
    leaves no name, so the hire never lists a skill it does not have.
    """
    bundled = await _live_bundled_skills(user_id, [template_id])
    for skill in bundled.get(template_id, []):
        try:
            await skill_db.install_marketplace_skill(
                user_id, skill.slug, expert_id=expert_id
            )
        except Exception:
            logger.exception(
                f"Failed to install bundled skill {skill.slug!r} on expert #{expert_id}"
            )


async def install_workflow(
    user_id: str,
    expert_id: str,
    *,
    store_listing_version_id: str | None = None,
    library_agent_id: str | None = None,
) -> ExpertWorkflowRef:
    """Attach a workflow to a hired expert.

    Exactly one source: a marketplace listing version, or one of the caller's
    own library agents. The library path records no listing version, so an
    agent that was never published is installable.
    """
    if bool(store_listing_version_id) == bool(library_agent_id):
        raise ValueError(
            "install_workflow takes exactly one of store_listing_version_id, "
            "library_agent_id"
        )

    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    if library_agent_id is not None:
        return await _install_library_workflow(user_id, expert_id, library_agent_id)
    assert store_listing_version_id is not None
    return await _install_marketplace_workflow(
        user_id, expert_id, store_listing_version_id
    )


async def _install_library_workflow(
    user_id: str, expert_id: str, library_agent_id: str
) -> ExpertWorkflowRef:
    library_agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={"id": library_agent_id, "userId": user_id, "isDeleted": False}
    )
    if library_agent is None:
        raise NotFoundError(f"Library agent #{library_agent_id} not found")

    # No listing version means the (expertId, storeListingVersionId) unique
    # index cannot dedupe these rows — Postgres treats each NULL as distinct.
    existing = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id, "libraryAgentId": library_agent_id},
        include=_WORKFLOW_ROW_INCLUDE,
    )
    if existing is not None:
        return _to_workflow_ref(existing)

    row = await prisma.models.ExpertWorkflow.prisma().create(
        data={"expertId": expert_id, "libraryAgentId": library_agent_id},
        include=_WORKFLOW_ROW_INCLUDE,
    )
    return _to_workflow_ref(row)


async def _install_marketplace_workflow(
    user_id: str, expert_id: str, store_listing_version_id: str
) -> ExpertWorkflowRef:
    existing = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={
            "expertId": expert_id,
            "storeListingVersionId": store_listing_version_id,
        },
        include=_WORKFLOW_ROW_INCLUDE,
    )
    if existing is not None:
        return _to_workflow_ref(existing)

    library_agent = await library_db.add_store_agent_to_library(
        store_listing_version_id, user_id
    )
    try:
        row = await prisma.models.ExpertWorkflow.prisma().create(
            data={
                "expertId": expert_id,
                "storeListingVersionId": store_listing_version_id,
                "libraryAgentId": library_agent.id,
            },
            include=_WORKFLOW_ROW_INCLUDE,
        )
    except prisma.errors.UniqueViolationError:
        # Lost a concurrent duplicate-install race; return the winner's row.
        raced = await prisma.models.ExpertWorkflow.prisma().find_first(
            where={
                "expertId": expert_id,
                "storeListingVersionId": store_listing_version_id,
            },
            include=_WORKFLOW_ROW_INCLUDE,
        )
        if raced is None:
            raise
        return _to_workflow_ref(raced)
    return _to_workflow_ref(row)


async def remove_workflow(user_id: str, expert_id: str, workflow_id: str) -> None:
    """Detach a workflow from a hired expert, dropping its install-time
    schedule. The library agent itself is left alone — it is still the
    user's, and another expert may share it."""
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"id": workflow_id, "expertId": expert_id}
    )
    if row is None:
        raise NotFoundError(f"Workflow #{workflow_id} not found on expert")

    if row.scheduleId:
        await scheduling.delete_workflow_schedule(row.scheduleId, user_id, expert_id)
    await prisma.models.ExpertWorkflow.prisma().delete(where={"id": row.id})


async def resolve_expert_for_graph(user_id: str, graph_id: str) -> str | None:
    """Expert attribution for a manually scheduled graph.

    Returns the id of the single active hired expert that has *graph_id*
    installed as a workflow. Two experts can install the same listing and
    share one LibraryAgent, which makes the join ambiguous — on anything
    but a unique match this declines (returns ``None``) rather than guess.

    Fails closed on visibility: a graph mapped to a TEAM/ORG expert raises
    ``ExpertNotFoundError`` (mirroring the 404 an explicit non-private
    ``expert_id`` gets) instead of returning ``None`` — silently detaching
    attribution would create an UNATTRIBUTED run that the expert budget
    guard never sees.

    Raises:
        ExpertNotFoundError: if any matching expert is not PRIVATE.
    """
    rows = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={
            "Expert": {
                "is": {
                    "ownerUserId": user_id,
                    "isTemplate": False,
                    "isArchived": False,
                }
            },
            "LibraryAgent": {
                "is": {
                    "userId": user_id,
                    "agentGraphId": graph_id,
                    "isDeleted": False,
                }
            },
        },
        include={"Expert": True},
    )
    for row in rows:
        if row.Expert and row.Expert.visibility != ResourceVisibility.PRIVATE:
            raise ExpertNotFoundError(row.expertId)
    expert_ids = {row.expertId for row in rows}
    if len(expert_ids) != 1:
        return None
    return expert_ids.pop()


async def resolve_attributable_expert(
    user_id: str, expert_id: str | None
) -> str | None:
    """Read-only expert-attribution lookup.

    Durable writes use the same shared guard with a row lock inside their own
    transaction; this lookup is for discovery and compatibility only.
    """
    return await resolve_attributable_expert_row(
        db_client,
        user_id,
        expert_id,
    )


async def archive_expert(user_id: str, expert_id: str) -> None:
    # Pause BEFORE flipping isArchived: pause_expert_schedules refuses
    # archived rows, and pausing first still records the pause event + stamp
    # for the archive. A nonexistent/foreign expert makes the pause a no-op
    # and the archive update below raises the 404.
    await scheduling.pause_expert_schedules(
        user_id, expert_id, reason="Expert archived"
    )
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        data={"isArchived": True},
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)
    try:
        await scheduling.detach_expert_triggers(user_id, expert_id)
    except Exception:
        # The archive itself must not fail on a scheduler hiccup: presets
        # are already deactivated first inside detach, and the run-time
        # gate refuses archived experts as the backstop.
        logger.exception(
            f"Failed to detach triggers while archiving expert #{expert_id}"
        )


# ─── Pods (owner-scoped named groups) ──────────────────────────────────

# Pods are a personal organisation aid, not a modelling primitive: a roster
# large enough to need more groups than this is not a roster any more. The cap
# also bounds what a scripted client can create.
MAX_PODS_PER_USER = 100


async def create_pod(user_id: str, name: str) -> ExpertPod:
    """Create a pod owned by *user_id*.

    The count is deliberately not serialized against the insert. This cap is a
    guardrail on a self-scoped resource, not a billed quota, so a burst of
    concurrent creates may overshoot by the burst width before the next call is
    rejected — the bound that matters (a scripted client cannot grow the table
    without limit) still holds. Making it exact would mean an advisory lock or
    row lock on every create, which is the treatment ``credit.py`` reserves for
    balances and is not warranted here.
    """
    existing = await prisma.models.ExpertPod.prisma().count(where={"userId": user_id})
    if existing >= MAX_PODS_PER_USER:
        raise ExpertPodLimitReachedError(MAX_PODS_PER_USER)
    try:
        row = await prisma.models.ExpertPod.prisma().create(
            data={"userId": user_id, "name": name}
        )
    except prisma.errors.UniqueViolationError:
        raise ExpertPodNameTakenError(name)
    return _to_pod(row)


async def list_pods(user_id: str) -> list[ExpertPod]:
    rows = await prisma.models.ExpertPod.prisma().find_many(
        where={"userId": user_id},
        order={"createdAt": "asc"},
    )
    return [_to_pod(row) for row in rows]


async def assign_pod(user_id: str, expert_id: str, pod_id: str | None) -> Expert:
    """Move a hired expert into *pod_id*, or clear it when ``None``.

    Both the expert and the target pod must belong to *user_id*; a pod owned
    by someone else is treated as not found rather than silently ignored.
    """
    if pod_id is not None:
        pod = await prisma.models.ExpertPod.prisma().find_first(
            where={"id": pod_id, "userId": user_id}
        )
        if pod is None:
            raise ExpertPodNotFoundError(pod_id)

    try:
        updated = await prisma.models.Expert.prisma().update_many(
            where={
                "id": expert_id,
                "ownerUserId": user_id,
                "isTemplate": False,
                "isArchived": False,
            },
            data={"podId": pod_id},
        )
    except prisma.errors.ForeignKeyViolationError:
        # Clearing the FK cannot violate it, so pod_id is set here: the pod was
        # deleted between the ownership check above and this write. The None
        # branch is unreachable; re-raise rather than name a pod that isn't.
        if pod_id is None:
            raise
        raise ExpertPodNotFoundError(pod_id)
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


def _to_pod(row: prisma.models.ExpertPod) -> ExpertPod:
    return ExpertPod(id=row.id, name=row.name, created_at=row.createdAt)
