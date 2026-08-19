import asyncio
import logging
from collections import defaultdict
from typing import Literal, cast

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types
from prisma.enums import ResourceVisibility
from pydantic import JsonValue, ValidationError

from backend.api.features.experts import raise_attachments, scheduling
from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertIdentity,
    ExpertPod,
    ExpertRun,
    ExpertRunStatus,
    ExpertSoulFieldsPatch,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
    RaiseAttachment,
    RaiseResult,
    decode_voice_preferences,
)
from backend.api.features.library import db as library_db
from backend.api.features.orgs.db import get_user_default_team
from backend.blocks import get_output_block_ids
from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME, run_link
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
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
)
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)


def _raised_identity(name: str) -> str:
    # f-string, not str.format on a template: user names may contain { or },
    # which str.format would choke on.
    return f"I'm {name}, raised by you. I learn how you work and grow with you."


# The active cap bounds team-list fan-out. The lifetime raised-expert cap also
# bounds durable rows when users repeatedly raise and archive experts.
ACTIVE_EXPERT_LIMIT = 20
LIFETIME_RAISED_EXPERT_LIMIT = 100

_WORKFLOW_ROW_INCLUDE = {"LibraryAgent": True, "StoreListingVersion": True}
_WORKFLOW_INCLUDE = {"Workflows": {"include": _WORKFLOW_ROW_INCLUDE}}
_MAX_EXPERT_RUNS = 20


class ExpertTemplateNotFoundError(Exception):
    def __init__(self, template_id: str):
        super().__init__(f"Expert template {template_id} not found")
        self.template_id = template_id


class ExpertHireUnavailableError(Exception):
    def __init__(self, expert_id: str):
        super().__init__(expert_id)
        self.expert_id = expert_id


class ExpertPodNotFoundError(Exception):
    def __init__(self, pod_id: str):
        super().__init__(f"Pod {pod_id} not found")
        self.pod_id = pod_id


class ExpertPodNameTakenError(Exception):
    def __init__(self, name: str):
        super().__init__(f"A pod named {name!r} already exists")
        self.name = name


class ExpertPodLimitReachedError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"You can have at most {limit} pods")
        self.limit = limit


class ExpertLimitExceededError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"Active expert limit of {limit} reached")
        self.limit = limit


class RaisedExpertLifetimeLimitExceededError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"Raised expert lifetime limit of {limit} reached")
        self.limit = limit


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
        name, description = library_agent.name, library_agent.description
    else:
        name, description = None, None
    return ExpertWorkflowRef(
        id=row.id,
        store_listing_version_id=row.storeListingVersionId,
        library_agent_id=row.libraryAgentId,
        graph_id=library_agent.agentGraphId if library_agent else None,
        name=name,
        description=description,
        schedule_cron=row.scheduleCron,
        schedule_id=row.scheduleId,
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
        identity=row.identity,
        voice_preferences=voice_preferences,
        voice_samples=voice_samples,
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


async def list_templates() -> list[Expert]:
    rows = await prisma.models.Expert.prisma().find_many(
        where={"isTemplate": True, "isArchived": False},
        include=_WORKFLOW_INCLUDE,
    )
    return [_to_model(row) for row in rows]


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


async def list_experts(user_id: str) -> list[Expert]:
    rows = await prisma.models.Expert.prisma().find_many(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        },
        include=_WORKFLOW_INCLUDE,
    )
    latest_runs = await _latest_runs([row.id for row in rows])
    weekly_spends = await _weekly_spends([row.id for row in rows])
    return [
        _to_model(row, latest_runs.get(row.id), weekly_spends.get(row.id, 0))
        for row in rows
    ]


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
        SELECT "id", "name", "avatarUrl" AS "avatar_url", "role",
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


async def get_expert(
    user_id: str,
    expert_id: str,
    *,
    include_workflows: bool = True,
    include_archived: bool = False,
) -> Expert | None:
    """Fetch a hired expert owned by *user_id*.

    Set ``include_workflows=False`` to skip the ExpertWorkflow → LibraryAgent
    + StoreListingVersion joins when the caller only needs the expert's own
    columns. The returned model then always carries an empty ``workflows``
    list — never use that flag to decide whether workflows are installed.

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
    return _to_model(row, latest_runs.get(row.id), await get_weekly_spend(row.id))


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


def _to_expert_run(
    execution: prisma.models.AgentGraphExecution,
    workflow: prisma.models.ExpertWorkflow | None,
    output_type: OutputType,
    output_key: str | None,
    *,
    needs_review: bool,
) -> ExpertRun:
    listing = workflow.StoreListingVersion if workflow else None
    library_agent_id = workflow.libraryAgentId if workflow else None
    return ExpertRun(
        execution_id=execution.id,
        graph_id=execution.agentGraphId,
        agent_name=listing.name if listing else DEFAULT_AGENT_NAME,
        library_agent_id=library_agent_id,
        status=cast(ExpertRunStatus, str(execution.executionStatus).lower()),
        output_type=output_type,
        output_key=output_key,
        needs_review=needs_review,
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
        "skills": template.skills or [],
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


async def create_raised_expert(
    user_id: str,
    name: str,
    role: str | None,
    voice_preferences: str | None,
    *,
    avatar_url: str | None = None,
    color: str | None = None,
    about: str | None = None,
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
        about=about,
        weekly_budget=weekly_budget,
        skills=resolved.skill_names,
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


async def _create_raised_expert_row(
    user_id: str,
    name: str,
    role: str | None,
    voice_preferences: str | None,
    *,
    avatar_url: str | None,
    color: str | None,
    about: str | None,
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
                "identity": about or _raised_identity(name),
                "voicePreferences": voice_preferences or "",
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
        if preload.scheduleCron:
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


async def install_workflow(
    user_id: str, expert_id: str, store_listing_version_id: str
) -> ExpertWorkflowRef:
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
