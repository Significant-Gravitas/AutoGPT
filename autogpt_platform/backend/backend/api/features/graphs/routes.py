from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Mapping, Sequence

from autogpt_libs.auth import get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Security
from prisma.enums import GrantCapability
from typing_extensions import Optional

from backend.api.features.graphs.model import (
    DeleteGraphResponse,
    SetActiveGraphVersionResponse,
    UpdateGraphResponse,
)
from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.live_auth import live_dependency
from backend.api.live_auth import (
    requires_live_resource_permission as requires_resource_permission,
)
from backend.api.live_auth import resolve_write_team_id
from backend.api.model import CreateGraph, GraphExecutionSource, SetGraphActiveVersion
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.credit import get_credit_model
from backend.data.grants import resolve_graph_grant
from backend.data.graph import GraphSettings
from backend.data.model import CredentialsMetaInput
from backend.data.onboarding import OnboardingStep, complete_onboarding_step
from backend.data.tenancy import (
    agent_graph_attachment_barrier,
    agent_graph_attachment_barriers,
    live_resource_access_barrier,
)
from backend.executor import utils as execution_utils
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    before_graph_activate,
    clear_unowned_auto_credentials,
    on_graph_deactivate,
)
from backend.monitoring.instrumentation import record_graph_operation
from backend.util.exceptions import GraphValidationError

# Every route here is authenticated by the router and tagged "graphs" at the
# mount; execute_graph adds Depends(enforce_payment_paywall) on top.
router = APIRouter(dependencies=[Security(requires_user)])


@asynccontextmanager
async def _stable_graph_view(
    graph_id: str,
    version: int | None,
    user_id: str,
    for_export: bool,
    ctx: RequestContext,
) -> AsyncIterator[graph_db.GraphModel | None]:
    for _ in range(3):
        candidate = await graph_db.get_graph(
            graph_id,
            version,
            user_id=user_id,
            for_export=for_export,
            include_subgraphs=True,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        )
        if candidate is None:
            yield None
            return
        expected_ids = {candidate.id, *(graph.id for graph in candidate.sub_graphs)}
        async with agent_graph_attachment_barriers(expected_ids):
            locked = await graph_db.get_graph(
                graph_id,
                version,
                user_id=user_id,
                for_export=for_export,
                include_subgraphs=True,
                organization_id=ctx.org_id,
                team_id=ctx.team_id,
            )
            if locked is None:
                yield None
                return
            locked_ids = {locked.id, *(graph.id for graph in locked.sub_graphs)}
            if locked_ids != expected_ids:
                continue
            yield locked
            return
    raise HTTPException(409, detail="Graph composition changed while loading")


@asynccontextmanager
async def _live_graph_mutation(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    graph_id: str,
    access: Literal["create", "delete"],
) -> AsyncIterator[None]:
    async with live_resource_access_barrier(
        user_id, organization_id, team_id, access
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        async with agent_graph_attachment_barrier(graph_id):
            yield


async def _live_graph_write_dependency(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
) -> AsyncIterator[str | None]:
    versions = await graph_db.get_graph_all_versions(
        graph_id,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
    if not versions:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")
    target = next((version for version in versions if version.is_active), versions[0])
    target_team_id = target.team_id
    async with _live_graph_mutation(
        user_id, ctx.org_id, target_team_id, graph_id, "create"
    ):
        exact_versions = await graph_db.get_graph_all_versions(
            graph_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id=target_team_id,
        )
        if not exact_versions:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")
        yield target_team_id


async def _live_graph_delete_dependency(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.DELETE_AGENTS
        ),
    ],
) -> AsyncIterator[str | None]:
    versions = await graph_db.get_graph_all_versions(
        graph_id,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
    if not versions:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")
    target = next((version for version in versions if version.is_active), versions[0])
    target_team_id = target.team_id
    async with _live_graph_mutation(
        user_id, ctx.org_id, target_team_id, graph_id, "delete"
    ):
        exact_versions = await graph_db.get_graph_all_versions(
            graph_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id=target_team_id,
        )
        if not exact_versions:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")
        yield target_team_id


@router.get(
    path="/graphs",
    summary="List user graphs",
)
async def list_graphs(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> Sequence[graph_db.GraphMeta]:
    graphs, _ = await graph_db.list_graphs_paginated(
        user_id=user_id,
        page=1,
        page_size=250,
        filter_by="active",
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )
    return graphs


@router.get(
    path="/graphs/{graph_id}",
    summary="Get specific graph",
)
@router.get(
    path="/graphs/{graph_id}/versions/{version}",
    summary="Get graph version",
)
async def get_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
    version: int | None = None,
    for_export: bool = False,
) -> graph_db.GraphModel:
    async with _stable_graph_view(graph_id, version, user_id, for_export, ctx) as graph:
        if not graph:
            raise HTTPException(
                status_code=404, detail=f"Workflow #{graph_id} not found."
            )
        # Teammates and library members can read a graph they don't own; what
        # they get never carries the owner's stored secrets, trigger
        # subscription or identity, whatever `for_export` says.
        return _shared_if_not_owner(graph, user_id)


@router.get(
    path="/graphs/{graph_id}/versions",
    summary="Get all graph versions",
)
async def get_graph_all_versions(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> Sequence[graph_db.GraphModel]:
    async with agent_graph_attachment_barrier(graph_id):
        graphs = await graph_db.get_graph_all_versions(
            graph_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
            include_subgraphs=True,  # needed to construct full credentials input schema
        )
        if not graphs:
            raise HTTPException(
                status_code=404, detail=f"Workflow #{graph_id} not found."
            )
        return [_shared_if_not_owner(graph, user_id) for graph in graphs]


def _shared_if_not_owner(
    graph: graph_db.GraphModel, user_id: str
) -> graph_db.GraphModel:
    # An export comes back with its owner blanked, so an owner's own export
    # takes the `shared_copy` branch too. That is harmless only as long as
    # `shared_copy` removes nothing an export keeps.
    return graph if graph.user_id == user_id else graph.shared_copy()


@router.post(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/graphs",
    summary="Create new graph",
)
async def create_new_graph(
    create_graph: CreateGraph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
) -> graph_db.GraphModel:
    if (
        ctx.team_id is not None
        and create_graph.team_id is not None
        and create_graph.team_id != ctx.team_id
    ):
        raise HTTPException(
            status_code=400,
            detail="The requested team must match the active workspace context.",
        )
    # Explicit picker choice wins; otherwise fall back to the (already
    # validated) active-team context, which is org-home (None) unless a
    # legacy X-Team-Id header set it.
    if create_graph.team_id is not None:
        team_id = await resolve_write_team_id(user_id, ctx.org_id, create_graph.team_id)
    else:
        team_id = ctx.team_id

    graph = graph_db.make_graph_model(create_graph.graph, user_id)
    submitted_node_ids = graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    _validate_graph_to_save(graph, submitted_node_ids)

    async with _live_graph_mutation(user_id, ctx.org_id, team_id, graph.id, "create"):
        # Validate node credentials (and clear stale optional ones) BEFORE
        # persisting, so a credential issue can't leave the graph/library agent
        # half-saved. before_graph_activate may also mutate input_default; those
        # edits need to be persisted, so it must run before create_graph.
        graph = await before_graph_activate(graph, user_id=user_id)

        await graph_db.create_graph(
            graph,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id=team_id,
        )
        await library_db.create_library_agent(
            graph, user_id, organization_id=ctx.org_id, team_id=team_id
        )

    return graph


@router.delete(
    path="/graphs/{graph_id}",
    summary="Delete graph permanently",
)
async def delete_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.DELETE_AGENTS
        ),
    ],
    target_team_id: Annotated[
        str | None, live_dependency(_live_graph_delete_dependency)
    ],
) -> DeleteGraphResponse:
    if active_version := await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=target_team_id,
    ):
        await on_graph_deactivate(active_version, user_id=user_id)

    return {
        "version_counts": await graph_db.delete_graph(
            graph_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id_restriction=target_team_id,
        )
    }


@router.put(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/graphs/{graph_id}",
    summary="Update graph version",
)
async def update_graph(
    graph_id: str,
    graph: graph_db.Graph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    target_team_id: Annotated[
        str | None, live_dependency(_live_graph_write_dependency)
    ],
    # Sibling create endpoints (CreateGraph, CreateAPIKeyRequest,
    # ScheduleCreationRequest) carry team_id in the body; this one cannot.
    # The PUT body is the bare `Graph` payload — an existing public-API
    # contract — so wrapping it in an envelope to add a sibling field would
    # break every current client. The query param is the non-breaking
    # equivalent; keep it here rather than "unifying" it.
    team_id: Annotated[
        Optional[str],
        Query(
            description=(
                "Team to save the new version under. Must be a team in your org "
                "that you are an active member of. Omit to use the active team "
                "from the request context, falling back to the agent's current "
                "team."
            ),
        ),
    ] = None,
) -> UpdateGraphResponse:
    if graph.id and graph.id != graph_id:
        raise HTTPException(400, detail="Graph ID does not match ID in URI")
    if ctx.team_id is not None and team_id is not None and team_id != ctx.team_id:
        raise HTTPException(
            status_code=400,
            detail="The requested team must match the active workspace context.",
        )

    existing_versions = await graph_db.get_graph_all_versions(
        graph_id,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=target_team_id,
    )
    if not existing_versions:
        raise HTTPException(404, detail=f"Workflow #{graph_id} not found")

    latest_version = max(g.version for g in existing_versions)
    graph.version = latest_version + 1
    current_active_version = next((v for v in existing_versions if v.is_active), None)

    # Team resolution for the new version: an explicit picker choice wins,
    # then the (already validated) active-team context, otherwise inherit the
    # agent's existing tenant. Inheriting on omission is what keeps a team
    # agent in its team instead of falling back to org-home when no team_id
    # is supplied.
    #
    # The inherited team is only carried over when the existing version lives
    # in the org this request is acting in: a graph saved under org A's team
    # must not have that team id stamped onto a row tagged with org B (the
    # user can own a graph across orgs and switch context between saves).
    if team_id is not None:
        resolved_team_id = await resolve_write_team_id(user_id, ctx.org_id, team_id)
    elif ctx.team_id is not None:
        resolved_team_id = ctx.team_id
    else:
        inherited_from = current_active_version or existing_versions[0]
        resolved_team_id = (
            inherited_from.team_id
            if inherited_from.organization_id == ctx.org_id
            else None
        )

    requested_graph = graph
    graph = graph_db.make_graph_model(graph, user_id)
    submitted_node_ids = graph.reassign_ids(user_id=user_id, reassign_graph_id=False)
    _validate_graph_to_save(graph, submitted_node_ids)

    if await graph_db.graph_owner_id(graph_id) != user_id:
        return await _save_as_fork(
            requested_graph, latest_version, user_id, ctx, resolved_team_id
        )

    # If this new version is going to be active, validate node credentials
    # BEFORE persisting so a credential issue can't leave a half-saved version
    # behind. before_graph_activate may also clear stale optional credentials —
    # those edits must be persisted, hence the pre-save call.
    if graph.is_active:
        graph = await before_graph_activate(graph, user_id=user_id)
    else:
        await clear_unowned_auto_credentials(graph, user_id)

    new_graph_version = await graph_db.create_graph(
        graph,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=resolved_team_id,
    )

    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = []
    if new_graph_version.is_active:
        # Point the library entry in the version's scope at it: list badges
        # and team filters read org/team off the LibraryAgent row.
        await library_db.update_agent_version_in_library(
            user_id,
            new_graph_version.id,
            new_graph_version.version,
            organization_id=ctx.org_id,
            team_id=resolved_team_id,
        )
        await graph_db.set_graph_active_version(
            graph_id=graph_id,
            version=new_graph_version.version,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id=resolved_team_id,
        )
        if current_active_version:
            await on_graph_deactivate(current_active_version, user_id=user_id)

        # Migrate webhook-attached presets to the new version so that
        # existing webhook URLs continue to trigger the latest agent version.
        if new_graph_version.webhook_input_node:
            migration = await library_db.migrate_webhook_presets_to_new_version(
                user_id=user_id,
                new_graph=new_graph_version,
            )
            skipped_webhook_presets = migration.skipped_presets

    new_graph_version_with_subgraphs = await graph_db.get_graph(
        graph_id,
        new_graph_version.version,
        user_id=user_id,
        include_subgraphs=True,
        organization_id=ctx.org_id,
        team_id=resolved_team_id,
    )
    assert new_graph_version_with_subgraphs
    return UpdateGraphResponse(
        graph=new_graph_version_with_subgraphs,
        skipped_webhook_presets=skipped_webhook_presets,
    )


@router.put(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/graphs/{graph_id}/versions/active",
    summary="Set active graph version",
)
async def set_graph_active_version(
    graph_id: str,
    request_body: SetGraphActiveVersion,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    target_team_id: Annotated[
        str | None, live_dependency(_live_graph_write_dependency)
    ],
) -> SetActiveGraphVersionResponse:
    new_active_version = request_body.active_graph_version
    new_active_graph = await graph_db.get_graph(
        graph_id,
        new_active_version,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=target_team_id,
    )
    if not new_active_graph:
        raise HTTPException(
            404, f"Workflow #{graph_id} v{new_active_version} not found"
        )

    current_active_graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=target_team_id,
    )

    # Validate the new graph's credentials before flipping the active version.
    # Capture the returned graph: before_graph_activate may clear stale
    # optional credential references, which we want propagated to the library
    # agent's settings sync below.
    new_active_graph = await before_graph_activate(new_active_graph, user_id=user_id)
    # Ensure new version is the only active version
    await graph_db.set_graph_active_version(
        graph_id=graph_id,
        version=new_active_version,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=new_active_graph.team_id,
    )

    # Keep the library agent up to date with the new active version
    await library_db.update_agent_version_in_library(
        user_id,
        new_active_graph.id,
        new_active_graph.version,
        organization_id=ctx.org_id,
        team_id=new_active_graph.team_id,
    )

    if current_active_graph and current_active_graph.version != new_active_version:
        # Handle deactivation of the previously active version
        await on_graph_deactivate(current_active_graph, user_id=user_id)

    # Migrate webhook-attached presets to the new active version so that
    # existing webhook URLs continue to trigger the latest agent version.
    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = []
    if new_active_graph.webhook_input_node:
        migration = await library_db.migrate_webhook_presets_to_new_version(
            user_id=user_id,
            new_graph=new_active_graph,
        )
        skipped_webhook_presets = migration.skipped_presets

    return SetActiveGraphVersionResponse(
        skipped_webhook_presets=skipped_webhook_presets
    )


@router.patch(
    path="/graphs/{graph_id}/settings",
    summary="Update graph settings",
)
async def update_graph_settings(
    graph_id: str,
    settings: GraphSettings,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    target_team_id: Annotated[
        str | None, live_dependency(_live_graph_write_dependency)
    ],
) -> GraphSettings:
    """Update graph settings for the user's library agent."""
    library_agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph_id,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id_restriction=target_team_id,
        exact_scope=True,
    )
    if not library_agent:
        raise HTTPException(404, f"Workflow #{graph_id} not found in user's library")

    updated_agent = await library_db.update_library_agent(
        library_agent_id=library_agent.id,
        user_id=user_id,
        settings=settings,
        organization_id=ctx.org_id,
        team_id_restriction=target_team_id,
    )

    return GraphSettings.model_validate(updated_agent.settings)


@router.post(
    path="/graphs/{graph_id}/execute/{graph_version}",
    summary="Execute graph agent",
    dependencies=[Depends(enforce_payment_paywall)],
    # The route dep enforces fail-closed (503-on-blip) so a transient
    # Supabase outage surfaces as a retryable error, not a free run
    # for a paywalled user. The deep gate inside ``add_graph_execution``
    # still covers scheduled / webhook / copilot-internal runs that
    # don't pass through this route — those callers prefer fail-open
    # so background work doesn't abandon valid jobs during a blip.
    responses={
        402: {
            "description": "Payment required: NO_TIER paywall, or insufficient credit balance"
        },
        503: {"description": "Subscription state temporarily unavailable"},
    },
)
async def execute_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.EXECUTE_RESOURCES, TeamAction.EXECUTE_AGENTS
        ),
    ],
    inputs: Annotated[dict[str, Any], Body(..., embed=True, default_factory=dict)],
    credentials_inputs: Annotated[
        dict[str, CredentialsMetaInput], Body(..., embed=True, default_factory=dict)
    ],
    source: Annotated[GraphExecutionSource | None, Body(embed=True)] = None,
    graph_version: Optional[int] = None,
    preset_id: Optional[str] = None,
    dry_run: Annotated[bool, Body(embed=True)] = False,
) -> execution_db.GraphExecutionMeta:
    if not dry_run:
        credit_model = await get_credit_model(user_id, ctx.org_id)
        current_balance = await credit_model.get_credits(user_id)
        if current_balance <= 0:
            raise HTTPException(
                status_code=402,
                detail="Insufficient balance to execute the workflow. Please top up your account.",
            )

    try:
        if preset_id is not None:
            preset = await library_db.get_preset(
                user_id,
                preset_id,
                organization_id=ctx.org_id,
                team_id=ctx.team_id,
                enforce_team_scope=True,
            )
            if (
                preset is None
                or preset.graph_id != graph_id
                or (graph_version is not None and preset.graph_version != graph_version)
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Preset #{preset_id} is not available for this graph in this workspace",
                )
            if graph_version is None:
                graph_version = preset.graph_version

        library_agent = await library_db.get_library_agent_by_graph_id(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            organization_id=ctx.org_id,
            team_id_restriction=ctx.team_id,
            exact_scope=True,
        )
        execution_team_id = library_agent.team_id if library_agent else None
        authorized_scope = library_agent is not None
        if library_agent is None and ctx.team_id is not None:
            grant = await resolve_graph_grant(
                user_id,
                graph_id,
                capability=GrantCapability.EXECUTE,
                graph_version=graph_version,
                organization_id=ctx.org_id,
                team_id_restriction=ctx.team_id,
            )
            if grant is not None:
                execution_team_id = grant.principalId
                authorized_scope = True
        if not authorized_scope:
            raise HTTPException(
                status_code=404,
                detail=f"Graph #{graph_id} is not available in this workspace",
            )

        result = await execution_utils.add_graph_execution(
            graph_id=graph_id,
            user_id=user_id,
            inputs=inputs,
            preset_id=preset_id,
            graph_version=graph_version,
            graph_credentials_inputs=credentials_inputs,
            dry_run=dry_run,
            organization_id=ctx.org_id,
            team_id=execution_team_id,
            trigger=execution_db.ExecutionTrigger.MANUAL,
            trigger_ref=source,
        )
        record_graph_operation(operation="execute", status="success")
        if source == "library":
            await complete_onboarding_step(user_id, OnboardingStep.LIBRARY_RUN_AGENT)
        elif source == "builder":
            await complete_onboarding_step(user_id, OnboardingStep.BUILDER_RUN_AGENT)
        return result
    except GraphValidationError as e:
        record_graph_operation(operation="execute", status="validation_error")
        # Return structured validation errors that the frontend can parse
        # TODO: only return node-specific errors if user has access to graph
        raise HTTPException(status_code=400, detail=_validation_error_detail(e))
    except Exception:
        record_graph_operation(operation="execute", status="error")
        raise


async def _save_as_fork(
    graph: graph_db.Graph,
    forked_from_version: int,
    user_id: str,
    ctx: RequestContext,
    team_id: str | None,
) -> UpdateGraphResponse:
    """A graph ID belonging to somebody else is saved as the caller's own copy,
    in the organization and team the save was made in."""
    fork, _ = await library_db.fork_graph_in_library(
        graph,
        user_id,
        forked_from_version=forked_from_version,
        organization_id=ctx.org_id,
        team_id=team_id,
    )
    fork_with_subgraphs = await graph_db.get_graph(
        fork.id,
        fork.version,
        user_id=user_id,
        include_subgraphs=True,
        organization_id=ctx.org_id,
        team_id=team_id,
    )
    assert fork_with_subgraphs
    return UpdateGraphResponse(graph=fork_with_subgraphs, skipped_webhook_presets=[])


def _validate_graph_to_save(
    graph: graph_db.GraphModel, submitted_node_ids: Mapping[str, str]
) -> None:
    """Refuse an invalid graph with errors keyed by the node IDs the caller
    sent, not the fresh ones `reassign_ids` gave it, so the builder can mark
    the fields on its own nodes."""
    try:
        graph.validate_graph(for_run=False)
    except GraphValidationError as e:
        raise HTTPException(
            status_code=400, detail=_validation_error_detail(e, submitted_node_ids)
        ) from e


def _validation_error_detail(
    error: GraphValidationError, node_ids: Mapping[str, str] | None = None
) -> dict[str, Any]:
    node_ids = node_ids or {}
    return {
        "type": "validation_error",
        "message": error.message,
        "node_errors": {
            node_ids.get(node_id, node_id): field_errors
            for node_id, field_errors in error.node_errors.items()
        },
    }
