from typing import Annotated, Any, Sequence

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Body, Depends, HTTPException, Security
from typing_extensions import Optional

from backend.api.features.graphs.model import (
    DeleteGraphResponse,
    SetActiveGraphVersionResponse,
    UpdateGraphResponse,
)
from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.model import CreateGraph, GraphExecutionSource, SetGraphActiveVersion
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.credit import get_credit_model
from backend.data.graph import GraphSettings
from backend.data.model import CredentialsMetaInput
from backend.data.onboarding import OnboardingStep, complete_onboarding_step
from backend.executor import utils as execution_utils
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    before_graph_activate,
    on_graph_deactivate,
)
from backend.monitoring.instrumentation import record_graph_operation
from backend.util.exceptions import GraphValidationError

# All ten routes carry tags=["graphs"], so the tag lives at the mount; nine of
# them carry only Security(requires_user), so that lives here. execute_graph
# keeps its extra Depends(enforce_payment_paywall).
router = APIRouter(dependencies=[Security(requires_user)])


@router.get(
    path="/graphs",
    summary="List user graphs",
)
async def list_graphs(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> Sequence[graph_db.GraphMeta]:
    paginated_result = await graph_db.list_graphs_paginated(
        user_id=user_id,
        page=1,
        page_size=250,
        filter_by="active",
        organization_id=ctx.org_id,
    )
    return paginated_result.graphs


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
    ctx: Annotated[RequestContext, Security(get_request_context)],
    version: int | None = None,
    for_export: bool = False,
) -> graph_db.GraphModel:
    graph = await graph_db.get_graph(
        graph_id,
        version,
        user_id=user_id,
        for_export=for_export,
        include_subgraphs=True,  # needed to construct full credentials input schema
        organization_id=ctx.org_id,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return graph


@router.get(
    path="/graphs/{graph_id}/versions",
    summary="Get all graph versions",
)
async def get_graph_all_versions(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> Sequence[graph_db.GraphModel]:
    graphs = await graph_db.get_graph_all_versions(
        graph_id, user_id=user_id, organization_id=ctx.org_id
    )
    if not graphs:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return graphs


@router.post(
    path="/graphs",
    summary="Create new graph",
)
async def create_new_graph(
    create_graph: CreateGraph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> graph_db.GraphModel:
    graph = graph_db.make_graph_model(create_graph.graph, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    graph.validate_graph(for_run=False)

    # Validate node credentials (and clear stale optional ones) BEFORE
    # persisting, so a credential issue can't leave the graph/library agent
    # half-saved. before_graph_activate may also mutate input_default; those
    # edits need to be persisted, so it must run before create_graph.
    graph = await before_graph_activate(graph, user_id=user_id)

    await graph_db.create_graph(
        graph,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
    await library_db.create_library_agent(
        graph, user_id, organization_id=ctx.org_id, team_id=ctx.team_id
    )

    return graph


@router.delete(
    path="/graphs/{graph_id}",
    summary="Delete graph permanently",
)
async def delete_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> DeleteGraphResponse:
    if active_version := await graph_db.get_graph(
        graph_id=graph_id, version=None, user_id=user_id
    ):
        await on_graph_deactivate(active_version, user_id=user_id)

    return {
        "version_counts": await graph_db.delete_graph(
            graph_id, user_id=user_id, organization_id=ctx.org_id
        )
    }


@router.put(
    path="/graphs/{graph_id}",
    summary="Update graph version",
)
async def update_graph(
    graph_id: str,
    graph: graph_db.Graph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> UpdateGraphResponse:
    if graph.id and graph.id != graph_id:
        raise HTTPException(400, detail="Graph ID does not match ID in URI")

    existing_versions = await graph_db.get_graph_all_versions(graph_id, user_id=user_id)
    if not existing_versions:
        raise HTTPException(404, detail=f"Graph #{graph_id} not found")

    graph.version = max(g.version for g in existing_versions) + 1
    current_active_version = next((v for v in existing_versions if v.is_active), None)

    graph = graph_db.make_graph_model(graph, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=False)
    graph.validate_graph(for_run=False)

    # If this new version is going to be active, validate node credentials
    # BEFORE persisting so a credential issue can't leave a half-saved version
    # behind. before_graph_activate may also clear stale optional credentials —
    # those edits must be persisted, hence the pre-save call.
    if graph.is_active:
        graph = await before_graph_activate(graph, user_id=user_id)

    new_graph_version = await graph_db.create_graph(
        graph,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )

    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = []
    if new_graph_version.is_active:
        await library_db.update_library_agent_version_and_settings(
            user_id, new_graph_version
        )
        await graph_db.set_graph_active_version(
            graph_id=graph_id, version=new_graph_version.version, user_id=user_id
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
    )
    assert new_graph_version_with_subgraphs
    return UpdateGraphResponse(
        graph=new_graph_version_with_subgraphs,
        skipped_webhook_presets=skipped_webhook_presets,
    )


@router.put(
    path="/graphs/{graph_id}/versions/active",
    summary="Set active graph version",
)
async def set_graph_active_version(
    graph_id: str,
    request_body: SetGraphActiveVersion,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> SetActiveGraphVersionResponse:
    new_active_version = request_body.active_graph_version
    new_active_graph = await graph_db.get_graph(
        graph_id, new_active_version, user_id=user_id
    )
    if not new_active_graph:
        raise HTTPException(404, f"Graph #{graph_id} v{new_active_version} not found")

    current_active_graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=user_id,
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
    )

    # Keep the library agent up to date with the new active version
    await library_db.update_library_agent_version_and_settings(
        user_id, new_active_graph
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
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> GraphSettings:
    """Update graph settings for the user's library agent."""
    library_agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph_id, user_id=user_id
    )
    if not library_agent:
        raise HTTPException(404, f"Graph #{graph_id} not found in user's library")

    updated_agent = await library_db.update_library_agent(
        library_agent_id=library_agent.id,
        user_id=user_id,
        settings=settings,
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
    ctx: Annotated[RequestContext, Security(get_request_context)],
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
                detail="Insufficient balance to execute the agent. Please top up your account.",
            )

    try:
        result = await execution_utils.add_graph_execution(
            graph_id=graph_id,
            user_id=user_id,
            inputs=inputs,
            preset_id=preset_id,
            graph_version=graph_version,
            graph_credentials_inputs=credentials_inputs,
            dry_run=dry_run,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
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
        raise HTTPException(
            status_code=400,
            detail={
                "type": "validation_error",
                "message": e.message,
                # TODO: only return node-specific errors if user has access to graph
                "node_errors": e.node_errors,
            },
        )
    except Exception:
        record_graph_operation(operation="execute", status="error")
        raise
