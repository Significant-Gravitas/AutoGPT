"""
V2 External API - Library Preset Endpoints

Provides endpoints for managing agent presets (saved run configurations) and
the webhook triggers built on them.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Response, Security
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.features.library import db as library_db
from backend.api.features.library import triggers
from backend.api.features.library.model import (
    LibraryAgentPreset,
    LibraryAgentPresetCreatable,
)
from backend.data import graph as graph_db
from backend.executor import utils as execution_utils
from backend.util.exceptions import ConflictError, ExpertRunPausedError, NotFoundError

from ..errors import ErrorResponse
from ..idempotency import (
    idempotency_key,
    idempotent_run,
    replayed_run,
    request_fingerprint,
)
from ..models import (
    AgentGraphRun,
    AgentPreset,
    AgentPresetCreateRequest,
    AgentPresetRunRequest,
    AgentPresetUpdateRequest,
    AgentTriggerSetupRequest,
)
from ..pagination import Page, PageRequest, page_request
from ..rate_limit import enforce, graph_exec_limiter
from ..tenancy import (
    TenantContext,
    in_tenant,
    require_paid_permission,
    require_permission,
)
from .helpers import assert_can_pay, assert_inputs_match

logger = logging.getLogger(__name__)

presets_router = APIRouter(tags=["presets"])


@presets_router.get(
    path="/presets",
    summary="List agent execution presets",
    operation_id="listAgentRunPresets",
)
async def list_presets(
    graph_id: Optional[str] = Query(default=None, description="Filter by graph ID"),
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_LIBRARY)),
) -> Page[AgentPreset]:
    """List presets in the user's library, optionally filtered by graph ID."""
    result = await library_db.list_presets(
        user_id=auth.user_id,
        page=page.page,
        page_size=page.limit,
        graph_id=graph_id,
        organization_id=auth.organization_id,
    )

    return page.paged(
        [AgentPreset.from_internal(p) for p in result.presets],
        total_count=result.pagination.total_items,
    )


@presets_router.get(
    path="/presets/{preset_id}",
    summary="Get agent execution preset",
    operation_id="getAgentRunPreset",
)
async def get_preset(
    preset_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_LIBRARY)),
) -> AgentPreset:
    """Get details of a specific preset."""
    return AgentPreset.from_internal(await _preset_in_tenant(preset_id, auth))


@presets_router.post(
    path="/presets",
    summary="Create agent execution preset",
    operation_id="createAgentRunPreset",
    status_code=status.HTTP_201_CREATED,
)
async def create_preset(
    request: AgentPresetCreateRequest,
    auth: TenantContext = Security(
        require_paid_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> AgentPreset:
    """Create a new preset with saved inputs and credentials for an agent.

    The inputs may leave out required ones, which a run then supplies.
    """
    graph = await _graph_in_tenant(request.graph_id, request.graph_version, auth)
    assert_inputs_match(graph.input_schema, request.inputs, partial=True)

    preset = await library_db.create_preset(
        user_id=auth.user_id,
        preset=LibraryAgentPresetCreatable(
            graph_id=request.graph_id,
            graph_version=request.graph_version,
            name=request.name,
            description=request.description,
            inputs=request.inputs,
            credentials=request.credentials_inputs,
            is_active=request.is_active,
        ),
    )
    return AgentPreset.from_internal(preset)


@presets_router.post(
    path="/presets/setup-trigger",
    summary="Setup triggered preset",
    operation_id="setupAgentRunTrigger",
    status_code=status.HTTP_201_CREATED,
)
async def setup_trigger(
    request: AgentTriggerSetupRequest,
    auth: TenantContext = Security(
        require_paid_permission(
            APIKeyPermission.WRITE_LIBRARY, APIKeyPermission.RUN_AGENT
        )
    ),
) -> AgentPreset:
    """
    Create a preset with a webhook trigger for automatic execution.

    The agent's `trigger_setup_info` describes the required trigger configuration
    schema and credentials. Use it to populate `trigger_config` and
    `credentials_inputs`. For a trigger you set up by hand in the other service,
    the response's `webhook_url` is where that service must send its events.
    """
    await _graph_in_tenant(request.graph_id, request.graph_version, auth)

    preset = await triggers.setup_triggered_preset(
        user_id=auth.user_id,
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        name=request.name,
        description=request.description,
        trigger_config=request.trigger_config,
        agent_credentials=request.credentials_inputs,
    )
    return AgentPreset.from_internal(preset)


@presets_router.patch(
    path="/presets/{preset_id}",
    operation_id="updateAgentRunPreset",
    summary="Update agent execution preset",
)
async def update_preset(
    request: AgentPresetUpdateRequest,
    preset_id: str,
    auth: TenantContext = Security(
        require_paid_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> AgentPreset:
    """Update properties of a preset. Only provided fields will be updated.

    New `inputs` on a webhook trigger re-register its webhook, with the saved
    credentials unless the request brings new ones.
    """
    current = await _preset_in_tenant(preset_id, auth)

    credentials = request.credentials_inputs
    if request.inputs is not None and credentials is None:
        # Without credentials the trigger's webhook would keep the old config.
        credentials = current.credentials

    preset = await triggers.update_triggered_preset(
        user_id=auth.user_id,
        preset_id=preset_id,
        inputs=request.inputs,
        credentials=credentials,
        name=request.name,
        description=request.description,
        is_active=request.is_active,
    )
    return AgentPreset.from_internal(preset)


@presets_router.delete(
    path="/presets/{preset_id}",
    summary="Delete agent execution preset",
    operation_id="deleteAgentRunPreset",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_preset(
    preset_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> None:
    """Delete a preset. A webhook trigger's webhook is deregistered with it."""
    await _preset_in_tenant(preset_id, auth)

    await triggers.delete_preset_with_webhook_cleanup(
        user_id=auth.user_id, preset_id=preset_id
    )


@presets_router.post(
    path="/presets/{preset_id}/execute",
    summary="Execute agent preset",
    operation_id="executeAgentRunPreset",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_402_PAYMENT_REQUIRED: {
            "model": ErrorResponse,
            "description": "No active plan, or a credit balance of zero",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The `Idempotency-Key`'s run has not been recorded yet, "
            "or the expert this preset runs for is paused",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Invalid inputs, or an `Idempotency-Key` already used "
            "for a different request",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The `Idempotency-Key` cannot be checked right now",
        },
    },
)
async def execute_preset(
    response: Response,
    preset_id: str,
    request: AgentPresetRunRequest = AgentPresetRunRequest(),
    idempotency: Annotated[Optional[str], Depends(idempotency_key)] = None,
    auth: TenantContext = Security(require_paid_permission(APIKeyPermission.RUN_AGENT)),
) -> AgentGraphRun:
    """
    Execute a preset, optionally overriding saved inputs and credentials.

    Send an `Idempotency-Key` to make a retry safe, as with
    `POST /library/agents/{agent_id}/runs`.

    **Rate limit:** 60 requests per minute per user, shared with that endpoint.
    """
    await enforce(graph_exec_limiter, auth.user_id, response)

    fingerprint = request_fingerprint(
        preset_id=preset_id, request=request.model_dump(mode="json")
    )
    async with idempotent_run(idempotency, auth, fingerprint) as claim:
        if claim.existing_run_id:
            return await replayed_run(claim, auth)

        await assert_can_pay(auth)

        preset = await _preset_in_tenant(preset_id, auth)
        merged_inputs = {**preset.inputs, **request.inputs}
        merged_credentials = {**preset.credentials, **request.credentials_inputs}
        if not preset.webhook_id:
            # A trigger's inputs are its block's config, not the graph's inputs.
            graph = await _graph_in_tenant(preset.graph_id, preset.graph_version, auth)
            assert_inputs_match(graph.input_schema, merged_inputs)

        # The run belongs where the preset does, as on the internal API.
        organization_id, team_id = (
            (preset.organization_id, preset.team_id)
            if preset.organization_id
            else (auth.organization_id, auth.team_id)
        )
        try:
            result = await execution_utils.add_graph_execution(
                graph_id=preset.graph_id,
                user_id=auth.user_id,
                inputs=merged_inputs,
                graph_version=preset.graph_version,
                graph_credentials_inputs=merged_credentials,
                preset_id=preset_id,
                expert_id=preset.expert_id,
                organization_id=organization_id,
                team_id=team_id,
            )
        except ExpertRunPausedError as e:
            # A ValueError, which the global handlers would answer with a 400.
            raise ConflictError(str(e)) from e
        await claim.record(result.id)
        return AgentGraphRun.from_internal(result)


async def _preset_in_tenant(preset_id: str, auth: TenantContext) -> LibraryAgentPreset:
    """The caller's preset, or 404 if it is missing or in another organization."""
    return in_tenant(
        await library_db.get_preset(user_id=auth.user_id, preset_id=preset_id),
        auth,
        f"Preset #{preset_id}",
    )


async def _graph_in_tenant(
    graph_id: str, graph_version: int, auth: TenantContext
) -> graph_db.GraphModel:
    """404 for a graph the credentials' organization can't see: a preset takes
    its graph's organization, so this keeps it from landing in another one."""
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=graph_version,
        user_id=auth.user_id,
        organization_id=auth.organization_id,
    )
    if not graph:
        raise NotFoundError(f"Graph #{graph_id} v{graph_version} not found")
    return graph
