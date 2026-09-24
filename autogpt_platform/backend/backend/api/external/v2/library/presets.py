"""
V2 External API - Library Preset Endpoints

Provides endpoints for managing agent presets (saved run configurations).
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Security
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.features.library import db as library_db
from backend.api.features.library.model import LibraryAgentPresetCreatable
from backend.api.features.library.model import (
    TriggeredPresetSetupRequest as _TriggeredPresetSetupRequest,
)
from backend.executor import utils as execution_utils

from ..idempotency import idempotency_key, idempotent_run, replayed_run
from ..models import (
    AgentGraphRun,
    AgentPreset,
    AgentPresetCreateRequest,
    AgentPresetRunRequest,
    AgentPresetUpdateRequest,
    AgentTriggerSetupRequest,
)
from ..pagination import Page, PageRequest, page_request
from ..rate_limit import graph_exec_limiter
from ..tenancy import TenantContext, in_tenant, require_permission
from .helpers import assert_can_pay

logger = logging.getLogger(__name__)

presets_router = APIRouter(tags=["library", "presets"])


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
    preset = in_tenant(
        await library_db.get_preset(user_id=auth.user_id, preset_id=preset_id),
        auth,
        f"Preset #{preset_id}",
    )

    return AgentPreset.from_internal(preset)


@presets_router.post(
    path="/presets",
    summary="Create agent execution preset",
    operation_id="createAgentRunPreset",
    status_code=status.HTTP_201_CREATED,
)
async def create_preset(
    request: AgentPresetCreateRequest,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> AgentPreset:
    """Create a new preset with saved inputs and credentials for an agent."""
    creatable = LibraryAgentPresetCreatable(
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        name=request.name,
        description=request.description,
        inputs=request.inputs,
        credentials=request.credentials_inputs,
        is_active=request.is_active,
    )

    preset = await library_db.create_preset(
        user_id=auth.user_id,
        preset=creatable,
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
        require_permission(APIKeyPermission.WRITE_LIBRARY, APIKeyPermission.RUN_AGENT)
    ),
) -> AgentPreset:
    """
    Create a preset with a webhook trigger for automatic execution.

    The agent's `trigger_setup_info` describes the required trigger configuration
    schema and credentials. Use it to populate `trigger_config` and
    `credentials_inputs`.
    """
    # Use internal trigger setup endpoint to avoid logic duplication:
    from backend.api.features.library.routes.presets import (
        setup_trigger as _internal_setup_trigger,
    )

    internal_request = _TriggeredPresetSetupRequest(
        name=request.name,
        description=request.description,
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        trigger_config=request.trigger_config,
        agent_credentials=request.credentials_inputs,
    )

    preset = await _internal_setup_trigger(
        params=internal_request,
        user_id=auth.user_id,
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
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> AgentPreset:
    """Update properties of a preset. Only provided fields will be updated."""
    await _assert_preset_in_tenant(preset_id, auth)

    preset = await library_db.update_preset(
        user_id=auth.user_id,
        preset_id=preset_id,
        name=request.name,
        description=request.description,
        inputs=request.inputs,
        credentials=request.credentials_inputs,
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
    """Delete a preset."""
    await _assert_preset_in_tenant(preset_id, auth)

    await library_db.delete_preset(
        user_id=auth.user_id,
        preset_id=preset_id,
    )


@presets_router.post(
    path="/presets/{preset_id}/runs",
    summary="Run agent preset",
    operation_id="runAgentRunPreset",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_preset(
    preset_id: str,
    request: AgentPresetRunRequest = AgentPresetRunRequest(),
    idempotency: Annotated[Optional[str], Depends(idempotency_key)] = None,
    auth: TenantContext = Security(require_permission(APIKeyPermission.RUN_AGENT)),
) -> AgentGraphRun:
    """
    Run a preset, optionally overriding its saved inputs and credentials.

    Send an `Idempotency-Key` to make a retry safe: a second request carrying the
    same key returns the run the first one started rather than starting another.

    **Rate limit:** 60 requests per minute per user.
    """
    await graph_exec_limiter.check(auth.user_id)

    async with idempotent_run(idempotency, auth.user_id) as claim:
        if claim.existing_run_id:
            return await replayed_run(claim, auth)

        await assert_can_pay(auth)

        preset = in_tenant(
            await library_db.get_preset(user_id=auth.user_id, preset_id=preset_id),
            auth,
            f"Preset #{preset_id}",
        )

        result = await execution_utils.add_graph_execution(
            graph_id=preset.graph_id,
            user_id=auth.user_id,
            inputs={**preset.inputs, **request.inputs},
            graph_version=preset.graph_version,
            graph_credentials_inputs={
                **preset.credentials,
                **request.credentials_inputs,
            },
            preset_id=preset_id,
            organization_id=auth.organization_id,
            team_id=auth.team_id,
        )
        await claim.record(result.id)
        return AgentGraphRun.from_internal(result)


async def _assert_preset_in_tenant(preset_id: str, auth: TenantContext) -> None:
    """404 before mutating a preset the credentials cannot reach."""
    in_tenant(
        await library_db.get_preset(user_id=auth.user_id, preset_id=preset_id),
        auth,
        f"Preset #{preset_id}",
    )
