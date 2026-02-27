"""
V2 External API - Library Preset Endpoints

Provides endpoints for managing agent presets (saved run configurations).
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.api.features.library.model import LibraryAgentPresetCreatable
from backend.api.features.library.model import (
    TriggeredPresetSetupRequest as _TriggeredPresetSetupRequest,
)
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import get_user_credit_model
from backend.executor import utils as execution_utils

from ..common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..models import (
    AgentGraphRun,
    AgentPreset,
    AgentPresetCreateRequest,
    AgentPresetListResponse,
    AgentPresetRunRequest,
    AgentPresetUpdateRequest,
    AgentTriggerSetupRequest,
)
from ..rate_limit import execute_limiter

logger = logging.getLogger(__name__)

presets_router = APIRouter()


@presets_router.get(
    path="/presets",
    summary="List presets",
)
async def list_presets(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
    graph_id: Optional[str] = Query(default=None, description="Filter by graph ID"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> AgentPresetListResponse:
    """
    List presets in the user's library.

    Optionally filter by graph ID to list presets for a specific agent.
    """
    result = await library_db.list_presets(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
        graph_id=graph_id,
    )

    return AgentPresetListResponse(
        presets=[AgentPreset.from_internal(p) for p in result.presets],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@presets_router.get(
    path="/presets/{preset_id}",
    summary="Get preset",
)
async def get_preset(
    preset_id: str = Path(description="Preset ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> AgentPreset:
    """
    Get details of a specific preset.
    """
    preset = await library_db.get_preset(
        user_id=auth.user_id,
        preset_id=preset_id,
    )
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset #{preset_id} not found")

    return AgentPreset.from_internal(preset)


@presets_router.post(
    path="/presets",
    summary="Create a preset",
    status_code=HTTP_201_CREATED,
)
async def create_preset(
    request: AgentPresetCreateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> AgentPreset:
    """
    Create a new preset with saved inputs and credentials for an agent.
    """
    creatable = LibraryAgentPresetCreatable(
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        name=request.name,
        description=request.description,
        inputs=request.inputs,
        credentials=request.credentials,
        is_active=request.is_active,
    )

    try:
        preset = await library_db.create_preset(
            user_id=auth.user_id,
            preset=creatable,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AgentPreset.from_internal(preset)


@presets_router.post(
    path="/presets/setup-trigger",
    summary="Setup a triggered preset",
    status_code=HTTP_201_CREATED,
)
async def setup_trigger(
    request: AgentTriggerSetupRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> AgentPreset:
    """
    Create a preset with a webhook trigger.

    This sets up a webhook that will automatically execute the preset
    when triggered by an external event.
    """
    from backend.api.features.library.routes.presets import (
        setup_trigger as _internal_setup_trigger,
    )

    internal_request = _TriggeredPresetSetupRequest(
        name=request.name,
        description=request.description,
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        trigger_config=request.trigger_config,
        agent_credentials=request.agent_credentials,
    )

    try:
        preset = await _internal_setup_trigger(
            params=internal_request,
            user_id=auth.user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AgentPreset.from_internal(preset)


@presets_router.patch(
    path="/presets/{preset_id}",
    summary="Update a preset",
)
async def update_preset(
    request: AgentPresetUpdateRequest,
    preset_id: str = Path(description="Preset ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> AgentPreset:
    """
    Update properties of a preset.

    Only the fields provided in the request body will be updated.
    """
    try:
        preset = await library_db.update_preset(
            user_id=auth.user_id,
            preset_id=preset_id,
            name=request.name,
            description=request.description,
            inputs=request.inputs,
            credentials=request.credentials,
            is_active=request.is_active,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AgentPreset.from_internal(preset)


@presets_router.delete(
    path="/presets/{preset_id}",
    summary="Delete a preset",
    status_code=HTTP_204_NO_CONTENT,
)
async def delete_preset(
    preset_id: str = Path(description="Preset ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> None:
    """
    Delete a preset.
    """
    try:
        await library_db.delete_preset(
            user_id=auth.user_id,
            preset_id=preset_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Preset #{preset_id} not found")


@presets_router.post(
    path="/presets/{preset_id}/execute",
    summary="Execute a preset",
)
async def execute_preset(
    preset_id: str = Path(description="Preset ID"),
    request: AgentPresetRunRequest = AgentPresetRunRequest(),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.RUN_AGENT)
    ),
) -> AgentGraphRun:
    """
    Execute a preset.

    Optionally override inputs and credentials from the saved preset values.
    """
    execute_limiter.check(auth.user_id)

    # Check credit balance
    user_credit_model = await get_user_credit_model(auth.user_id)
    current_balance = await user_credit_model.get_credits(auth.user_id)
    if current_balance <= 0:
        raise HTTPException(
            status_code=402,
            detail="Insufficient balance to execute the agent. Please top up your account.",
        )

    # Fetch preset
    preset = await library_db.get_preset(
        user_id=auth.user_id,
        preset_id=preset_id,
    )
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset #{preset_id} not found")

    # Merge preset inputs with overrides
    merged_inputs = {**preset.inputs, **request.inputs}
    merged_credentials = {**preset.credentials, **request.credentials_inputs}

    try:
        result = await execution_utils.add_graph_execution(
            graph_id=preset.graph_id,
            user_id=auth.user_id,
            inputs=merged_inputs,
            graph_version=preset.graph_version,
            graph_credentials_inputs=merged_credentials,
            preset_id=preset_id,
        )
        return AgentGraphRun.from_internal(result)
    except Exception as e:
        logger.error(f"Failed to execute preset: {e}")
        raise HTTPException(status_code=400, detail=str(e))
