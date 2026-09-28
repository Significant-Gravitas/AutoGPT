"""Otto's delegation settings, and (see ``list_delegations``) the hand-offs
it made, under ``/experts``.

Mounted by ``routes.py`` ahead of its ``/{expert_id}`` routes so these paths
are never read as an expert id.
"""

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Security

from backend.copilot import delegation_db
from backend.copilot.delegation_settings import (
    DelegationSettings,
    DelegationSettingsUpdate,
)

router = APIRouter(tags=["experts"])


@router.get("/delegation-settings", operation_id="get_delegation_settings")
async def get_delegation_settings(
    user_id: str = Security(get_user_id),
) -> DelegationSettings:
    """How Otto hands work to the team; the defaults until the user saves."""
    return await delegation_db.get_delegation_settings(user_id)


@router.put("/delegation-settings", operation_id="update_delegation_settings")
async def update_delegation_settings(
    settings: DelegationSettingsUpdate,
    user_id: str = Security(get_user_id),
) -> DelegationSettings:
    return await delegation_db.update_delegation_settings(
        user_id, DelegationSettings(**settings.model_dump())
    )
