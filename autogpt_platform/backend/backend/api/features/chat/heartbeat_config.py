"""The copilot heartbeat's settings, and a manual run for trying it out.

Served under ``/api/chat`` beside the other chat routes; the heartbeat itself
lives in ``backend.copilot.heartbeat``.
"""

from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.heartbeat import state
from backend.copilot.heartbeat.config import (
    HeartbeatConfig,
    checklist_is_empty,
    load_config,
    resolve_timezone,
    save_config,
)
from backend.copilot.heartbeat.runner import HeartbeatRunResult, run_heartbeat
from backend.copilot.heartbeat.scheduling import sync_heartbeat_schedule, wants_schedule

router = APIRouter(tags=["chat"])

# Each manual run is a model call.
MANUAL_RUN_COOLDOWN_SECONDS = 60


class HeartbeatSettingsResponse(BaseModel):
    config: HeartbeatConfig
    # The zone the active hours are read in: the config's own, else the
    # profile's, else UTC.
    effective_timezone: str
    # Whether a beat would run at all: on, with something on the checklist.
    scheduled: bool
    checklist_empty: bool


class HeartbeatSettingsUpdateResponse(HeartbeatSettingsResponse):
    # False when the scheduler could not be reached; the settings are saved
    # and the next save retries.
    schedule_synced: bool


@router.get(
    "/heartbeat",
    summary="Get heartbeat settings",
    dependencies=[Security(auth.requires_user)],
)
async def get_heartbeat_settings(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> HeartbeatSettingsResponse:
    """The caller's heartbeat settings; the defaults (off) if never saved."""
    config = await load_config(user_id)
    return await _describe(user_id, config)


@router.put(
    "/heartbeat",
    summary="Update heartbeat settings",
    dependencies=[Security(auth.requires_user)],
)
async def update_heartbeat_settings(
    config: HeartbeatConfig,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> HeartbeatSettingsUpdateResponse:
    """Replace the caller's heartbeat settings and register, refresh or remove
    their heartbeat job to match. The next beat runs whatever changed."""
    await save_config(user_id, config)
    await state.clear_last_run(user_id)
    synced = await sync_heartbeat_schedule(user_id, config)
    described = await _describe(user_id, config)
    return HeartbeatSettingsUpdateResponse(
        **described.model_dump(), schedule_synced=synced
    )


@router.post(
    "/heartbeat/run",
    summary="Run the heartbeat now",
    dependencies=[Security(auth.requires_user)],
    responses={429: {"description": "A manual run started under a minute ago"}},
)
async def run_heartbeat_now(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> HeartbeatRunResult:
    """Run one beat now and wait for it, for trying out a checklist.

    Runs even when the heartbeat is off, outside the active hours or with
    nothing new since the last beat; an empty checklist or a running turn
    still skips. An alert is delivered, suppressed and deduped like any
    scheduled one.
    """
    if not await state.claim_manual_run(user_id, MANUAL_RUN_COOLDOWN_SECONDS):
        raise HTTPException(
            status_code=429,
            detail="A heartbeat run started less than a minute ago.",
        )
    return await run_heartbeat(user_id, force=True)


async def _describe(user_id: str, config: HeartbeatConfig) -> HeartbeatSettingsResponse:
    return HeartbeatSettingsResponse(
        config=config,
        effective_timezone=await resolve_timezone(user_id, config),
        scheduled=wants_schedule(config),
        checklist_empty=checklist_is_empty(config.checklist),
    )
