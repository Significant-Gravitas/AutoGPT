"""Tools for listing, updating, and deleting library agent presets.

A preset is a saved run configuration for a library agent; a webhook trigger is a
preset with an attached webhook. These tools mirror the ``/presets`` routes and
work for any preset (triggered or not). Webhook re-registration + cleanup is
delegated to the shared helpers in ``library/triggers.py`` (RPC-exposed), so the
copilot worker reuses the exact route logic.
"""

import logging
from typing import Any

from pydantic import BaseModel

from backend.api.features.library.model import LibraryAgentPreset
from backend.copilot.model import ChatSession
from backend.data.db_accessors import library_db, triggers_db
from backend.util.exceptions import (
    InvalidInputError,
    NotFoundError,
    WebhookRegistrationError,
)

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 100


def _is_in_session_scope(preset: LibraryAgentPreset, session: ChatSession) -> bool:
    return preset.expert_id == session.expert_id


async def _get_scoped_preset(
    user_id: str, preset_id: str, session: ChatSession
) -> LibraryAgentPreset | None:
    try:
        preset = await library_db().get_preset(user_id=user_id, preset_id=preset_id)
    except NotFoundError:
        return None
    if preset is None or not _is_in_session_scope(preset, session):
        return None
    return preset


async def _list_scoped_presets(
    user_id: str,
    graph_id: str | None,
    session: ChatSession,
) -> list[LibraryAgentPreset]:
    ldb = library_db()
    first_page = await ldb.list_presets(
        user_id=user_id,
        page=1,
        page_size=_MAX_PAGE_SIZE,
        graph_id=graph_id,
    )
    scoped = [
        preset for preset in first_page.presets if _is_in_session_scope(preset, session)
    ]
    total_pages = max(
        1,
        (first_page.pagination.total_items + _MAX_PAGE_SIZE - 1) // _MAX_PAGE_SIZE,
    )
    for page in range(2, total_pages + 1):
        response = await ldb.list_presets(
            user_id=user_id,
            page=page,
            page_size=_MAX_PAGE_SIZE,
            graph_id=graph_id,
        )
        scoped.extend(
            preset
            for preset in response.presets
            if _is_in_session_scope(preset, session)
        )
    return scoped


class PresetSummary(BaseModel):
    """Summary of a single preset (saved run config / webhook trigger)."""

    id: str
    name: str
    description: str = ""
    graph_id: str
    graph_version: int
    is_active: bool
    # Webhook-trigger presets only.
    webhook_id: str | None = None
    # Ingress URL for manual-setup webhooks — give to the user verbatim.
    webhook_url: str | None = None
    provider: str | None = None


class PresetListResponse(ToolResponseBase):
    """Response listing the user's presets."""

    type: ResponseType = ResponseType.PRESET_LIST
    presets: list[PresetSummary]
    total_count: int
    page: int
    page_size: int


class PresetUpdatedResponse(ToolResponseBase):
    """Response confirming a preset was updated."""

    type: ResponseType = ResponseType.PRESET_UPDATED
    preset_id: str
    name: str
    is_active: bool
    webhook_url: str | None = None


class PresetDeletedResponse(ToolResponseBase):
    """Response confirming a preset was deleted."""

    type: ResponseType = ResponseType.PRESET_DELETED
    preset_id: str
    name: str


class ListPresetsTool(BaseTool):
    """List the user's presets (saved run configs + webhook triggers).

    Optionally filtered by library agent or graph. Use this to find a
    ``preset_id`` before update_preset / delete_preset.
    """

    @property
    def name(self) -> str:
        return "list_presets"

    @property
    def description(self) -> str:
        return (
            "List the user's presets (saved run configurations and webhook "
            "triggers). Optionally filter by library_agent_id or graph_id. Use "
            "before update_preset/delete_preset to find a preset_id."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_agent_id": {
                    "type": "string",
                    "description": "Filter by library agent.",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Filter by graph.",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Page number (1-based).",
                },
                "page_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_PAGE_SIZE,
                    "default": _DEFAULT_PAGE_SIZE,
                    "description": "Presets per page (1-100).",
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        library_agent_id: str | None = kwargs.get("library_agent_id")
        graph_id: str | None = kwargs.get("graph_id")
        page = kwargs.get("page", 1)
        page_size = kwargs.get("page_size", _DEFAULT_PAGE_SIZE)
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= _MAX_PAGE_SIZE
        ):
            return ErrorResponse(
                message="page must be at least 1 and page_size must be between 1 and 100.",
                error="invalid_pagination",
                session_id=session_id,
            )

        ldb = library_db()
        if library_agent_id:
            try:
                lib_agent = await ldb.get_library_agent(
                    id=library_agent_id, user_id=user_id
                )
            except NotFoundError as e:
                return ErrorResponse(
                    message=f"Library agent not found: {e}",
                    error="library_agent_not_found",
                    session_id=session_id,
                )
            graph_id = lib_agent.graph_id

        scoped_presets = await _list_scoped_presets(user_id, graph_id, session)
        total_count = len(scoped_presets)
        start = (page - 1) * page_size
        page_presets = scoped_presets[start : start + page_size]
        presets = [
            PresetSummary(
                id=p.id,
                name=p.name,
                description=p.description or "",
                graph_id=p.graph_id,
                graph_version=p.graph_version,
                is_active=p.is_active,
                webhook_id=p.webhook_id,
                webhook_url=p.webhook.url if p.webhook else None,
                provider=p.webhook.provider if p.webhook else None,
            )
            for p in page_presets
        ]

        if not presets:
            message = (
                "No presets found."
                if total_count == 0
                else f"No presets found on page {page}; {total_count} preset(s) match."
            )
        elif start > 0 or start + len(presets) < total_count:
            message = (
                f"Showing presets {start + 1}-{start + len(presets)} "
                f"of {total_count}."
            )
        else:
            message = f"Found {total_count} preset(s)."
        return PresetListResponse(
            message=message,
            presets=presets,
            total_count=total_count,
            page=page,
            page_size=page_size,
            session_id=session_id,
        )


class UpdatePresetTool(BaseTool):
    """Update a preset: rename, pause/resume, or reconfigure its inputs."""

    @property
    def name(self) -> str:
        return "update_preset"

    @property
    def description(self) -> str:
        return (
            "Update a preset by preset_id: rename, change description, pause or "
            "resume it (is_active=false/true), or reconfigure its inputs. For a "
            "webhook trigger, 'inputs' is the trigger block's config (e.g. repo, "
            "events) and changing it re-registers the webhook with the preset's "
            "existing credentials. Find preset_id via list_presets."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "preset_id": {
                    "type": "string",
                    "description": "ID of the preset to update.",
                },
                "name": {"type": "string", "description": "New name."},
                "description": {
                    "type": "string",
                    "description": "New description.",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Set false to pause the trigger, true to resume.",
                },
                "inputs": {
                    "type": "object",
                    "description": (
                        "Inputs to change, merged over the preset's current "
                        "inputs. For a webhook trigger these are the trigger "
                        "block's config (e.g. repo, events); changing them "
                        "re-registers the webhook."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["preset_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        preset_id: str | None = kwargs.get("preset_id")
        if not preset_id:
            return ErrorResponse(
                message="preset_id is required.",
                error="missing_argument",
                session_id=session_id,
            )

        current = await _get_scoped_preset(user_id, preset_id, session)
        if current is None:
            return ErrorResponse(
                message=f"Preset '{preset_id}' not found.",
                error="preset_not_found",
                session_id=session_id,
            )

        merged_inputs = None
        credentials = None
        new_inputs = kwargs.get("inputs")
        if new_inputs:
            # Reconfigure: merge over current inputs and reuse the stored
            # credentials so the webhook can be re-registered.
            merged_inputs = {**current.inputs, **new_inputs}
            credentials = current.credentials

        try:
            updated = await triggers_db().update_triggered_preset(
                user_id=user_id,
                preset_id=preset_id,
                inputs=merged_inputs,
                credentials=credentials,
                name=kwargs.get("name"),
                description=kwargs.get("description"),
                is_active=kwargs.get("is_active"),
            )
        except NotFoundError:
            return ErrorResponse(
                message=f"Preset '{preset_id}' not found.",
                error="preset_not_found",
                session_id=session_id,
            )
        except (InvalidInputError, WebhookRegistrationError) as e:
            return ErrorResponse(
                message=str(e),
                error="preset_update_failed",
                session_id=session_id,
            )

        return PresetUpdatedResponse(
            message=f"Preset '{updated.name}' updated.",
            preset_id=updated.id,
            name=updated.name,
            is_active=updated.is_active,
            webhook_url=updated.webhook.url if updated.webhook else None,
            session_id=session_id,
        )


class DeletePresetTool(BaseTool):
    """Delete a preset, cleaning up its webhook if it's a trigger."""

    @property
    def name(self) -> str:
        return "delete_preset"

    @property
    def description(self) -> str:
        return (
            "Delete a preset by preset_id. If it's a webhook trigger, its webhook "
            "is deregistered and cleaned up. Find preset_id via list_presets."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "preset_id": {
                    "type": "string",
                    "description": "ID of the preset to delete.",
                },
            },
            "required": ["preset_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        preset_id: str | None = kwargs.get("preset_id")
        if not preset_id:
            return ErrorResponse(
                message="preset_id is required.",
                error="missing_argument",
                session_id=session_id,
            )

        # Fetch first for the name + a clean not-found message.
        current = await _get_scoped_preset(user_id, preset_id, session)
        if current is None:
            return ErrorResponse(
                message=f"Preset '{preset_id}' not found.",
                error="preset_not_found",
                session_id=session_id,
            )

        await triggers_db().delete_preset_with_webhook_cleanup(
            user_id=user_id, preset_id=preset_id
        )

        return PresetDeletedResponse(
            message=f"Preset '{current.name}' deleted.",
            preset_id=preset_id,
            name=current.name,
            session_id=session_id,
        )
