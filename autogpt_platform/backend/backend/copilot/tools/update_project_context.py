import logging
from typing import Any

from pydantic import ValidationError

from backend.api.features.experts.models import ProjectContextArtifact
from backend.copilot.context import get_workspace_manager
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ProjectContextUpdatedResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_ITEMS = 25
_MAX_ARTIFACTS = 50


class UpdateProjectContextTool(BaseTool):
    @property
    def name(self) -> str:
        return "update_project_context"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Create or refresh the active shared project brief. AutoPilot uses "
            "it to give every expert the approved decisions, current phase, "
            "constraints, accessible artifacts, and live ownership before "
            "delegating. Omitted fields retain their current value."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Project name."},
                "summary": {
                    "type": "string",
                    "description": "Compact approved outcome and business context.",
                },
                "phase": {"type": "string", "description": "Current phase."},
                "decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved decisions only.",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current constraints and approval boundaries.",
                },
                "artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "description": "workspace:// URI or workspace path.",
                            },
                            "purpose": {"type": "string"},
                            "verification": {
                                "type": "string",
                                "enum": [
                                    "verified",
                                    "likely",
                                    "unknown",
                                    "disqualified",
                                ],
                            },
                        },
                        "required": ["uri"],
                    },
                    "description": "Shared files experts should open before asking.",
                },
                "activate": {
                    "type": "boolean",
                    "description": "Make this the user's active project.",
                    "default": True,
                },
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        title: str | None = None,
        summary: str | None = None,
        phase: str | None = None,
        decisions: list[str] | None = None,
        constraints: list[str] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        activate: bool = True,
        **kwargs,
    ) -> ToolResponseBase:
        if not await _enabled(user_id):
            return self._error("Shared project context is not available.", session)
        if user_id is None:
            return self._error("Authentication required.", session)
        if session.expert_id is not None:
            return self._error(
                "Only AutoPilot can update the shared project brief. Report "
                "new information to AutoPilot instead.",
                session,
            )

        db = experts_db()
        existing = await db.get_manager_project_context(user_id, session.session_id)
        resolved_title = _text(title, 160) if title is not None else None
        if not resolved_title and existing is None:
            return self._error(
                "A project title is required for the first update.", session
            )
        final_title = resolved_title or (existing.title if existing else "")

        try:
            resolved_artifacts = (
                await _resolve_artifacts(user_id, session.session_id, artifacts)
                if artifacts is not None
                else (existing.artifacts if existing else [])
            )
            context = await db.upsert_project_context(
                user_id=user_id,
                manager_session_id=session.session_id,
                title=final_title,
                summary=(
                    _text(summary, 2_000)
                    if summary is not None
                    else (existing.summary if existing else "")
                ),
                phase=(
                    _text(phase, 160)
                    if phase is not None
                    else (existing.phase if existing else "")
                ),
                decisions=(
                    _clean_list(decisions)
                    if decisions is not None
                    else (existing.decisions if existing else [])
                ),
                constraints=(
                    _clean_list(constraints)
                    if constraints is not None
                    else (existing.constraints if existing else [])
                ),
                artifacts=resolved_artifacts,
                activate=activate,
            )
        except (ValueError, ValidationError) as error:
            return self._error(str(error), session)
        except Exception:
            logger.warning("Could not update shared project context", exc_info=True)
            return self._error(
                "The shared project brief could not be updated.", session
            )

        return ProjectContextUpdatedResponse(
            message=(
                f"Project brief updated for {context.title}. Experts will receive "
                "this context before working."
            ),
            session_id=session.session_id,
            title=context.title,
            phase=context.phase,
            artifact_names=[artifact.name for artifact in context.artifacts],
            decision_count=len(context.decisions),
        )

    @staticmethod
    def _error(message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)


async def _enabled(user_id: str | None) -> bool:
    if user_id is None:
        return False
    try:
        return await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False)
    except Exception:
        logger.warning("Could not resolve shared project context flag", exc_info=True)
        return False


def _text(value: str | None, maximum: int) -> str:
    if value is None:
        return ""
    return value.strip()[:maximum]


def _clean_list(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _text(value, 500)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
        if len(result) == _MAX_ITEMS:
            break
    return result


async def _resolve_artifacts(
    user_id: str,
    session_id: str,
    values: list[dict[str, Any]],
) -> list[ProjectContextArtifact]:
    manager = await get_workspace_manager(user_id, session_id)
    result: list[ProjectContextArtifact] = []
    seen: set[str] = set()
    for value in values[:_MAX_ARTIFACTS]:
        raw_uri = _text(value.get("uri"), 2_000)
        if not raw_uri:
            raise ValueError("Every project artifact needs a workspace URI or path")
        if raw_uri.startswith("workspace://"):
            file_id = raw_uri.removeprefix("workspace://").split("#", 1)[0]
            file = await manager.get_file_info(file_id)
        else:
            file = await manager.get_file_info_by_path(raw_uri)
        if file is None or file.is_deleted:
            raise ValueError(
                "A project artifact is not accessible in the shared workspace"
            )
        if file.id in seen:
            continue
        seen.add(file.id)
        result.append(
            ProjectContextArtifact(
                name=file.name,
                uri=f"workspace://{file.id}#{file.mime_type}",
                path=file.path,
                mime_type=file.mime_type,
                purpose=_text(value.get("purpose"), 500),
                verification=value.get("verification", "unknown"),
            )
        )
    return result
