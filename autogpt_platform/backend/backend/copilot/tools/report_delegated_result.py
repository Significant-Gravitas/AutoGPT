import logging
from typing import Any, Literal, cast

from pydantic import TypeAdapter, ValidationError

from backend.api.features.experts import work_items
from backend.api.features.experts.models import (
    ExpertWorkArtifact,
    ExpertWorkConfidence,
    ExpertWorkCriterion,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import (
    DelegatedWorkReportedResponse,
    ErrorResponse,
    ToolResponseBase,
)
from .run_sub_session import list_sub_workspace_files

logger = logging.getLogger(__name__)

_CRITERIA = TypeAdapter(list[ExpertWorkCriterion])
_ARTIFACTS = TypeAdapter(list[ExpertWorkArtifact])
DelegatedReportStatus = Literal["delivered", "partial", "blocked_manager", "failed"]


class ReportDelegatedResultTool(BaseTool):
    @property
    def name(self) -> str:
        return "report_delegated_result"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Report a delegated work item to AutoPilot. Only use this inside "
            "an expert thread created by delegate_to_expert. Report delivered, "
            "partial, blocked_manager, or failed; never ask the founder directly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "work_item_id": {
                    "type": "string",
                    "description": "Assigned work item id from the delegation message.",
                },
                "status": {
                    "type": "string",
                    "enum": ["delivered", "partial", "blocked_manager", "failed"],
                    "description": "Terminal handoff state reported to AutoPilot.",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise outcome, attempts, and recommended next step.",
                },
                "blocker": {
                    "type": "string",
                    "description": "What AutoPilot must resolve; required for blocked_manager.",
                    "default": "",
                },
                "progress": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 100,
                    "description": "Estimated percent complete.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["verified", "likely", "unknown", "disqualified"],
                    "default": "unknown",
                    "description": "Evidence strength for the reported result.",
                },
                "success_criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["met", "unmet", "unknown"],
                            },
                            "evidence": {"type": "string"},
                        },
                        "required": ["criterion", "status"],
                    },
                    "default": [],
                    "description": "Definition-of-done checks and supporting evidence.",
                },
                "artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "uri": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "size_bytes": {"type": "integer"},
                        },
                        "required": ["name", "uri"],
                    },
                    "default": [],
                    "description": "Persistent outputs AutoPilot can open.",
                },
            },
            "required": ["work_item_id", "status", "summary"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        work_item_id: str = "",
        status: DelegatedReportStatus = "delivered",
        summary: str = "",
        blocker: str = "",
        progress: int = 100,
        confidence: ExpertWorkConfidence = "unknown",
        success_criteria: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id or not await is_feature_enabled(
            Flag.HIRE_EXPERTS, user_id, default=False
        ):
            return ErrorResponse(
                message="Delegated expert reporting is not available.",
                session_id=session.session_id,
            )
        if (
            not session.expert_id
            or not session.session_id
            or not session.metadata.delegated_by_session_id
        ):
            return ErrorResponse(
                message="Only an expert doing delegated work can report this result.",
                session_id=session.session_id,
            )
        if status not in ("delivered", "partial", "blocked_manager", "failed"):
            return ErrorResponse(
                message="Use delivered, partial, blocked_manager, or failed.",
                session_id=session.session_id,
            )
        if not work_item_id.strip() or not summary.strip():
            return ErrorResponse(
                message="work_item_id and summary are required.",
                session_id=session.session_id,
            )
        if status == "blocked_manager" and not blocker.strip():
            return ErrorResponse(
                message="Explain what AutoPilot must resolve in blocker.",
                session_id=session.session_id,
            )
        try:
            criteria = _CRITERIA.validate_python(success_criteria or [])
            reported_artifacts = _ARTIFACTS.validate_python(artifacts or [])
        except ValidationError:
            return ErrorResponse(
                message="The criteria or artifact report is malformed.",
                session_id=session.session_id,
            )

        item = await work_items.get_work_item(work_item_id.strip(), user_id)
        if (
            item is None
            or item.expert_id != session.expert_id
            or item.delegated_session_id != session.session_id
            or item.manager_session_id != session.metadata.delegated_by_session_id
        ):
            return ErrorResponse(
                message="That work item is not assigned to this expert thread.",
                session_id=session.session_id,
            )

        final_status: DelegatedReportStatus = status
        final_blocker = blocker.strip() or None
        final_artifacts = reported_artifacts
        if item.deliverable_mode == "workspace_files":
            files = await list_sub_workspace_files(user_id, session.session_id)
            final_artifacts = [
                ExpertWorkArtifact(
                    name=file.name,
                    uri=file.path,
                    mime_type=file.mime_type,
                    size_bytes=file.size_bytes,
                )
                for file in files or []
            ]
            if status == "delivered" and not final_artifacts:
                final_status = "partial"
                final_blocker = (
                    "The required files were not promoted to the shared workspace."
                )

        updated_item, changed = await work_items.report_work_item(
            work_item_id=item.id,
            user_id=user_id,
            delegated_session_id=session.session_id,
            expert_id=session.expert_id,
            status=final_status,
            result=summary.strip()[:12_000],
            blocker=final_blocker,
            progress=progress,
            confidence=confidence,
            success_criteria=criteria or item.success_criteria,
            artifacts=final_artifacts,
        )
        if updated_item is None:
            return ErrorResponse(
                message="That work item is no longer available.",
                session_id=session.session_id,
            )

        manager_notified = False
        if changed:
            should_enqueue = await work_items.should_enqueue_parent_wake(
                item.id, user_id
            )
            manager_notified = True
            if should_enqueue:
                await run_copilot_turn_via_queue(
                    session_id=item.manager_session_id,
                    user_id=user_id,
                    message=_manager_notice(updated_item),
                    timeout=0,
                    tool_call_id=f"expert-work:{item.id}",
                    tool_name="report_delegated_result",
                )

        return DelegatedWorkReportedResponse(
            message=(
                "AutoPilot received the structured result."
                if changed
                else "AutoPilot already received this work item's terminal result."
            ),
            session_id=session.session_id,
            work_item_id=item.id,
            status=cast(DelegatedReportStatus, updated_item.status),
            manager_notified=manager_notified,
        )


def _manager_notice(item) -> str:
    parts = [
        f"[Expert work update: {item.task_title}",
        f"Status: {item.status}.",
        f"Summary: {item.result or 'No summary provided.'}",
    ]
    if item.blocker:
        parts.append(f"Manager blocker: {item.blocker}")
    if item.artifacts:
        parts.append(
            "Artifacts: " + ", ".join(artifact.name for artifact in item.artifacts)
        )
    parts.append(
        "Resolve manager-level questions yourself when possible and keep other "
        "independent work moving.]"
    )
    return " ".join(parts)
