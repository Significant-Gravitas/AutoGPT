import json
import logging
from typing import Any

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.util.clients import get_scheduler_client
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExecutionStartedResponse,
    ExpertSummary,
    ExpertWorkflowInstalledResponse,
    ToolResponseBase,
)
from .run_agent import RunAgentTool, SCHEDULED_STATUS

logger = logging.getLogger(__name__)


class InstallExpertWorkflowTool(BaseTool):
    @property
    def name(self) -> str:
        return "install_expert_workflow"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Install one tested workflow. AutoPilot may target any owned "
            "expert; experts may target only themselves. Pass exactly one "
            "source. Private workflows require a run_agent dry run."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": "Target expert.",
                },
                "store_listing_version_id": {
                    "type": "string",
                    "description": "Marketplace source.",
                },
                "library_agent_id": {
                    "type": "string",
                    "description": "Private source.",
                },
                "purpose": {
                    "type": "string",
                    "description": "Reuse reason.",
                },
                "expected_inputs": {
                    "type": "string",
                    "description": "Input contract.",
                },
                "expected_outputs": {
                    "type": "string",
                    "description": "Output contract.",
                },
                "cadence": {
                    "type": "string",
                    "description": "Reuse cadence.",
                },
                "schedule_cron": {
                    "type": "string",
                    "description": "Optional cron.",
                },
                "schedule_name": {
                    "type": "string",
                    "description": "Schedule name.",
                },
                "schedule_inputs": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Scheduled inputs.",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone.",
                    "default": "UTC",
                },
                "schedule_approved": {
                    "type": "boolean",
                    "description": "Approval confirmed.",
                    "default": False,
                },
            },
            "required": [
                "purpose",
                "expected_inputs",
                "expected_outputs",
                "cadence",
            ],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        store_listing_version_id: str = "",
        library_agent_id: str = "",
        purpose: str = "",
        expected_inputs: str = "",
        expected_outputs: str = "",
        cadence: str = "",
        schedule_cron: str = "",
        schedule_name: str = "",
        schedule_inputs: dict[str, Any] | None = None,
        timezone: str = "UTC",
        schedule_approved: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id or not await is_feature_enabled(
            Flag.HIRE_EXPERTS, user_id, default=False
        ):
            return ErrorResponse(
                message="Expert workflow management is not available.",
                session_id=session.session_id,
            )

        target_id = expert_id.strip()
        if session.expert_id:
            if target_id and target_id != session.expert_id:
                return ErrorResponse(
                    message=(
                        "Experts can manage only their own workflows. Ask "
                        "AutoPilot or that teammate to make this change."
                    ),
                    session_id=session.session_id,
                )
            target_id = session.expert_id
        elif not target_id:
            return ErrorResponse(
                message="AutoPilot must choose which expert owns this workflow.",
                session_id=session.session_id,
            )

        store_source = store_listing_version_id.strip()
        library_source = library_agent_id.strip()
        if bool(store_source) == bool(library_source):
            return ErrorResponse(
                message=(
                    "Choose exactly one workflow source: marketplace or the "
                    "user's private library."
                ),
                session_id=session.session_id,
            )
        if not all(
            value.strip()
            for value in (purpose, expected_inputs, expected_outputs, cadence)
        ):
            return ErrorResponse(
                message=(
                    "Record the workflow's purpose, expected inputs, expected "
                    "outputs, and cadence before installing it."
                ),
                session_id=session.session_id,
            )
        if library_source and not _has_successful_safe_test(session, library_source):
            return ErrorResponse(
                message=(
                    "This private workflow has not passed a completed safe test "
                    "in this chat. Run it with dry_run=true and wait for the "
                    "result; fix any failed nodes before installing."
                ),
                session_id=session.session_id,
            )

        db = experts_db()
        try:
            expert = await db.get_expert(user_id, target_id, include_workflows=False)
            if expert is None or expert.is_archived:
                return ErrorResponse(
                    message="That expert is not active on this team.",
                    session_id=session.session_id,
                )
            install_kwargs = {
                "purpose": purpose.strip()[:1000],
                "expected_inputs": expected_inputs.strip()[:2000],
                "expected_outputs": expected_outputs.strip()[:2000],
                "cadence": cadence.strip()[:500],
            }
            if store_source:
                workflow = await db.install_workflow(
                    user_id,
                    target_id,
                    store_source,
                    **install_kwargs,
                )
            else:
                workflow = await db.install_library_workflow(
                    user_id,
                    target_id,
                    library_source,
                    **install_kwargs,
                )
        except Exception:
            logger.warning("Expert workflow installation failed", exc_info=True)
            return ErrorResponse(
                message="That workflow could not be installed on this expert.",
                session_id=session.session_id,
            )

        if schedule_cron.strip():
            if not schedule_name.strip():
                return ErrorResponse(
                    message="Give the schedule a short, meaningful name.",
                    session_id=session.session_id,
                )
            if not schedule_approved:
                return ErrorResponse(
                    message=(
                        "The workflow is installed, but scheduling needs scope "
                        "that covers its recurring external actions."
                    ),
                    session_id=session.session_id,
                )
            scheduled = await _schedule_for_expert(
                user_id=user_id,
                session=session,
                expert_id=target_id,
                workflow=workflow,
                schedule_name=schedule_name.strip(),
                schedule_cron=schedule_cron.strip(),
                schedule_inputs=schedule_inputs or {},
                timezone=timezone.strip() or "UTC",
            )
            if not isinstance(scheduled, ExpertWorkflowInstalledResponse):
                return scheduled
            return scheduled

        return ExpertWorkflowInstalledResponse(
            message=f"Installed {workflow.name or 'the workflow'} on {expert.name}.",
            session_id=session.session_id,
            expert=_expert_summary(expert),
            workflow=workflow,
        )


async def _schedule_for_expert(
    *,
    user_id: str,
    session: ChatSession,
    expert_id: str,
    workflow: ExpertWorkflowRef,
    schedule_name: str,
    schedule_cron: str,
    schedule_inputs: dict[str, Any],
    timezone: str,
) -> ToolResponseBase:
    if workflow.schedule_id:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False
        )
        if expert is None:
            return ErrorResponse(
                message="That expert is no longer active.",
                session_id=session.session_id,
            )
        return ExpertWorkflowInstalledResponse(
            message=f"{workflow.name or 'The workflow'} is already scheduled for {expert.name}.",
            session_id=session.session_id,
            expert=_expert_summary(expert),
            workflow=workflow,
            scheduled=True,
            schedule_status="scheduled",
        )
    if not workflow.library_agent_id:
        return ErrorResponse(
            message="The installed workflow is not ready to schedule.",
            session_id=session.session_id,
        )

    attributed_session = session.model_copy(
        deep=True,
        update={"expert_id": expert_id},
    )
    result = await RunAgentTool()._execute(
        user_id,
        attributed_session,
        library_agent_id=workflow.library_agent_id,
        inputs=schedule_inputs,
        schedule_name=schedule_name,
        cron=schedule_cron,
        timezone=timezone,
    )
    if (
        not isinstance(result, ExecutionStartedResponse)
        or result.status != SCHEDULED_STATUS
    ):
        return result

    claimed = await experts_db().claim_workflow_schedule(
        user_id,
        expert_id,
        workflow.id,
        result.execution_id,
        schedule_cron,
    )
    if not claimed:
        try:
            await get_scheduler_client().delete_schedule(
                result.execution_id, user_id=user_id
            )
        except Exception:
            logger.warning(
                "Could not remove a duplicate expert schedule", exc_info=True
            )

    expert = await experts_db().get_expert(user_id, expert_id)
    if expert is None:
        return ErrorResponse(
            message="The schedule was created, but the expert is no longer active.",
            session_id=session.session_id,
        )
    current = next(
        (candidate for candidate in expert.workflows if candidate.id == workflow.id),
        workflow,
    )
    return ExpertWorkflowInstalledResponse(
        message=f"Scheduled {current.name or 'the workflow'} for {expert.name}.",
        session_id=session.session_id,
        expert=_expert_summary(expert),
        workflow=current,
        scheduled=True,
        schedule_status="scheduled",
    )


def _expert_summary(expert: Expert) -> ExpertSummary:
    return ExpertSummary(
        id=expert.id,
        name=expert.name,
        role=expert.role,
        avatar_url=expert.avatar_url,
        color=expert.color,
    )


def _has_successful_safe_test(session: ChatSession, library_agent_id: str) -> bool:
    calls: dict[str, dict[str, Any]] = {}
    for message in session.messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for call in message.tool_calls:
            function = call.get("function", call)
            if function.get("name") != "run_agent":
                continue
            arguments = _json_object(function.get("arguments"))
            if (
                arguments.get("library_agent_id") == library_agent_id
                and arguments.get("dry_run") is True
            ):
                call_id = call.get("id")
                if isinstance(call_id, str):
                    calls[call_id] = arguments
    if not calls:
        return False

    for message in reversed(session.messages):
        if message.role != "tool" or message.tool_call_id not in calls:
            continue
        output = _json_object(message.content)
        execution = output.get("execution")
        if not isinstance(execution, dict):
            continue
        if (
            output.get("type") == "agent_output"
            and output.get("library_agent_id") == library_agent_id
            and str(execution.get("status", "")).lower() == "completed"
            and not execution.get("nodes_failed")
        ):
            return True
    return False


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
