"""Hand a task to a *different* hired expert on the user's team.

``run_sub_session`` spawns a sub that is always the caller's own scope — same
expert identity, same memory namespace, same budget. That makes it a context
isolator, not a hand-off: an expert cannot ask a teammate with different
skills, workflows, or integrations to do a piece of work.

This tool closes that gap. It creates a sub ``ChatSession`` bound to the
*target* expert and runs one turn on the same ``copilot_executor`` queue that
backs every other copilot turn, so the delegated work:

- speaks and acts as the target expert (``<expert_identity>`` is rebuilt from
  the target's Soul by the normal per-turn path),
- reads and writes the *target's* memory namespace
  (:func:`derive_memory_group_id` keys on the session's ``expert_id``),
- attributes the agent runs it starts to the *target's* weekly budget (only
  graph executions accrue weekly spend; the delegated conversation's own LLM
  cost does not), which is why a paused/archived teammate is refused here,
- inherits the caller's ``dry_run`` and LLM routing, and can only ever narrow
  the caller's tool permissions (``merged_with_parent``).

Provenance lives in the sub's session metadata rather than a new column:
``delegated_by_expert_id`` records who asked, and ``delegated_by_session_id``
doubles as the poll capability — ``get_sub_session_result`` accepts an
out-of-scope sub only when it names the caller's session there. Walking that
same chain upwards bounds how far a task may be passed on and catches an
expert handing work back to one already waiting on it
(:func:`expert_delegation.chain_refusal`, shared with ``handoff_to_expert``).
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.api.features.experts import work_items
from backend.api.features.experts.models import (
    Expert,
    ExpertWorkArtifact,
    ExpertWorkCriterion,
)
from backend.copilot.context import get_current_permissions
from backend.copilot.model import (
    ChatSession,
    child_session_origin,
    create_chat_session,
    get_chat_session,
)
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .delegated_results import DeliverableMode, delegated_response_from_outcome
from .expert_delegation import (
    chain_refusal,
    resolve_target_expert,
    safe_caller_name,
    unknown_target_message,
)
from .models import DelegatedExpertInfo, ErrorResponse, ToolResponseBase
from .run_sub_session import MAX_SUB_SESSION_WAIT_SECONDS, list_sub_workspace_files

logger = logging.getLogger(__name__)


class DelegateToExpertTool(BaseTool):
    """Delegate a task to another hired expert and wait for their answer."""

    @property
    def name(self) -> str:
        return "delegate_to_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Hand a task to a DIFFERENT expert on the user's team when it "
            "needs their skills, workflows, or integrations rather than "
            "yours. The teammate answers in their own voice, memory, and "
            "budget. Use run_sub_session instead for isolating your own "
            f"work. Waits up to wait_for_result sec (max "
            f"{MAX_SUB_SESSION_WAIT_SECONDS}); if not done, returns "
            "status=running + sub_session_id — poll via "
            "get_sub_session_result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": (
                        "Teammate to delegate to: their expert id from "
                        "<team_context>, or their exact name if you don't "
                        "have the id. Must not be you."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task, written for the teammate. Include the "
                        "context they need — they cannot see this "
                        "conversation."
                    ),
                },
                "system_context": {
                    "type": "string",
                    "description": "Optional context prepended to the prompt.",
                    "default": "",
                },
                "delegated_session_id": {
                    "type": "string",
                    "description": (
                        "Continue a prior delegation to this teammate; empty = new."
                    ),
                    "default": "",
                },
                "wait_for_result": {
                    "type": "integer",
                    "description": (
                        "Seconds to wait inline. 0 = return immediately. "
                        f"Clamped to {MAX_SUB_SESSION_WAIT_SECONDS}."
                    ),
                    "default": 60,
                },
                "deliverable_mode": {
                    "type": "string",
                    "enum": ["message", "workspace_files"],
                    "description": (
                        "Use workspace_files when completion requires one or "
                        "more persistent files; otherwise use message."
                    ),
                    "default": "message",
                },
                "task_title": {
                    "type": "string",
                    "description": "Short founder-readable title for this work item.",
                    "default": "",
                },
                "project_phase": {
                    "type": "string",
                    "description": "Current project phase this work belongs to.",
                    "default": "",
                },
                "expected_deliverable": {
                    "type": "string",
                    "description": "The concrete outcome or files the expert must return.",
                    "default": "",
                },
                "success_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Observable conditions that define done.",
                    "default": [],
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Work or decisions this task depends on.",
                    "default": [],
                },
                "source_artifacts": {
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
                    "description": "Persistent inputs the expert can open.",
                    "default": [],
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Scope, quality, timing, or implementation constraints.",
                    "default": [],
                },
                "approval_boundaries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actions that still require manager or founder approval.",
                    "default": [],
                },
                "estimate_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10080,
                    "description": "Best estimate for this attempt in minutes.",
                },
            },
            "required": ["expert_id", "prompt"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        prompt: str = "",
        system_context: str = "",
        delegated_session_id: str = "",
        wait_for_result: int = 60,
        deliverable_mode: DeliverableMode = "message",
        task_title: str = "",
        project_phase: str = "",
        expected_deliverable: str = "",
        success_criteria: list[str] | None = None,
        dependencies: list[str] | None = None,
        source_artifacts: list[dict[str, Any]] | None = None,
        constraints: list[str] | None = None,
        approval_boundaries: list[str] | None = None,
        estimate_minutes: int | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        target_id = expert_id.strip()
        if not target_id:
            return self._error("expert_id is required", session)
        if not prompt.strip():
            return self._error("prompt is required", session)
        if user_id is None:
            return self._error("Authentication required", session)
        if deliverable_mode not in ("message", "workspace_files"):
            return self._error("deliverable_mode is invalid", session)
        if not session.session_id:
            return self._error("This chat cannot manage delegated work.", session)
        if target_id == session.expert_id:
            return self._error(
                "You are that expert — do the work yourself, or use "
                "run_sub_session to isolate it in a fresh context.",
                session,
            )

        target = await self._load_delegate_target(user_id, target_id, session)
        if isinstance(target, ErrorResponse):
            return target
        if target.id == session.expert_id:
            # A name reference can resolve back to the caller even though the
            # raw-id self check above passed.
            return self._error(
                "You are that expert — do the work yourself, or use "
                "run_sub_session to isolate it in a fresh context.",
                session,
            )

        refusal = await chain_refusal(user_id, session, target)
        if refusal is not None:
            return self._error(refusal, session)

        inner_session_id = await self._resolve_session(
            user_id=user_id,
            session=session,
            target=target,
            delegated_session_id=delegated_session_id.strip(),
            deliverable_mode=deliverable_mode,
        )
        if isinstance(inner_session_id, ErrorResponse):
            return inner_session_id

        caller = await self._caller_name(user_id, session.expert_id)
        criteria = [
            ExpertWorkCriterion(criterion=value.strip())
            for value in success_criteria or []
            if isinstance(value, str) and value.strip()
        ]
        artifacts = _source_artifacts(source_artifacts or [])
        timeout = max(0, min(wait_for_result, MAX_SUB_SESSION_WAIT_SECONDS))
        work_item = await work_items.create_work_item(
            user_id=user_id,
            expert_id=target.id,
            manager_session_id=session.session_id,
            delegated_session_id=inner_session_id,
            project_phase=project_phase.strip(),
            task_title=_task_title(task_title, prompt),
            expected_deliverable=(expected_deliverable.strip() or prompt.strip()),
            deliverable_mode=deliverable_mode,
            success_criteria=criteria,
            dependencies=_clean_strings(dependencies),
            source_artifacts=artifacts,
            constraints=_clean_strings(constraints),
            approval_boundaries=_clean_strings(approval_boundaries),
            estimate_minutes=(
                max(1, min(estimate_minutes, 10_080))
                if isinstance(estimate_minutes, int)
                else None
            ),
            manager_wait_expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=timeout)
                if timeout > 0
                else None
            ),
        )
        await work_items.mark_work_started(work_item.id, user_id)
        started_at = time.monotonic()
        outcome, result = await run_copilot_turn_via_queue(
            session_id=inner_session_id,
            user_id=user_id,
            message=_handoff_message(
                caller,
                system_context,
                prompt,
                deliverable_mode=deliverable_mode,
                work_item=work_item,
            ),
            timeout=timeout,
            permissions=get_current_permissions(),
            tool_call_id=(
                f"delegate:{session.session_id}" if session.session_id else "delegate"
            ),
            tool_name="delegate_to_expert",
        )
        elapsed = time.monotonic() - started_at
        workspace_files = (
            await list_sub_workspace_files(user_id, inner_session_id)
            if outcome == "completed"
            else None
        )
        response = delegated_response_from_outcome(
            outcome=outcome,
            result=result,
            inner_session_id=inner_session_id,
            parent_session_id=session.session_id,
            elapsed=elapsed,
            workspace_files=workspace_files,
            deliverable_mode=deliverable_mode,
            work_item_id=work_item.id,
            expert=DelegatedExpertInfo(
                id=target.id,
                name=target.name,
                role=target.role,
                avatar_url=target.avatar_url,
                color=target.color,
            ),
        )
        await _record_work_outcome(
            work_item.id,
            user_id,
            response,
            parent_seen=outcome
            in {"completed", "failed", "rejected_concurrent_turn_cap"},
        )
        return response

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)

    async def _load_delegate_target(
        self, user_id: str, target_id: str, session: ChatSession
    ) -> Expert | ErrorResponse:
        """Resolve the teammate, refusing anyone who can't safely take work."""
        try:
            target = await resolve_target_expert(user_id, target_id)
        except Exception as e:
            logger.warning(f"Delegate target lookup failed for {target_id}: {e}")
            return self._error(
                "Could not reach that expert right now. Try again.", session
            )
        if target is None or target.is_archived:
            return self._error(
                await unknown_target_message(user_id, target_id, session.expert_id),
                session,
            )
        if target.schedules_paused_at is not None:
            return self._error(
                f"{target.name} is paused (budget guardrail or archive) and "
                "cannot take delegated work until the user resumes them.",
                session,
            )
        return target

    async def _resolve_session(
        self,
        *,
        user_id: str,
        session: ChatSession,
        target: Expert,
        delegated_session_id: str,
        deliverable_mode: DeliverableMode,
    ) -> str | ErrorResponse:
        """Reuse a prior delegation thread with this teammate, or open one.

        Resuming is restricted to threads this session itself delegated, so a
        session can never read or steer another scope's conversation by
        guessing an id.
        """
        if not delegated_session_id:
            new_session = await create_chat_session(
                user_id,
                dry_run=session.dry_run,
                llm_auth_provider=session.metadata.llm_auth_provider,
                llm_credential_id=session.metadata.llm_credential_id,
                expert_id=target.id,
                delegated_by_expert_id=session.expert_id,
                delegated_by_session_id=session.session_id,
                delegated_deliverable_mode=deliverable_mode,
                origin=child_session_origin(session.metadata),
            )
            return new_session.session_id

        prior = await get_chat_session(delegated_session_id)
        if (
            prior is None
            or prior.user_id != user_id
            or prior.expert_id != target.id
            or prior.metadata.delegated_by_session_id != session.session_id
            or prior.metadata.handed_off_from_expert_id is not None
        ):
            return self._error(
                f"delegated_session_id {delegated_session_id} is not a "
                f"delegation you started with {target.name}. Leave it empty "
                "to open a fresh one.",
                session,
            )
        if (
            prior.metadata.llm_auth_provider != session.metadata.llm_auth_provider
            or prior.metadata.llm_credential_id != session.metadata.llm_credential_id
        ):
            return self._error(
                f"That delegation thread with {target.name} runs on a "
                "different model connection than this chat does, so it "
                "cannot be resumed from here. Leave delegated_session_id "
                "empty to open a fresh one.",
                session,
            )
        if prior.metadata.delegated_deliverable_mode != deliverable_mode:
            return self._error(
                "That delegation thread uses a different deliverable mode. "
                "Leave delegated_session_id empty to start a fresh task.",
                session,
            )
        return delegated_session_id

    async def _caller_name(self, user_id: str, caller_expert_id: str | None) -> str:
        """Who to introduce the hand-off as. Plain sessions are AutoPilot."""
        if caller_expert_id is None:
            return "AutoPilot"
        try:
            caller = await experts_db().get_expert(
                user_id, caller_expert_id, include_workflows=False
            )
        except Exception as e:
            logger.warning(f"Delegating expert lookup failed: {e}")
            return "a teammate"
        return caller.name if caller else "a teammate"


def _handoff_message(
    caller: str,
    system_context: str,
    prompt: str,
    *,
    deliverable_mode: DeliverableMode,
    work_item,
) -> str:
    """Frame the task so the teammate knows a colleague — not the user — asked.

    Without this the delegated prompt reads as the user speaking, and the
    teammate would address them directly and ask follow-up questions nobody
    is there to answer.
    """
    name = safe_caller_name(caller)
    preamble = (
        f"[Delegated task from {name}, a teammate on this user's team — not "
        "the user. They cannot see your thread, so report the outcome in your "
        "final message. If the task needs something only the user can "
        "provide, say what is missing instead of guessing.]"
    )
    preamble += (
        f"\n\n[Work item: {work_item.id}\n"
        f"Phase: {work_item.project_phase or 'Unspecified'}\n"
        f"Task: {work_item.task_title}\n"
        f"Expected deliverable: {work_item.expected_deliverable}\n"
        f"Success criteria: {_lines(work_item.success_criteria)}\n"
        f"Dependencies: {_lines(work_item.dependencies)}\n"
        f"Source artifacts: {_artifact_lines(work_item.source_artifacts)}\n"
        f"Constraints: {_lines(work_item.constraints)}\n"
        f"Approval boundaries: {_lines(work_item.approval_boundaries)}\n"
        f"Estimate: {work_item.estimate_minutes or 'not provided'} minutes.\n"
        "Before ending, call report_delegated_result with this work item id. "
        "Report questions or missing context as blocked_manager, never as a "
        "founder-facing blocker.]"
    )
    if system_context.strip():
        preamble += f"\n\n[Context: {system_context.strip()}]"
    if deliverable_mode == "workspace_files":
        preamble += (
            "\n\n[Persistent files are required. Before reporting completion, "
            "promote every promised output with "
            "write_workspace_file(source_path=...). Local sandbox paths are "
            "not deliverables.]"
        )
    return f"{preamble}\n\n{prompt}"


def _task_title(title: str, prompt: str) -> str:
    candidate = title.strip() or prompt.strip().splitlines()[0]
    return candidate[:160] or "Delegated expert task"


def _clean_strings(values: list[str] | None) -> list[str]:
    return [value.strip()[:1_000] for value in values or [] if value.strip()]


def _source_artifacts(values: list[dict[str, Any]]) -> list[ExpertWorkArtifact]:
    artifacts: list[ExpertWorkArtifact] = []
    for value in values[:50]:
        try:
            artifacts.append(ExpertWorkArtifact.model_validate(value))
        except (ValueError, TypeError):
            continue
    return artifacts


def _lines(values) -> str:
    if not values:
        return "none"
    return "; ".join(
        value.criterion if isinstance(value, ExpertWorkCriterion) else str(value)
        for value in values
    )


def _artifact_lines(values: list[ExpertWorkArtifact]) -> str:
    return "; ".join(f"{item.name}: {item.uri}" for item in values) or "none"


async def _record_work_outcome(
    work_item_id: str,
    user_id: str,
    response,
    *,
    parent_seen: bool,
) -> None:
    if response.status in {"queued", "running"}:
        status = response.status
    elif response.status == "completed":
        status = "delivered"
    elif response.status == "incomplete":
        status = "partial"
    else:
        status = "failed"
    await work_items.record_delegation_outcome(
        work_item_id=work_item_id,
        user_id=user_id,
        status=status,
        result=response.summary,
        blocker="; ".join(response.blockers) or None,
        progress=100 if status == "delivered" else None,
        artifacts=[
            ExpertWorkArtifact(
                name=artifact.name,
                uri=artifact.read_path,
                mime_type=artifact.mime_type,
                size_bytes=artifact.size_bytes,
            )
            for artifact in response.artifacts
        ],
        parent_seen=parent_seen,
    )
