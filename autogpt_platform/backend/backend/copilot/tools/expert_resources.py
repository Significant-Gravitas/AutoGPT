"""Manage an expert's installed workflows and credential grants from chat.

An expert session installs onto itself. Personal AutoPilot names the expert.
Credential grants are the owner's decision, so grant/revoke live in the
``expert_admin`` group and only personal AutoPilot sees them.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from backend.api.features.experts.models import ExpertCredentialRef, ExpertWorkflowRef
from backend.copilot.model import ChatSession, PendingQuestion
from backend.data.activity_event import ActivityEventDraft
from backend.data.db_accessors import chat_db, experts_db
from backend.util.exceptions import ExpertNotFoundError, NotFoundError

from .base import BaseTool
from .expert_scope import resolve_target_expert
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .utils import fetch_graph_from_store_slug

logger = logging.getLogger(__name__)

_EXPERT_ID_PARAM = {
    "type": "string",
    "description": (
        "Personal AutoPilot only: the expert to act on (see list_team). "
        "An expert session always acts on itself and must omit this."
    ),
}


class ExpertWorkflowResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_WORKFLOW
    expert_id: str
    workflow_id: str | None = None
    library_agent_id: str | None = None
    graph_id: str | None = None
    name: str | None = None


class ExpertCredentialsResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_CREDENTIALS
    expert_id: str
    granted: list[ExpertCredentialRef]


class InstallExpertWorkflowTool(BaseTool):
    @property
    def name(self) -> str:
        return "install_expert_workflow"

    @property
    def description(self) -> str:
        return (
            "Install a workflow on an expert so it can run, edit, and schedule "
            "it. Source is one of: library_agent_id (owner's library), "
            "username_agent_slug ('user/agent', marketplace), or "
            "store_listing_version_id (marketplace)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_agent_id": {"type": "string"},
                "username_agent_slug": {"type": "string"},
                "store_listing_version_id": {"type": "string"},
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        library_agent_id: str = "",
        username_agent_slug: str = "",
        store_listing_version_id: str = "",
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        sources = [
            s
            for s in (library_agent_id, username_agent_slug, store_listing_version_id)
            if s.strip()
        ]
        if len(sources) != 1:
            return ErrorResponse(
                message=(
                    "Provide exactly one of library_agent_id, "
                    "username_agent_slug, or store_listing_version_id."
                ),
                session_id=session_id,
            )
        try:
            listing_id = store_listing_version_id.strip() or None
            if username_agent_slug.strip():
                listing_id = await _listing_id_from_slug(username_agent_slug)
                if listing_id is None:
                    return ErrorResponse(
                        message=f"Marketplace agent '{username_agent_slug}' not found",
                        session_id=session_id,
                    )
            ref: ExpertWorkflowRef = await experts_db().install_workflow(
                user_id,
                target,
                library_agent_id=library_agent_id.strip() or None,
                store_listing_version_id=listing_id,
            )
        except ExpertNotFoundError:
            return ErrorResponse(
                message="This expert is no longer available.",
                error="expert_not_found",
                session_id=session_id,
            )
        except (NotFoundError, ValueError) as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        return ExpertWorkflowResponse(
            expert_id=target,
            workflow_id=ref.id,
            library_agent_id=ref.library_agent_id,
            graph_id=ref.graph_id,
            name=ref.name,
            message=(
                f"Installed '{ref.name or 'workflow'}' on expert {target}. "
                f"Run it with run_agent using library_agent_id='{ref.library_agent_id}'."
            ),
            session_id=session_id,
        )


async def _listing_id_from_slug(slug: str) -> str | None:
    if "/" not in slug:
        return None
    username, agent_name = slug.strip().split("/", 1)
    _, details = await fetch_graph_from_store_slug(username, agent_name)
    return details.store_listing_version_id if details is not None else None


class RemoveExpertWorkflowTool(BaseTool):
    @property
    def name(self) -> str:
        return "remove_expert_workflow"

    @property
    def description(self) -> str:
        return (
            "Uninstall a workflow from an expert by workflow_id or "
            "library_agent_id. The agent stays in the owner's library."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "library_agent_id": {"type": "string"},
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        workflow_id: str = "",
        library_agent_id: str = "",
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        expert = await experts_db().get_expert(user_id, target)
        if expert is None:
            return ErrorResponse(
                message="This expert is no longer available.",
                error="expert_not_found",
                session_id=session_id,
            )
        selectors = [v.strip() for v in (workflow_id, library_agent_id) if v.strip()]
        if len(selectors) != 1:
            return ErrorResponse(
                message="Provide exactly one of workflow_id or library_agent_id.",
                session_id=session_id,
            )
        row = next(
            (
                w
                for w in expert.workflows
                if w.id == workflow_id.strip()
                or (
                    library_agent_id.strip()
                    and w.library_agent_id == library_agent_id.strip()
                )
            ),
            None,
        )
        if row is None:
            return ErrorResponse(
                message="That workflow is not installed on this expert.",
                error="workflow_not_installed",
                session_id=session_id,
            )
        try:
            await experts_db().remove_workflow(user_id, target, row.id)
        except (ExpertNotFoundError, NotFoundError) as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        return ExpertWorkflowResponse(
            expert_id=target,
            workflow_id=row.id,
            library_agent_id=row.library_agent_id,
            graph_id=row.graph_id,
            name=row.name,
            message=f"Removed '{row.name or 'workflow'}' from expert {target}.",
            session_id=session_id,
        )


class GrantExpertCredentialTool(BaseTool):
    @property
    def name(self) -> str:
        return "grant_expert_credential"

    @property
    def description(self) -> str:
        return (
            "Let an expert use one of the owner's integration credentials. "
            "credential_id comes from a run's missing-credentials hint or the "
            "expert's Integrations page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {"type": "string", "description": "Expert to grant to."},
                "credential_id": {"type": "string"},
            },
            "required": ["expert_id", "credential_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str = "",
        credential_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        return await _change_grant(
            user_id, session, expert_id, credential_id, grant=True
        )


class RevokeExpertCredentialTool(BaseTool):
    @property
    def name(self) -> str:
        return "revoke_expert_credential"

    @property
    def description(self) -> str:
        return "Stop an expert from using one of the owner's integration credentials."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": "Expert to revoke from.",
                },
                "credential_id": {"type": "string"},
            },
            "required": ["expert_id", "credential_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str = "",
        credential_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        return await _change_grant(
            user_id, session, expert_id, credential_id, grant=False
        )


async def _change_grant(
    user_id: str | None,
    session: ChatSession,
    expert_id: str,
    credential_id: str,
    *,
    grant: bool,
) -> ToolResponseBase:
    session_id = session.session_id
    if not user_id:
        return ErrorResponse(message="Authentication required", session_id=session_id)
    target = await resolve_target_expert(user_id, session, expert_id.strip() or None)
    if isinstance(target, ErrorResponse):
        return target
    if not credential_id.strip():
        return ErrorResponse(message="credential_id is required", session_id=session_id)
    try:
        if grant:
            granted = await experts_db().grant_expert_credentials(
                user_id, target, [credential_id.strip()]
            )
        else:
            granted = await experts_db().revoke_expert_credential(
                user_id, target, credential_id.strip()
            )
    except ExpertNotFoundError:
        return ErrorResponse(
            message="This expert is no longer available.",
            error="expert_not_found",
            session_id=session_id,
        )
    except ValueError as exc:
        return ErrorResponse(message=str(exc), session_id=session_id)
    verb = "Granted" if grant else "Revoked"
    return ExpertCredentialsResponse(
        expert_id=target,
        granted=granted,
        message=(
            f"{verb} credential {credential_id} for expert {target}. "
            f"It now has {len(granted)} granted credential(s)."
        ),
        session_id=session_id,
    )


class ExpertWorkflowsResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_WORKFLOWS
    expert_id: str
    workflows: list[ExpertWorkflowRef]


class ListExpertWorkflowsTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_expert_workflows"

    @property
    def description(self) -> str:
        return (
            "List the workflows installed on an expert — the only ones it can "
            "run, edit, or schedule."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expert_id": _EXPERT_ID_PARAM},
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        expert = await experts_db().get_expert(user_id, target)
        if expert is None:
            return ErrorResponse(
                message="This expert is no longer available.",
                error="expert_not_found",
                session_id=session_id,
            )
        lines = "; ".join(
            f"{w.name or 'Unnamed'} (workflow_id={w.id}, "
            f"library_agent_id={w.library_agent_id})"
            for w in expert.workflows
        )
        return ExpertWorkflowsResponse(
            expert_id=target,
            workflows=expert.workflows,
            message=(
                f"{len(expert.workflows)} workflow(s) installed on {expert.name}: "
                f"{lines or 'none'}. Add more with install_expert_workflow."
            ),
            session_id=session_id,
        )


class ListExpertCredentialsTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_expert_credentials"

    @property
    def description(self) -> str:
        return (
            "List the integration credentials an expert has been granted — the "
            "only ones its workflows, blocks, and MCP tools can use."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expert_id": _EXPERT_ID_PARAM},
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        try:
            granted = await experts_db().list_expert_credentials(user_id, target)
        except ExpertNotFoundError:
            return ErrorResponse(
                message="This expert is no longer available.",
                error="expert_not_found",
                session_id=session_id,
            )
        lines = "; ".join(
            f"{c.title} ({c.provider}, credential_id={c.credential_id})"
            for c in granted
        )
        return ExpertCredentialsResponse(
            expert_id=target,
            granted=granted,
            message=(
                f"Expert {target} can use {len(granted)} credential(s): "
                f"{lines or 'none'}. Platform system credentials are always available."
            ),
            session_id=session_id,
        )


class CredentialGrantRequestedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_CREDENTIALS
    expert_id: str
    credential_id: str
    provider: str | None = None


class RequestCredentialGrantTool(BaseTool):
    """An expert asks the owner for one of the account's credentials.

    The request is recorded as this chat's pending question, so it appears
    on Home under "Needs you", and as an integration activity event. Only the
    owner can grant it — on the expert's Integrations page or from personal
    AutoPilot with grant_expert_credential.
    """

    @property
    def name(self) -> str:
        return "request_credential_grant"

    @property
    def description(self) -> str:
        return (
            "Ask the owner to grant this expert an existing account credential "
            "(credential_id from a missing-credentials hint). Ends the turn "
            "waiting on the owner; do not retry the run until they answer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "credential_id": {"type": "string"},
                "provider": {
                    "type": "string",
                    "description": "Provider slug, for the note.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence on what the credential unblocks.",
                    "maxLength": 300,
                },
            },
            "required": ["credential_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    def activity_event(
        self, session: ChatSession, result: ToolResponseBase, **kwargs
    ) -> ActivityEventDraft | None:
        if not isinstance(result, CredentialGrantRequestedResponse):
            return None
        return ActivityEventDraft(
            category="INTEGRATION",
            event_type="credential.grant_requested",
            title=f"Expert asked for access to a {result.provider or 'credential'}",
            expert_id=result.expert_id,
            session_id=session.session_id,
            provider=result.provider,
            object_id=result.credential_id,
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        credential_id: str = "",
        provider: str = "",
        reason: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if session.expert_id is None:
            return ErrorResponse(
                message=(
                    "Personal AutoPilot already holds every credential; use "
                    "grant_expert_credential to give one to an expert."
                ),
                error="not_an_expert_session",
                session_id=session_id,
            )
        credential_id = credential_id.strip()
        if not credential_id:
            return ErrorResponse(
                message="credential_id is required", session_id=session_id
            )
        provider_slug = provider.strip().lower() or None
        reason = reason.strip()[:300]
        text = (
            f"Grant me the {provider_slug or 'integration'} credential "
            f"(credential_id={credential_id})"
            + (f" so I can {reason}" if reason else "")
            + ". Approve it on my Integrations page, or tell personal AutoPilot "
            f"to run grant_expert_credential for expert {session.expert_id}."
        )
        asked_at = datetime.now(timezone.utc)
        try:
            await chat_db().set_session_pending_question(
                session_id, session.user_id, text, asked_at
            )
        except Exception:
            logger.exception(
                "Could not record credential grant request for session %s",
                session_id,
            )
            return ErrorResponse(
                message=(
                    "Couldn't record the request for the owner right now. "
                    "Try again in a moment."
                ),
                error="request_not_recorded",
                session_id=session_id,
            )
        session.metadata.pending_question = PendingQuestion(
            text=text, asked_at=asked_at
        )
        return CredentialGrantRequestedResponse(
            expert_id=session.expert_id,
            credential_id=credential_id,
            provider=provider_slug,
            message=(
                f"Asked the owner to grant credential {credential_id}. This shows "
                "on Home under Needs You; stop here and wait for their answer."
            ),
            session_id=session_id,
        )
