"""Trusted expert scope for workflows, credentials, and runs.

Every check here derives from ``session.expert_id`` — the persisted expert
attribution the executor loaded — never from a tool argument. A personal
AutoPilot session (``expert_id is None``) is the account owner acting and is
unrestricted; it may also name one of the owner's experts to manage that
expert's resources.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.model import Credentials
from backend.integrations.credentials_store import is_system_credential

from .models import AgentSavedResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

WORKFLOW_NOT_INSTALLED = (
    "'{name}' is not installed on this expert. Experts can only run, edit, "
    "and schedule their installed workflows. Install it first with "
    "install_expert_workflow, from the marketplace or the owner's library."
)
EXPERT_OWNER_DENIED = (
    "Experts can only manage their own workflows and integrations. Open "
    "personal AutoPilot to manage another expert."
)
EXPERT_REQUIRED = (
    "Name the expert with expert_id. Personal AutoPilot manages experts' "
    "resources on their behalf; list_team shows their ids."
)


class ExpertWorkflowScope(BaseModel):
    """The workflows an expert may run, edit, and schedule."""

    expert_id: str
    library_agent_ids: list[str] = Field(default_factory=list)
    graph_ids: list[str] = Field(default_factory=list)

    def allows_graph(self, graph_id: str) -> bool:
        return graph_id in self.graph_ids

    def allows_agent(
        self, *, library_agent_id: str | None, graph_id: str | None
    ) -> bool:
        return (library_agent_id in self.library_agent_ids) or (
            graph_id is not None and self.allows_graph(graph_id)
        )


async def expert_workflow_scope(user_id: str, expert_id: str) -> ExpertWorkflowScope:
    """Installed workflows of *expert_id*. Fails closed to an empty scope."""
    expert = await experts_db().get_expert(user_id, expert_id)
    if expert is None:
        return ExpertWorkflowScope(expert_id=expert_id)
    return ExpertWorkflowScope(
        expert_id=expert_id,
        library_agent_ids=[
            w.library_agent_id for w in expert.workflows if w.library_agent_id
        ],
        graph_ids=[w.graph_id for w in expert.workflows if w.graph_id],
    )


async def session_workflow_scope(
    user_id: str, session: ChatSession
) -> ExpertWorkflowScope | None:
    """``None`` for personal AutoPilot: every workflow in the account."""
    if session.expert_id is None:
        return None
    return await expert_workflow_scope(user_id, session.expert_id)


async def require_installed_workflow(
    user_id: str,
    session: ChatSession,
    *,
    graph_id: str | None = None,
    library_agent_id: str | None = None,
    name: str,
) -> ErrorResponse | None:
    """Refuse a workflow an expert session has not installed.

    Either identifier form satisfies the check, so callers that accept a
    library agent id or a graph id can pass the value as both.
    """
    scope = await session_workflow_scope(user_id, session)
    if scope is None or scope.allows_agent(
        library_agent_id=library_agent_id, graph_id=graph_id
    ):
        return None
    return ErrorResponse(
        message=WORKFLOW_NOT_INSTALLED.format(name=name),
        error="workflow_not_installed",
        session_id=session.session_id,
    )


async def resolve_target_expert(
    user_id: str, session: ChatSession, requested_expert_id: str | None
) -> str | ErrorResponse:
    """Which expert a management tool call acts on.

    An expert session always acts on itself and may not name another expert.
    Personal AutoPilot must name one of the owner's active experts.
    """
    session_id = session.session_id
    if session.expert_id is not None:
        if requested_expert_id and requested_expert_id != session.expert_id:
            return ErrorResponse(
                message=EXPERT_OWNER_DENIED,
                error="access_denied",
                session_id=session_id,
            )
        return session.expert_id
    if not requested_expert_id:
        return ErrorResponse(
            message=EXPERT_REQUIRED, error="expert_required", session_id=session_id
        )
    expert = await experts_db().get_expert(
        user_id, requested_expert_id, include_workflows=False
    )
    if expert is None:
        return ErrorResponse(
            message=f"Expert '{requested_expert_id}' was not found on this account.",
            error="expert_not_found",
            session_id=session_id,
        )
    return expert.id


async def install_saved_agent(
    user_id: str, session: ChatSession, result: ToolResponseBase
) -> ToolResponseBase:
    """An agent an expert just built becomes one of its installed workflows.

    Keeps the invariant that an expert only ever runs installed workflows
    without making it re-request its own creation. Failure is reported in
    the response message, never raised: the agent is already saved.
    """
    if session.expert_id is None or not isinstance(result, AgentSavedResponse):
        return result
    try:
        await experts_db().install_workflow(
            user_id, session.expert_id, library_agent_id=result.library_agent_id
        )
    except Exception:
        logger.exception(
            f"Failed to install created agent {result.library_agent_id} "
            f"on expert {session.expert_id}"
        )
        return result.model_copy(
            update={
                "message": (
                    f"{result.message} The agent was saved but could not be "
                    "installed on this expert; run install_expert_workflow "
                    f"with library_agent_id='{result.library_agent_id}'."
                )
            }
        )
    return result.model_copy(
        update={"message": f"{result.message} Installed on this expert."}
    )


def provider_slug(value: object) -> str:
    """The wire value of a provider, whether it arrives as the ``ProviderName``
    enum (whose ``str()`` is ``ProviderName.X``) or as a plain string."""
    return str(value.value) if isinstance(value, Enum) else str(value)


async def _ungranted_credentials(
    user_id: str, expert_id: str, providers: set[str]
) -> list[Credentials]:
    """Account credentials for *providers* that *expert_id* cannot use yet."""
    from backend.integrations.creds_manager import IntegrationCredentialsManager

    try:
        owned = await IntegrationCredentialsManager().store.get_all_creds(user_id)
        allowed = set(
            await experts_db().expert_allowed_credential_ids(user_id, expert_id)
        )
    except Exception:
        logger.warning("Could not resolve ungranted credentials", exc_info=True)
        return []
    return [
        c
        for c in owned
        if provider_slug(c.provider) in providers
        and c.id not in allowed
        and not is_system_credential(c.id)
    ]


async def annotate_expert_grants(
    user_id: str, expert_id: str | None, missing: dict[str, Any]
) -> dict[str, Any]:
    """Tell the setup card which expert is asking and what it could be granted.

    Each missing credential gains an ``expert_grant`` entry so the card can
    offer "Grant access" for an account credential the expert lacks, and can
    grant a freshly connected one to the expert instead of leaving it
    account-only. Returns a new mapping; personal AutoPilot passes through.
    """
    if expert_id is None or not missing:
        return missing
    providers = {str(entry.get("provider", "")) for entry in missing.values()} - {""}
    candidates = await _ungranted_credentials(user_id, expert_id, providers)
    return {
        key: {
            **entry,
            "expert_grant": {
                "expert_id": expert_id,
                "credentials": [
                    {
                        "id": c.id,
                        "title": c.title or str(c.provider),
                        "type": str(c.type),
                    }
                    for c in candidates
                    if str(c.provider) == str(entry.get("provider", ""))
                ],
            },
        }
        for key, entry in missing.items()
    }


async def ungranted_credential_hint(
    user_id: str, expert_id: str | None, providers: set[str]
) -> str:
    """Point at credentials the account already has but the expert lacks.

    Appended to a missing-credentials message in expert chats so the user is
    asked to grant an existing integration instead of connecting a duplicate
    the expert still could not use.
    """
    if expert_id is None or not providers:
        return ""
    candidates = await _ungranted_credentials(user_id, expert_id, providers)
    if not candidates:
        return ""
    lines = "\n".join(
        f"- {c.title or c.provider} ({c.provider}, credential_id={c.id})"
        for c in candidates
    )
    return (
        "\n\nThe account already has matching credentials that this expert has "
        f"not been granted:\n{lines}\nAsk the user to grant one on the expert's "
        "Integrations page, or from personal AutoPilot with "
        "grant_expert_credential."
    )
