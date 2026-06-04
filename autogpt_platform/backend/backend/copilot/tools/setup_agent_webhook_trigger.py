"""Tool for setting up a webhook-triggered preset for a library agent.

Wraps the same logic as the ``POST /presets/setup-trigger`` route so AutoPilot
can set up a webhook trigger end-to-end and hand the user the correct ingress
URL for manual-setup webhooks. Credentials for provider/auto-setup webhooks
must be chosen explicitly by the user — the webhook is registered under the
chosen account, so this tool never auto-matches a "fitting" credential.
"""

import logging
from typing import Any

from pydantic import BaseModel

from backend.api.features.library.model import LibraryAgentPreset
from backend.blocks._base import BlockType
from backend.copilot.model import ChatSession
from backend.data.db_accessors import graph_db, library_db, triggers_db
from backend.data.graph import GraphModel
from backend.data.model import Credentials, CredentialsMetaInput
from backend.util.exceptions import InvalidInputError, NotFoundError

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .utils import (
    create_credential_meta_from_match,
    get_or_create_library_agent,
    get_user_credentials,
)

logger = logging.getLogger(__name__)


class CredentialChoice(BaseModel):
    """An existing user credential that can be attached to the webhook."""

    id: str
    title: str | None = None
    provider: str
    type: str


class RequiredTriggerCredential(BaseModel):
    """A credential field that must be filled before the trigger can be set up."""

    field_name: str
    provider: str
    supported_types: list[str]
    options: list[CredentialChoice]


class TriggerCredentialsRequiredResponse(ToolResponseBase):
    """Returned when credentials must be selected before setting up the trigger.

    The user must pick which account each webhook is registered under, so the
    LLM should present the ``options`` and ask the user — never pick for them.
    """

    type: ResponseType = ResponseType.TRIGGER_CREDENTIALS_REQUIRED
    library_agent_id: str | None = None
    graph_id: str
    graph_version: int
    required_credentials: list[RequiredTriggerCredential]


class TriggerSetupResponse(ToolResponseBase):
    """Returned when a webhook-triggered preset has been created."""

    type: ResponseType = ResponseType.TRIGGER_SETUP
    preset_id: str
    library_agent_id: str
    library_agent_link: str
    name: str
    is_active: bool
    provider: str
    manual_setup_required: bool
    webhook_url: str | None = None


class SetupAgentWebhookTriggerTool(BaseTool):
    """Set up a webhook-triggered preset for an agent that has a trigger block.

    For manual-setup (generic) webhooks the response includes the ingress URL
    to give to the user. For provider webhooks (e.g. GitHub) the platform
    registers the webhook automatically once the user picks which account to
    use — credentials must be chosen explicitly, never auto-matched.
    """

    @property
    def name(self) -> str:
        return "setup_agent_webhook_trigger"

    @property
    def description(self) -> str:
        return (
            "Set up a webhook trigger (triggered preset) for an agent with a "
            "webhook trigger block. For manual/generic webhooks, returns the "
            "exact ingress URL to give the user — never reconstruct it. For "
            "provider webhooks (e.g. GitHub), credentials MUST be chosen "
            "explicitly: call without 'credentials' to list accounts, then "
            "call again with the user's choice."
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
                    "description": "Library agent ID (preferred identifier).",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Agent graph ID (alt to library_agent_id).",
                },
                "graph_version": {
                    "type": "integer",
                    "description": "Graph version (used with graph_id).",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the trigger.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description for the trigger.",
                },
                "trigger_config": {
                    "type": "object",
                    "description": (
                        "Webhook trigger block config inputs (from "
                        "trigger_setup_info.config_schema); omit credential "
                        "fields. Usually empty for generic webhooks."
                    ),
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": (
                        "Explicit credential choice: {field_name: credential_id}. "
                        "Required for provider webhooks; omit to list accounts."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name"],
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

        name: str = (kwargs.get("name") or "").strip()
        if not name:
            return ErrorResponse(
                message="A name for the trigger is required.",
                error="missing_argument",
                session_id=session_id,
            )

        graph, error = await self._resolve_graph(user_id, session, kwargs)
        if error:
            return error
        assert graph is not None

        if not graph.webhook_input_node:
            return ErrorResponse(
                message=(
                    f"Agent '{graph.name}' has no webhook trigger block, so it "
                    "can't have a webhook trigger. Run or schedule it instead."
                ),
                error="no_webhook_trigger",
                session_id=session_id,
            )

        agent_credentials, required = self._resolve_credentials(
            graph, kwargs.get("credentials") or {}, await get_user_credentials(user_id)
        )
        library_agent = await get_or_create_library_agent(graph, user_id)

        if required:
            return TriggerCredentialsRequiredResponse(
                message=(
                    "This trigger needs you to choose which connected account(s) "
                    "to use. Ask the user which account to use for each field "
                    "below, then call setup_agent_webhook_trigger again with "
                    "credentials={<field_name>: <credential_id>}. If a field has "
                    "no options, the user must connect an account first via "
                    "connect_integration."
                ),
                session_id=session_id,
                library_agent_id=library_agent.id,
                graph_id=graph.id,
                graph_version=graph.version,
                required_credentials=required,
            )

        try:
            preset = await triggers_db().setup_triggered_preset(
                user_id=user_id,
                graph_id=graph.id,
                graph_version=graph.version,
                name=name,
                description=(kwargs.get("description") or "").strip(),
                trigger_config=kwargs.get("trigger_config") or {},
                agent_credentials=agent_credentials,
            )
        except (NotFoundError, InvalidInputError) as e:
            return ErrorResponse(
                message=f"Could not set up the trigger: {e}",
                error="trigger_setup_failed",
                session_id=session_id,
            )

        return self._build_success_response(preset, graph, library_agent.id, session_id)

    async def _resolve_graph(
        self,
        user_id: str,
        session: ChatSession,
        kwargs: dict[str, Any],
    ) -> tuple[GraphModel | None, ErrorResponse | None]:
        """Resolve the target graph from library_agent_id, graph_id, or builder."""
        session_id = session.session_id if session else None
        library_agent_id: str = (kwargs.get("library_agent_id") or "").strip()
        graph_id: str = (kwargs.get("graph_id") or "").strip()
        graph_version: int | None = kwargs.get("graph_version")

        if not library_agent_id and not graph_id:
            builder_graph_id = session.metadata.builder_graph_id
            if builder_graph_id:
                graph_id = builder_graph_id
            else:
                return None, ErrorResponse(
                    message="Provide a library_agent_id or graph_id.",
                    error="missing_argument",
                    session_id=session_id,
                )

        if library_agent_id:
            library_agent = await library_db().get_library_agent(
                library_agent_id, user_id
            )
            if not library_agent:
                return None, ErrorResponse(
                    message=f"Library agent '{library_agent_id}' not found.",
                    error="library_agent_not_found",
                    session_id=session_id,
                )
            graph_id = library_agent.graph_id
            graph_version = library_agent.graph_version

        graph = await graph_db().get_graph(graph_id, graph_version, user_id=user_id)
        if not graph:
            return None, ErrorResponse(
                message=f"Agent graph '{graph_id}' not found.",
                error="graph_not_found",
                session_id=session_id,
            )
        return graph, None

    def _resolve_credentials(
        self,
        graph: GraphModel,
        selection: dict[str, str],
        available_creds: list[Credentials],
    ) -> tuple[dict[str, CredentialsMetaInput], list[RequiredTriggerCredential]]:
        """Resolve required credential fields from the user's explicit selection.

        Returns (resolved credentials, fields still requiring an explicit
        choice). Credentials are never auto-matched: a field is only resolved
        when the caller passed a valid credential ID for it.
        """
        resolved: dict[str, CredentialsMetaInput] = {}
        required: list[RequiredTriggerCredential] = []

        for field_name, (field_info, _, _) in graph.regular_credentials_inputs.items():
            matching = [
                cred
                for cred in available_creds
                if cred.provider in field_info.provider
                and cred.type in field_info.supported_types
            ]
            chosen_id = selection.get(field_name)
            chosen = next((c for c in matching if c.id == chosen_id), None)
            if chosen is not None:
                resolved[field_name] = create_credential_meta_from_match(chosen)
                continue

            required.append(
                RequiredTriggerCredential(
                    field_name=field_name,
                    provider=next(iter(field_info.provider), "unknown"),
                    supported_types=sorted(field_info.supported_types),
                    options=[
                        CredentialChoice(
                            id=cred.id,
                            title=cred.title,
                            provider=cred.provider,
                            type=cred.type,
                        )
                        for cred in matching
                    ],
                )
            )

        return resolved, required

    def _build_success_response(
        self,
        preset: LibraryAgentPreset,
        graph: GraphModel,
        library_agent_id: str,
        session_id: str | None,
    ) -> TriggerSetupResponse:
        """Build the success response, surfacing the ingress URL when manual."""
        trigger_node = graph.webhook_input_node
        assert trigger_node is not None
        manual = trigger_node.block.block_type == BlockType.WEBHOOK_MANUAL
        webhook_url = preset.webhook.url if preset.webhook else None
        provider = preset.webhook.provider if preset.webhook else None

        if manual and webhook_url:
            message = (
                f"Trigger '{preset.name}' is set up. Give the user this exact "
                f"webhook URL to configure in their external service — do not "
                f"modify or reconstruct it: {webhook_url}"
            )
        else:
            message = (
                f"Trigger '{preset.name}' is set up and registered automatically; "
                "no webhook URL handoff is needed."
            )

        return TriggerSetupResponse(
            message=message,
            session_id=session_id,
            preset_id=preset.id,
            library_agent_id=library_agent_id,
            library_agent_link=f"/library/agents/{library_agent_id}",
            name=preset.name,
            is_active=preset.is_active,
            provider=provider or "",
            manual_setup_required=manual,
            webhook_url=webhook_url,
        )
