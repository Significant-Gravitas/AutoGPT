"""Tool for setting up a webhook-triggered preset for a library agent.

Wraps the same logic as the ``POST /presets/setup-trigger`` route so AutoPilot
can set up a webhook trigger end-to-end and hand the user the correct ingress
URL for manual-setup webhooks.

Credential handling splits into two classes:

- The **trigger node's own credential** (the account a provider webhook is
  *registered under*) is surfaced for an explicit choice in the setup card and
  used verbatim — never silently auto-picked.
- All **other agent-body credentials** mirror ``run_agent``: auto-matched, and
  only shown in the card when missing (to connect).
"""

import logging
from typing import Any

from backend.api.features.library.model import LibraryAgentPreset
from backend.blocks._base import BlockType
from backend.copilot.model import ChatSession
from backend.data.db_accessors import graph_db, library_db, triggers_db
from backend.data.graph import GraphModel, Node
from backend.data.model import Credentials, CredentialsMetaInput
from backend.util.exceptions import (
    InvalidInputError,
    NotFoundError,
    WebhookRegistrationError,
)

from .base import BaseTool
from .models import (
    ErrorResponse,
    ResponseType,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .utils import (
    build_missing_credentials_from_graph,
    create_credential_meta_from_match,
    get_or_create_library_agent,
    get_user_credentials,
    match_user_credentials_to_graph,
)

logger = logging.getLogger(__name__)


def _is_filled(value: Any) -> bool:
    """Whether a trigger-config value counts as provided (non-empty)."""
    return value not in (None, "", {}, [])


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


class TriggerConfigRequiredResponse(ToolResponseBase):
    """Returned when the webhook trigger block needs configuration the LLM must
    collect from the user (e.g. a GitHub repo + which events to subscribe to).

    The config is gathered conversationally rather than via a card, because a
    trigger block's config (event filters etc.) can be arbitrarily structured.
    The LLM should ask the user for the fields in ``config_schema`` — never
    guess — then re-call with ``trigger_config`` filled in.
    """

    type: ResponseType = ResponseType.TRIGGER_CONFIG_REQUIRED
    missing_config: list[str]
    config_schema: dict[str, Any]
    graph_id: str
    graph_version: int


class SetupAgentWebhookTriggerTool(BaseTool):
    """Set up a webhook-triggered preset for an agent that has a trigger block.

    For manual-setup (generic) webhooks the response includes the ingress URL
    to give to the user. For provider webhooks (e.g. GitHub) the user picks
    which account the webhook is registered under via the inline setup card.
    """

    @property
    def name(self) -> str:
        return "setup_agent_webhook_trigger"

    @property
    def description(self) -> str:
        return (
            "Set up a webhook trigger for an agent with a webhook trigger block. "
            "This is the ONLY way to configure such a trigger: pass the trigger "
            "block's config as 'trigger_config' — never configure it by editing "
            "the agent's graph. If credentials are needed it returns a setup card; "
            "after the user proceeds, call again with the 'credentials' they selected. "
            "On success the result card shows any webhook URL with a copy button — "
            "don't reprint the URL; point the user to the card."
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
                        "Trigger block config inputs (from "
                        "trigger_setup_info.config_schema), e.g. repo + events; "
                        "omit credential fields. Set config HERE, not by editing "
                        "the trigger node. Usually empty for generic webhooks."
                    ),
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": (
                        "Credential selection {field_name: credential_id} the "
                        "user made in the setup card. Pass when re-calling after "
                        "the card; omit on the first call."
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

        if not (trigger_node := graph.webhook_input_node):
            return ErrorResponse(
                message=(
                    f"Agent '{graph.name}' has no webhook trigger block, so it "
                    "can't have a webhook trigger. Run or schedule it instead."
                ),
                error="no_webhook_trigger",
                session_id=session_id,
            )

        config_required = self._missing_trigger_config(
            graph, kwargs.get("trigger_config") or {}, session_id
        )
        if config_required:
            return config_required

        agent_credentials, card = await self._resolve_credentials(
            user_id, graph, trigger_node, kwargs.get("credentials") or {}, session_id
        )
        if card:
            return card

        library_agent = await get_or_create_library_agent(graph, user_id)
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
        except (
            InvalidInputError,
            NotFoundError,
            WebhookRegistrationError,
        ) as e:
            return ErrorResponse(
                message=f"Could not set up the trigger: {e}",
                error="trigger_setup_failed",
                session_id=session_id,
            )

        return self._build_success_response(
            preset, trigger_node, library_agent.id, session_id
        )

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
            try:
                library_agent = await library_db().get_library_agent(
                    library_agent_id, user_id
                )
            except NotFoundError:
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

    def _missing_trigger_config(
        self,
        graph: GraphModel,
        trigger_config: dict[str, Any],
        session_id: str | None,
    ) -> TriggerConfigRequiredResponse | None:
        """Return a config-required response if the trigger block has required
        config inputs (e.g. a repo + event filter) the caller hasn't supplied.

        These are collected from the user conversationally — the LLM must ask
        for them and never guess — rather than via the credentials card.
        """
        info = graph.trigger_setup_info
        schema = info.config_schema if info else None
        if not isinstance(schema, dict):
            return None
        required = schema.get("required", [])
        missing = [
            name for name in required if not _is_filled(trigger_config.get(name))
        ]
        if not missing:
            return None
        return TriggerConfigRequiredResponse(
            message=(
                "Before this trigger can be set up, ask the user for its "
                "configuration and pass it as `trigger_config` — do NOT guess "
                "values (e.g. don't invent a repository name). The required "
                "fields and their schema are below; once you have the user's "
                "answers, call setup_agent_webhook_trigger again with "
                "`trigger_config` filled in."
            ),
            session_id=session_id,
            missing_config=missing,
            config_schema=schema,
            graph_id=graph.id,
            graph_version=graph.version,
        )

    async def _resolve_credentials(
        self,
        user_id: str,
        graph: GraphModel,
        trigger_node: Node,
        selection: dict[str, str],
        session_id: str | None,
    ) -> tuple[dict[str, CredentialsMetaInput], SetupRequirementsResponse | None]:
        """Resolve the agent's credentials, or return a setup card to fill them.

        Body credentials are auto-matched (run_agent behaviour). The trigger
        node's own credential is only resolved from an explicit ``selection``,
        so it's always surfaced in the card for the user to pick the account the
        webhook is registered under — never silently auto-picked.

        Returns ``(agent_credentials, None)`` when ready to proceed, or
        ``({}, SetupRequirementsResponse)`` when the user must act first.
        """
        matched, _ = await match_user_credentials_to_graph(user_id, graph)
        trigger_cred_key = self._trigger_cred_key(graph, trigger_node)

        effective = dict(matched)
        if selection:
            available = await get_user_credentials(user_id)
            effective.update(self._resolve_selection(graph, selection, available))

        # Force the trigger credential into the card until it's explicitly
        # chosen, even when a candidate auto-matched.
        matched_for_card = {
            key: cred
            for key, cred in effective.items()
            if not (key == trigger_cred_key and trigger_cred_key not in selection)
        }
        card_missing = build_missing_credentials_from_graph(graph, matched_for_card)
        if card_missing:
            return {}, self._build_card(graph, card_missing, session_id)

        return effective, None

    @staticmethod
    def _trigger_cred_key(graph: GraphModel, trigger_node: Node) -> str | None:
        """Graph-level credential key that maps to the trigger node's own
        credential field (the webhook-registration account). None for manual
        webhooks, which have no credential."""
        return next(
            (
                key
                for key, (_, node_fields, _) in graph.regular_credentials_inputs.items()
                if any(node_id == trigger_node.id for node_id, _ in node_fields)
            ),
            None,
        )

    def _resolve_selection(
        self,
        graph: GraphModel,
        selection: dict[str, str],
        available_creds: list[Credentials],
    ) -> dict[str, CredentialsMetaInput]:
        """Resolve {field: credential_id} the user picked into credential metas,
        validating each against the field's provider/type. Invalid or unknown
        selections are dropped (so the card re-surfaces them)."""
        regular = graph.regular_credentials_inputs
        resolved: dict[str, CredentialsMetaInput] = {}
        for key, cred_id in selection.items():
            if not (entry := regular.get(key)):
                continue
            field_info = entry[0]
            cred = next(
                (
                    c
                    for c in available_creds
                    if c.id == cred_id
                    and c.provider in field_info.provider
                    and c.type in field_info.supported_types
                ),
                None,
            )
            if cred:
                resolved[key] = create_credential_meta_from_match(cred)
        return resolved

    def _build_card(
        self,
        graph: GraphModel,
        missing_credentials: dict[str, Any],
        session_id: str | None,
    ) -> SetupRequirementsResponse:
        """Build the inline credentials setup card (same shape as run_agent)."""
        return SetupRequirementsResponse(
            message=(
                "Choose or connect the account(s) in the card below, then "
                "proceed — connecting happens right here in this card, with no "
                "separate connection step needed first."
            ),
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=graph.id,
                agent_name=graph.name,
                user_readiness=UserReadiness(
                    has_all_credentials=False,
                    missing_credentials=missing_credentials,
                    ready_to_run=False,
                ),
                requirements={
                    "credentials": list(missing_credentials.values()),
                    "inputs": [],
                    "execution_modes": ["webhook"],
                },
            ),
            graph_id=graph.id,
            graph_version=graph.version,
        )

    def _build_success_response(
        self,
        preset: LibraryAgentPreset,
        trigger_node: Node,
        library_agent_id: str,
        session_id: str | None,
    ) -> TriggerSetupResponse:
        """Build the success response, surfacing the ingress URL when manual."""
        manual = trigger_node.block.block_type == BlockType.WEBHOOK_MANUAL
        webhook_url = preset.webhook.url if preset.webhook else None
        provider = preset.webhook.provider if preset.webhook else None

        if manual and webhook_url:
            # User-facing copy only — the card renders the URL itself in a copy
            # box; the tool description tells the model not to reprint it.
            message = (
                f"Trigger '{preset.name}' is set up. Copy the webhook URL below "
                "and paste it into your external service (e.g. TypeForm) to finish."
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
