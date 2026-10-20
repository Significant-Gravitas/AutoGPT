"""Tool for prompting the user to connect a required integration.

When the copilot encounters an authentication failure (e.g. `gh` CLI returns
"authentication required"), it calls this tool to surface the credentials
setup card in the chat — the same UI that appears when a GitHub block runs
without configured credentials.
"""

import json
from typing import Any, cast

from backend.copilot.model import ChatSession
from backend.copilot.providers import SUPPORTED_PROVIDERS, get_provider_auth_types
from backend.copilot.tools.models import (
    ErrorResponse,
    ResponseType,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from backend.copilot.tools.utils import build_missing_credentials_from_field_info
from backend.data.model import CredentialsFieldInfo, CredentialsType
from backend.integrations.providers import ProviderName

from .ayrshare_managed import (
    AYRSHARE,
    ensure_profile,
    link_accounts_card,
    link_message,
    with_expert_grants,
)
from .base import BaseTool
from .expert_scope import annotate_expert_grants

CONNECT_INTEGRATION_TOOL = "connect_integration"

# Providers whose account AutoGPT holds: connecting shows the user a button to
# link their own accounts to it, never a key or token field.  Kept out of
# SUPPORTED_PROVIDERS, which also decides what is injected into the sandbox.
MANAGED_PROVIDERS: dict[str, str] = {AYRSHARE: "Ayrshare"}


def _supported_providers() -> list[str]:
    return [*SUPPORTED_PROVIDERS, *MANAGED_PROVIDERS]


def _merged_scopes(provider: str, extra: list[str]) -> frozenset[str]:
    entry = SUPPORTED_PROVIDERS.get(provider)
    defaults = entry["default_scopes"] if entry else []
    return frozenset(s for s in (*defaults, *extra) if s)


def requested_scopes(session: ChatSession | None) -> dict[str, frozenset[str]]:
    """The scopes this session's latest connect card asked for, per provider.

    The card counts an account as connected only when it grants every one of
    these, so whatever gets injected into the sandbox has to be chosen by the
    same rule. Read from the transcript, which already survives every turn.
    """
    found: dict[str, frozenset[str]] = {}
    for message in reversed(session.messages if session else []):
        for call in reversed(message.tool_calls or []):
            function = call.get("function") or {}
            name = str(function.get("name") or call.get("name") or "")
            if name.rsplit("__", 1)[-1] != CONNECT_INTEGRATION_TOOL:
                continue
            # Both transcript shapes: nested under "function", or flat.
            raw = function.get("arguments") or call.get("arguments") or "{}"
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(args, dict):
                continue
            provider = str(args.get("provider") or "").strip().lower()
            if not provider or provider in found:
                continue
            scopes = args.get("scopes")
            extra = [str(x).strip() for x in scopes] if isinstance(scopes, list) else []
            found[provider] = _merged_scopes(provider, extra)
    return found


class ConnectIntegrationTool(BaseTool):
    """Surface the credentials setup UI when an integration is not connected."""

    @property
    def name(self) -> str:
        return CONNECT_INTEGRATION_TOOL

    @property
    def description(self) -> str:
        supported = ", ".join(f"'{p}'" for p in _supported_providers())
        return (
            f"Prompt the user to connect a required integration. "
            f"Supported providers: {supported}. "
            "'ayrshare' (post_to_* blocks) returns the card where the user "
            "links their social accounts, whenever they want to link them; "
            "never ask for its API key. "
            "ONLY call this tool for one of the supported providers listed above — "
            "do NOT call it for Google, Gmail, Slack, or any other provider not in the list. "
            "For the other providers, call this ONLY when an external CLI or API "
            "call in the sandbox fails because the user has not connected the "
            "relevant account. "
            "Do NOT call this for agent block credential issues — `run_agent` "
            "automatically detects and prompts for the correct provider based "
            "on the agent's graph metadata. Using this tool for agent blocks "
            "risks requesting the WRONG provider. "
            "The `provider` parameter must match what the failing CLI/API "
            "actually needs. Double-check that the provider is in the supported "
            "list above before calling. "
            "The tool surfaces a credentials setup card in the chat so the user "
            "can authenticate without leaving the page. "
            "After the user connects the account, retry the operation. "
            "In E2B/cloud sandbox mode the token (GH_TOKEN/GITHUB_TOKEN) is "
            "automatically injected per-command in bash_exec — no manual export needed. "
            "In local bubblewrap mode network is isolated so GitHub CLI commands "
            "will still fail after connecting; inform the user of this limitation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "Integration provider slug, e.g. 'github'. "
                        "Must be one of the supported providers."
                    ),
                    "enum": _supported_providers(),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Brief explanation of why the integration is needed, "
                        "shown to the user in the setup card."
                    ),
                    "maxLength": 500,
                },
                "scopes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "OAuth scopes to request. Omit to use the provider default. "
                        "Add extra scopes when you need more access — e.g. for GitHub: "
                        "'repo' (clone/push/pull), 'read:org' (org membership), "
                        "'workflow' (GitHub Actions). "
                        "Requesting only the scopes you actually need is best practice."
                    ),
                },
            },
            "required": ["provider"],
        }

    @property
    def requires_auth(self) -> bool:
        # Require auth so only authenticated users can trigger the setup card.
        # The card itself is user-agnostic (no per-user data needed), so
        # user_id is intentionally unused in _execute.
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        provider: str = "",
        reason: str = "",
        scopes: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResponseBase:
        """Build and return a :class:`SetupRequirementsResponse` for the requested provider.

        Validates the *provider* slug against the known registry, merges any
        agent-requested OAuth *scopes* with the provider defaults, and constructs
        the credential setup card payload that the frontend renders as an inline
        authentication prompt.

        Returns an :class:`ErrorResponse` if *provider* is unknown.
        """
        session_id = session.session_id if session else None
        provider = (provider or "").strip().lower()
        reason = (reason or "").strip()[:500]  # cap LLM-controlled text
        extra_scopes: list[str] = [
            str(s).strip() for s in (scopes or []) if str(s).strip()
        ]

        if provider in MANAGED_PROVIDERS and user_id:
            # Connecting creates the user's real profile; a dry run may not.
            if session.dry_run:
                return ErrorResponse(
                    message=(
                        f"This is a dry-run session, so connecting "
                        f"{MANAGED_PROVIDERS[provider]} was skipped. Nothing was "
                        "created."
                    ),
                    error="dry_run",
                    session_id=session_id,
                )
            return await _connect_managed(
                user_id, provider, reason, session_id, session.expert_id
            )
        entry = SUPPORTED_PROVIDERS.get(provider)
        if not entry:
            supported = ", ".join(f"'{p}'" for p in _supported_providers())
            return ErrorResponse(
                message=(
                    f"Unknown provider '{provider}'. Supported providers: {supported}."
                ),
                error="unknown_provider",
                session_id=session_id,
            )

        display_name: str = entry["name"]
        supported_types: list[str] = get_provider_auth_types(provider)
        # Merge agent-requested scopes with provider defaults (deduplicated, order preserved).
        default_scopes: list[str] = entry["default_scopes"]
        seen: set[str] = set()
        merged_scopes: list[str] = []
        for s in default_scopes + extra_scopes:
            if s not in seen:
                seen.add(s)
                merged_scopes.append(s)
        field_key = f"{provider}_credentials"

        message_parts = [
            f"To continue, please connect your {display_name} account.",
        ]
        if reason:
            message_parts.append(reason)
        if session.expert_id is not None:
            message_parts.append(
                "Note: a credential connected here belongs to the account and "
                "is granted to this expert automatically."
            )

        # Route the single-provider entry through the shared serializer
        # used by run_block / run_agent so the payload shape (sorted scopes,
        # type+types fields, optional discriminator) stays in lockstep across
        # all three credential-surfacing tools. The casts narrow the runtime
        # strings — already validated upstream — to the typed enum/literal
        # the generic constructor expects.
        provider_enum = ProviderName(provider)
        typed_types: frozenset[CredentialsType] = cast(
            frozenset[CredentialsType], frozenset(supported_types)
        )
        field_info = CredentialsFieldInfo[ProviderName, CredentialsType](
            credentials_provider=frozenset([provider_enum]),
            credentials_types=typed_types,
            credentials_scopes=(frozenset(merged_scopes) if merged_scopes else None),
        )
        missing_credentials: dict[str, Any] = build_missing_credentials_from_field_info(
            credential_fields={field_key: field_info},
            matched_keys=set(),
        )
        # Preserve the registry's display name (e.g. "GitHub Credentials")
        # rather than the title-cased slug ("Github Credentials") that the
        # generic serializer produces from `field_key`.
        missing_credentials[field_key]["title"] = f"{display_name} Credentials"
        missing_credentials[field_key]["provider_name"] = display_name
        if user_id:
            missing_credentials = await annotate_expert_grants(
                user_id, session.expert_id, missing_credentials
            )

        return SetupRequirementsResponse(
            type=ResponseType.SETUP_REQUIREMENTS,
            message=" ".join(message_parts),
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=f"connect_{provider}",
                agent_name=display_name,
                user_readiness=UserReadiness(
                    has_all_credentials=False,
                    missing_credentials=missing_credentials,
                    ready_to_run=False,
                ),
                requirements={
                    "credentials": [missing_credentials[field_key]],
                    "inputs": [],
                    "execution_modes": [],
                },
            ),
        )


async def _connect_managed(
    user_id: str,
    provider: str,
    reason: str,
    session_id: str | None,
    expert_id: str | None,
) -> ToolResponseBase:
    """The card for a provider AutoGPT holds the account for.

    Provisions the user's profile when it does not exist yet, through the
    same path as the builder's connect button, and hands back the card with
    that button.
    """
    error = await ensure_profile(user_id)
    if error is not None:
        error.session_id = session_id
        return error
    display_name = MANAGED_PROVIDERS[provider]
    field_key = f"{provider}_credentials"
    field_info = CredentialsFieldInfo[ProviderName, CredentialsType](
        credentials_provider=frozenset([ProviderName(provider)]),
        credentials_types=frozenset(["api_key"]),
        credentials_scopes=None,
    )
    missing = build_missing_credentials_from_field_info(
        credential_fields={field_key: field_info}, matched_keys=set()
    )
    missing[field_key]["title"] = f"{display_name} Credentials"
    missing[field_key]["provider_name"] = display_name
    missing, needs_grant = await with_expert_grants(user_id, expert_id, missing)
    lead = "To post to social networks, the user links their accounts once."
    return link_accounts_card(
        agent_id=f"connect_{provider}",
        agent_name=display_name,
        missing_credentials=missing,
        message=link_message(f"{lead} {reason}".strip(), needs_grant=needs_grant),
        session_id=session_id,
        execution_modes=(),
    )
