"""Tool for prompting the user to connect a required integration.

When the copilot encounters an authentication failure (e.g. `gh` CLI returns
"authentication required"), it calls this tool to surface the credentials
setup card in the chat — the same UI that appears when a GitHub block runs
without configured credentials.
"""

from typing import Any, TypedDict

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

from .base import BaseTool


class _CredentialEntry(TypedDict):
    """Shape of each entry inside SetupRequirementsResponse.user_readiness.missing_credentials.

    Partially overlaps with :class:`~backend.data.model.CredentialsMetaInput`
    (``id``, ``title``, ``provider``) but carries extra UI-facing fields
    (``types``, ``scopes``) that the frontend ``SetupRequirementsCard`` needs
    to render the inline credential setup card.

    Display name is derived from :data:`SUPPORTED_PROVIDERS` at build time
    rather than stored here — eliminates the old ``provider_name`` field.
    ``types`` replaces the old singular ``type`` field; the frontend already
    prefers ``types`` and only fell back to ``type`` for compatibility.
    """

    id: str
    title: str
    # Slug used as the credential key (e.g. "github").
    provider: str
    # All supported credential types the user can choose from (e.g. ["api_key", "oauth2"]).
    # The first element is the default/primary type.
    types: list[str]
    scopes: list[str]


class ConnectIntegrationTool(BaseTool):
    """Surface the credentials setup UI when an integration is not connected."""

    @property
    def name(self) -> str:
        return "connect_integration"

    @property
    def description(self) -> str:
        return (
            "Prompt the user to connect a required integration (e.g. GitHub). "
            "Call this when an external CLI or API call fails because the user "
            "has not connected the relevant account. "
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
                    "enum": list(SUPPORTED_PROVIDERS.keys()),
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
        _ = user_id  # setup card is user-agnostic; auth is enforced via requires_auth
        session_id = session.session_id if session else None
        provider = (provider or "").strip().lower()
        reason = (reason or "").strip()[:500]  # cap LLM-controlled text
        extra_scopes: list[str] = [
            str(s).strip() for s in (scopes or []) if str(s).strip()
        ]

        entry = SUPPORTED_PROVIDERS.get(provider)
        if not entry:
            supported = ", ".join(f"'{p}'" for p in SUPPORTED_PROVIDERS)
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

        credential_entry: _CredentialEntry = {
            "id": field_key,
            "title": f"{display_name} Credentials",
            "provider": provider,
            "types": supported_types,
            "scopes": merged_scopes,
        }
        missing_credentials: dict[str, _CredentialEntry] = {field_key: credential_entry}

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
