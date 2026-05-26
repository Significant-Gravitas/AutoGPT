"""Tool for prompting the user to connect a required integration.

When the copilot encounters an authentication failure (e.g. `gh` CLI returns
"authentication required"), it calls this tool to surface the credentials
setup card in the chat — the same UI that appears when a GitHub block runs
without configured credentials.
"""

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

from .base import BaseTool


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
