"""Tool for prompting the user to connect a required integration.

When the copilot encounters an authentication failure (e.g. `gh` CLI returns
"authentication required"), it calls this tool to surface the credentials
setup card in the chat — the same UI that appears when a GitHub block runs
without configured credentials.
"""

import functools
from typing import Any, TypedDict

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import (
    ErrorResponse,
    ResponseType,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)

from .base import BaseTool


class _ProviderInfo(TypedDict):
    name: str
    types: list[str]
    # Default OAuth scopes requested when the agent doesn't specify any.
    scopes: list[str]


class _CredentialEntry(TypedDict):
    """Shape of each entry inside SetupRequirementsResponse.user_readiness.missing_credentials."""

    id: str
    title: str
    provider: str
    provider_name: str
    type: str
    types: list[str]
    scopes: list[str]


@functools.lru_cache(maxsize=1)
def _is_github_oauth_configured() -> bool:
    """Return True if GitHub OAuth env vars are set.

    Evaluated lazily (not at import time) to avoid triggering Secrets() during
    module import, which can fail in environments where secrets are not loaded.
    """
    from backend.blocks.github._auth import GITHUB_OAUTH_IS_CONFIGURED

    return GITHUB_OAUTH_IS_CONFIGURED


# Registry of known providers: name + supported credential types for the UI.
# When adding a new provider, also add its env var names to
# backend.copilot.integration_creds.PROVIDER_ENV_VARS.
def _get_provider_info() -> dict[str, _ProviderInfo]:
    """Build the provider registry, evaluating OAuth config lazily."""
    return {
        "github": {
            "name": "GitHub",
            "types": (
                ["api_key", "oauth2"] if _is_github_oauth_configured() else ["api_key"]
            ),
            # Default: repo scope covers clone/push/pull for public and private repos.
            # Agent can request additional scopes (e.g. "read:org") via the scopes param.
            "scopes": ["repo"],
        },
    }


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
                    "enum": list(_get_provider_info().keys()),
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
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id  # setup card is user-agnostic; auth is enforced via requires_auth
        session_id = session.session_id if session else None
        provider: str = (kwargs.get("provider") or "").strip().lower()
        reason: str = (kwargs.get("reason") or "").strip()[
            :500
        ]  # cap LLM-controlled text
        extra_scopes: list[str] = [
            str(s).strip() for s in (kwargs.get("scopes") or []) if str(s).strip()
        ]

        provider_info = _get_provider_info()
        info = provider_info.get(provider)
        if not info:
            supported = ", ".join(f"'{p}'" for p in provider_info)
            return ErrorResponse(
                message=(
                    f"Unknown provider '{provider}'. "
                    f"Supported providers: {supported}."
                ),
                error="unknown_provider",
                session_id=session_id,
            )

        provider_name: str = info["name"]
        supported_types: list[str] = info["types"]
        # Merge agent-requested scopes with provider defaults (deduplicated, order preserved).
        default_scopes: list[str] = info["scopes"]
        seen: set[str] = set()
        scopes: list[str] = []
        for s in default_scopes + extra_scopes:
            if s not in seen:
                seen.add(s)
                scopes.append(s)
        field_key = f"{provider}_credentials"

        message_parts = [
            f"To continue, please connect your {provider_name} account.",
        ]
        if reason:
            message_parts.append(reason)

        credential_entry: _CredentialEntry = {
            "id": field_key,
            "title": f"{provider_name} Credentials",
            "provider": provider,
            "provider_name": provider_name,
            "type": supported_types[0],
            "types": supported_types,
            "scopes": scopes,
        }
        missing_credentials: dict[str, _CredentialEntry] = {field_key: credential_entry}

        return SetupRequirementsResponse(
            type=ResponseType.SETUP_REQUIREMENTS,
            message=" ".join(message_parts),
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=f"connect_{provider}",
                agent_name=provider_name,
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
