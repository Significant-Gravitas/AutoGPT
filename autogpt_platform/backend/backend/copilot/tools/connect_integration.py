"""Tool for prompting the user to connect a required integration.

When the copilot encounters an authentication failure (e.g. `gh` CLI returns
"authentication required"), it calls this tool to surface the credentials
setup card in the chat — the same UI that appears when a GitHub block runs
without configured credentials.
"""

from typing import Any

from backend.blocks.github._auth import GITHUB_OAUTH_IS_CONFIGURED
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import (
    ResponseType,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)

from .base import BaseTool

# Registry of known providers: name + supported credential types.
# Extend this dict when adding support for new integrations.
_PROVIDER_INFO: dict[str, dict[str, Any]] = {
    "github": {
        "name": "GitHub",
        "types": (["api_key", "oauth2"] if GITHUB_OAUTH_IS_CONFIGURED else ["api_key"]),
        "scopes": [],
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
            "After the user connects the account, retry the operation — the token "
            "will be automatically available in the execution environment."
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
                    "enum": list(_PROVIDER_INFO.keys()),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Brief explanation of why the integration is needed, "
                        "shown to the user in the setup card."
                    ),
                },
            },
            "required": ["provider"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id  # not needed; setup card is user-agnostic
        session_id = session.session_id if session else None
        provider: str = (kwargs.get("provider") or "").strip().lower()
        reason: str = (kwargs.get("reason") or "").strip()

        info = _PROVIDER_INFO.get(provider)
        if not info:
            supported = ", ".join(f"'{p}'" for p in _PROVIDER_INFO)
            return SetupRequirementsResponse(
                type=ResponseType.SETUP_REQUIREMENTS,
                message=(
                    f"Unknown provider '{provider}'. "
                    f"Supported providers: {supported}."
                ),
                session_id=session_id,
                setup_info=SetupInfo(
                    agent_id=f"connect_{provider}",
                    agent_name=f"{provider.title()} Integration",
                ),
            )

        provider_name: str = info["name"]
        supported_types: list[str] = info["types"]
        scopes: list[str] = info["scopes"]
        field_key = f"{provider}_credentials"

        message_parts = [
            f"To continue, please connect your {provider_name} account.",
        ]
        if reason:
            message_parts.append(reason)

        missing_credentials: dict[str, Any] = {
            field_key: {
                "id": field_key,
                "title": f"{provider_name} Credentials",
                "provider": provider,
                "provider_name": provider_name,
                "type": supported_types[0],
                "types": supported_types,
                "scopes": scopes,
            }
        }

        return SetupRequirementsResponse(
            type=ResponseType.SETUP_REQUIREMENTS,
            message=" ".join(message_parts),
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=f"connect_{provider}",
                agent_name=f"{provider_name} Integration",
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
