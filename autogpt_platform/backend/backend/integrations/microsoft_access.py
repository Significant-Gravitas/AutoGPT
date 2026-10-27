"""What a Microsoft connection can do right now, service by service.

Each service reports one status, so a missing extra never hides the ones that
work. ``scopes`` names the permissions to ask for when the person can grant
them themselves.
"""

import asyncio
import re
from typing import Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from backend.blocks.mcp.client import MCPClient
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft_identity import microsoft_identity
from backend.integrations.oauth.microsoft_recovery import MicrosoftAuthError
from backend.integrations.oauth.microsoft_services import (
    DEFAULT_MICROSOFT_SCOPES,
    GRAPH_PREFIX,
    MICROSOFT_SERVICE_SCOPES,
    WORK_IQ_SCOPE,
    WORK_ONLY_SERVICES,
    MicrosoftService,
)
from backend.integrations.oauth.microsoft_tokens import (
    LOGIN_BASE,
    browser_sign_in_available,
    microsoft_app,
    microsoft_resource_token,
)
from backend.util.settings import Settings

GRAPH = "https://graph.microsoft.com/v1.0"
WORK_IQ_URL = "https://workiq.svc.cloud.microsoft/mcp"

MicrosoftCapabilityStatus = Literal[
    "available",
    "permission_needed",
    "admin_required",
    "setup_required",
    "work_account_only",
    "unavailable",
]


class MicrosoftCapability(BaseModel):
    service: MicrosoftService
    name: str
    status: MicrosoftCapabilityStatus
    scopes: list[str] = []


class MicrosoftAccessCheck(BaseModel):
    verified: bool
    account_type: Literal["personal", "work"] | None
    capabilities: list[MicrosoftCapability]
    message: str
    # Approves every work service for the whole organization at once. Only
    # an administrator can complete it.
    admin_consent_url: str | None = None


SERVICE_NAMES: dict[MicrosoftService, str] = {
    "mail": "Mail",
    "calendar": "Calendar",
    "files": "Files",
    "teams": "Teams chats",
    "copilot": "Microsoft 365 Copilot",
    "work_iq": "Work IQ",
}
GRAPH_PROBES: dict[MicrosoftService, str] = {
    "mail": "/me/mailFolders/inbox?$select=id",
    "calendar": "/me/calendar?$select=id",
    "files": "/me/drive?$select=driveType",
    "teams": "/me/chats?$top=1",
}


async def check_microsoft_access(credential: OAuth2Credentials) -> MicrosoftAccessCheck:
    identity = microsoft_identity(credential)
    account_type = identity.account_type if identity else None
    async with httpx.AsyncClient(
        headers={"Authorization": credential.auth_header()}, timeout=20
    ) as client:
        capabilities = await asyncio.gather(
            *(
                _check(client, credential, service, account_type)
                for service in MICROSOFT_SERVICE_SCOPES
            )
        )
    verified = any(c.status == "available" for c in capabilities)
    return MicrosoftAccessCheck(
        verified=verified,
        account_type=account_type,
        capabilities=list(capabilities),
        admin_consent_url=(
            _admin_consent_url(identity.tenant_id, capabilities)
            if identity and account_type == "work"
            else None
        ),
        message=(
            "Connected. Each service shows whether it is ready."
            if verified
            else "Microsoft is connected, but no service is ready yet."
        ),
    )


async def _check(
    client: httpx.AsyncClient,
    credential: OAuth2Credentials,
    service: MicrosoftService,
    account_type: str | None,
) -> MicrosoftCapability:
    if service in WORK_ONLY_SERVICES and account_type != "work":
        return _capability(service, "work_account_only")
    if service == "work_iq":
        return await _check_work_iq(credential)
    missing = [
        s for s in MICROSOFT_SERVICE_SCOPES[service] if s not in credential.scopes
    ]
    if missing:
        return _capability(service, "permission_needed", missing)
    probe = GRAPH_PROBES.get(service)
    if probe is None:
        # Copilot has no free read to probe; its license is checked on first use.
        return _capability(service, "available")
    try:
        response = await client.get(GRAPH + probe)
    except httpx.HTTPError:
        return _capability(service, "unavailable")
    return _capability(service, "available" if response.is_success else "unavailable")


async def _check_work_iq(credential: OAuth2Credentials) -> MicrosoftCapability:
    try:
        token = await microsoft_resource_token(credential, WORK_IQ_SCOPE)
    except MicrosoftAuthError as error:
        return _capability("work_iq", _work_iq_status(error), [WORK_IQ_SCOPE])
    except (ValueError, httpx.HTTPError):
        return _capability("work_iq", "unavailable")
    client = MCPClient(WORK_IQ_URL, authorization=f"Bearer {token}")
    try:
        tools = await client.list_tools()
    except Exception:
        # Signed in and approved, so what remains is the organization's
        # Work IQ enablement or its usage billing.
        return _capability("work_iq", "setup_required")
    finally:
        await client.close()
    return _capability("work_iq", "available" if tools else "setup_required")


def _admin_consent_url(
    tenant_id: str, capabilities: list[MicrosoftCapability]
) -> str | None:
    frontend = Settings().config.frontend_base_url
    if not frontend or not browser_sign_in_available():
        return None
    pending = [
        c for c in capabilities if c.status in ("permission_needed", "admin_required")
    ]
    if not pending:
        return None
    # Asking for Work IQ before Microsoft has set it up for the organization
    # fails the whole approval (AADSTS650052), and a missing setup can look
    # like a missing permission until someone consents. So Work IQ joins only
    # when Microsoft says an admin must approve it, which it can only say once
    # Work IQ is set up.
    services: list[MicrosoftService] = [
        service
        for service in MICROSOFT_SERVICE_SCOPES
        if service != "work_iq"
        or any(c.service == "work_iq" and c.status == "admin_required" for c in pending)
    ]
    scopes = dict.fromkeys(
        scope if scope.startswith("api://") else GRAPH_PREFIX + scope
        for scope in [
            *DEFAULT_MICROSOFT_SCOPES,
            *(s for service in services for s in MICROSOFT_SERVICE_SCOPES[service]),
        ]
    )
    query = urlencode(
        {
            "client_id": microsoft_app().client_id,
            "scope": " ".join(scopes),
            "redirect_uri": f"{frontend}/auth/integrations/oauth_callback",
        }
    )
    return f"{LOGIN_BASE}/{tenant_id}/v2.0/adminconsent?{query}"


def _work_iq_status(error: MicrosoftAuthError) -> MicrosoftCapabilityStatus:
    reason = error.recovery.reason
    code = int(re.sub(r"\D", "", error.recovery.error_code or "") or 0)
    if reason == "tenant_setup_required":
        return "setup_required"
    if code == 65001:
        # Not consented yet: one more sign-in asks for it.
        return "permission_needed"
    if reason == "admin_consent_required":
        return "admin_required"
    return "unavailable"


def _capability(
    service: MicrosoftService,
    status: MicrosoftCapabilityStatus,
    scopes: list[str] | None = None,
) -> MicrosoftCapability:
    return MicrosoftCapability(
        service=service, name=SERVICE_NAMES[service], status=status, scopes=scopes or []
    )
