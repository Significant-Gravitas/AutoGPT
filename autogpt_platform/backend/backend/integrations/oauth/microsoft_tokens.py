"""Token plumbing shared by both ways of connecting Microsoft.

Every request goes out on behalf of the app that issued the grant: the
deployment's own Entra app proves itself with its secret, AutoGPT's public
client never sends one.
"""

import time

import httpx
from pydantic import BaseModel, SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft_365_copilot import (
    MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID,
)
from backend.integrations.oauth.microsoft_identity import (
    microsoft_identity,
    populate_microsoft_identity,
)
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery_from_response,
)
from backend.integrations.oauth.microsoft_services import GRAPH_PREFIX, token_scopes
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

LOGIN_BASE = "https://login.microsoftonline.com"
TIMEOUT_SECONDS = 20.0


class MicrosoftApp(BaseModel):
    client_id: str
    client_secret: SecretStr | None = None


def microsoft_app() -> MicrosoftApp:
    """The deployment's Entra app, or AutoGPT's public client when it has none."""
    secrets = Settings().secrets
    if not secrets.microsoft_client_id:
        return MicrosoftApp(client_id=MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID)
    secret = secrets.microsoft_client_secret
    return MicrosoftApp(
        client_id=secrets.microsoft_client_id,
        client_secret=SecretStr(secret) if secret else None,
    )


def browser_sign_in_available() -> bool:
    """Browser sign-in needs the app's secret and its redirect, which only the
    deployment can vouch for (MICROSOFT_BROWSER_SIGN_IN). Otherwise sign-in
    codes remain: the Teams bot's app has a secret long before the redirect."""
    if not Settings().config.microsoft_browser_sign_in:
        return False
    return microsoft_app().client_secret is not None


async def refresh_microsoft_credentials(
    credentials: OAuth2Credentials,
) -> OAuth2Credentials:
    """Renew a Microsoft connection with the app that granted it."""
    if not credentials.refresh_token:
        raise ValueError("Reconnect Microsoft to renew access.")
    client_id = issuing_client_id(credentials)
    payload = await token_request(
        authority(credentials),
        {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
            "scope": " ".join(token_scopes(credentials.scopes)),
            **client_fields(client_id),
        },
    )
    return await credentials_from_tokens(
        payload,
        client_id=client_id,
        auth_flow=credentials.metadata.get("microsoft_auth_flow", "device_code"),
        current=credentials,
    )


async def microsoft_resource_token(credentials: OAuth2Credentials, scope: str) -> str:
    """An access token for another Microsoft resource, from the same grant.

    Microsoft refresh tokens cover every permission the user consented to, so
    Work IQ needs no second sign-in once it has been approved.
    """
    identity = microsoft_identity(credentials)
    if identity is None or identity.account_type != "work":
        raise ValueError("This Microsoft service needs a work or school account.")
    if not credentials.refresh_token:
        raise ValueError("Reconnect Microsoft to renew access.")
    client_id = issuing_client_id(credentials)
    payload = await token_request(
        identity.tenant_id,
        {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
            "scope": scope,
            **client_fields(client_id),
        },
    )
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("Microsoft did not return an access token.")
    return token


def issuing_client_id(credentials: OAuth2Credentials) -> str:
    """Refresh tokens are bound to the app that issued them."""
    return credentials.metadata.get("microsoft_client_id") or microsoft_app().client_id


def client_fields(client_id: str) -> dict[str, str]:
    """Client authentication for a token request.

    Only the deployment's own app proves itself with its secret; a grant from
    the public client must never carry one (AADSTS700025), and the
    confidential app is refused without it (AADSTS7000218).
    """
    app = microsoft_app()
    fields = {"client_id": client_id}
    if client_id == app.client_id and app.client_secret:
        fields["client_secret"] = app.client_secret.get_secret_value()
    return fields


def authority(credentials: OAuth2Credentials) -> str:
    identity = microsoft_identity(credentials)
    if identity is None:
        return "common"
    return "consumers" if identity.account_type == "personal" else identity.tenant_id


def error_code(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return ""
    return str(data.get("error") or "") if isinstance(data, dict) else ""


async def token_request(authority: str, data: dict[str, str]) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{LOGIN_BASE}/{authority}/oauth2/v2.0/token", data=data
        )
    if not response.is_success:
        raise MicrosoftAuthError(microsoft_recovery_from_response(response))
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Microsoft returned an unreadable token response.")
    return payload


async def credentials_from_tokens(
    payload: dict,
    *,
    client_id: str,
    auth_flow: str,
    current: OAuth2Credentials | None = None,
) -> OAuth2Credentials:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Microsoft did not return an access token.")
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = (
            current.refresh_token.get_secret_value()
            if current and current.refresh_token
            else None
        )
    if not refresh_token:
        raise ValueError("Microsoft did not return a refresh token.")
    granted = payload.get("scope")
    scopes = (
        [scope.removeprefix(GRAPH_PREFIX) for scope in granted.split()]
        if isinstance(granted, str)
        else []
    ) or list(current.scopes if current else [])
    tokens = {
        "access_token": SecretStr(access_token),
        "refresh_token": SecretStr(refresh_token),
        "access_token_expires_at": _expires_at(payload.get("expires_in")),
        "scopes": scopes,
    }
    if current:
        credentials = current.model_copy(update=tokens)
    else:
        credentials = OAuth2Credentials(
            provider=ProviderName.MICROSOFT,
            title="Microsoft",
            refresh_token_expires_at=None,
            metadata={
                "microsoft_client_id": client_id,
                "microsoft_auth_flow": auth_flow,
            },
            **tokens,
        )
    id_token = payload.get("id_token")
    if isinstance(id_token, str) and id_token:
        await populate_microsoft_identity(credentials, id_token, client_id)
    elif current is None:
        raise ValueError("Microsoft did not return verified account information.")
    return credentials


def _expires_at(expires_in: object) -> int | None:
    try:
        return (
            int(time.time()) + int(expires_in)
            if isinstance(expires_in, (int, str))
            else None
        )
    except ValueError:
        return None
