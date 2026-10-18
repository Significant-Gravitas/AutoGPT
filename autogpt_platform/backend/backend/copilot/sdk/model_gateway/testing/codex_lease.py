"""A Codex credential lease for tests, and a client for a fake Responses backend."""

import base64
import json
from unittest.mock import AsyncMock

import httpx
from pydantic import SecretStr
from redis.asyncio.lock import Lock

from backend.copilot.sdk.model_gateway.route import CodexLeaseCredential, GatewayRoute
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.transport import CodexCredentialLease
from backend.integrations.credential_lease import CredentialLease


def fake_access_token(account_id: str = "acct-test") -> str:
    """An unsigned JWT: the platform reads its claims, never its signature."""
    claims = {
        "exp": 9_999_999_999,
        "email": "a@b.c",
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    header, payload = (
        base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
        for part in ({"alg": "none"}, claims)
    )
    return f"{header}.{payload}.unsigned"


def fake_codex_lease(
    credential_id: str = "codex-credential", account_id: str = "acct-test"
) -> CredentialLease:
    token = SecretStr(fake_access_token(account_id))
    bundle = CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=token, access_token=token, refresh_token=SecretStr("refresh")
        ),
        codex_runtime_version="http",
    )
    credentials = credentials_from_bundle(bundle).model_copy(
        update={"id": credential_id}
    )
    lock = AsyncMock(spec=Lock)
    lock.locked.return_value = lock.owned.return_value = True
    return CredentialLease(credentials, lock, AsyncMock())


def codex_route(
    model: str,
    *,
    lease: CodexCredentialLease | CredentialLease | None = None,
    context_window: int | None = None,
) -> GatewayRoute:
    return GatewayRoute(
        adapter="codex_responses",
        model=model,
        credential=CodexLeaseCredential(lease=lease or fake_codex_lease()),
        context_window=context_window,
        effort="medium",
    )


def fake_backend_client(
    base_url: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """A client for a fake Responses endpoint; the caller closes it.

    *transport* answers this client's requests in place of the network,
    without touching any other client in the process.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        trust_env=False,
        follow_redirects=False,
        timeout=15,
        transport=transport,
    )
