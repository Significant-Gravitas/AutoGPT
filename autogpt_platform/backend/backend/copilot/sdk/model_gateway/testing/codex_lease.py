"""A Codex credential lease for tests, and an adapter pointed at a fake backend."""

from openai import AsyncOpenAI
from pydantic import SecretStr

from backend.copilot.sdk.model_gateway.route import CodexLeaseCredential, GatewayRoute
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.http_session import CodexHttpSession
from backend.integrations.codex.transport import CodexAgentSession, CodexCredentialLease

# An unsigned JWT: the platform reads its claims, never its signature.
FAKE_ACCESS_TOKEN = (
    "eyJhbGciOiJub25lIn0.eyJleHAiOjk5OTk5OTk5OTksImVtYWlsIjoiYUBiLmMifQ.sig"
)


def fake_codex_lease(credential_id: str = "codex-credential") -> CodexCredentialLease:
    bundle = CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=SecretStr(FAKE_ACCESS_TOKEN),
            access_token=SecretStr(FAKE_ACCESS_TOKEN),
            refresh_token=SecretStr("refresh"),
        ),
        codex_runtime_version="http",
    )
    credentials = credentials_from_bundle(bundle).model_copy(
        update={"id": credential_id}
    )
    session = CodexHttpSession(
        credentials, turn_timeout_seconds=60, tool_timeout_seconds=60
    )
    return CodexCredentialLease(credentials, CodexAgentSession(session, credentials))


def codex_route(
    model: str,
    *,
    lease: CodexCredentialLease | None = None,
    context_window: int | None = None,
) -> GatewayRoute:
    return GatewayRoute(
        adapter="codex_responses",
        model=model,
        credential=CodexLeaseCredential(lease=lease or fake_codex_lease()),
        context_window=context_window,
        effort="medium",
    )


def fake_backend_client(base_url: str) -> AsyncOpenAI:
    """A client for a fake Responses endpoint; the caller closes it."""
    return AsyncOpenAI(api_key="codex-test", base_url=base_url, max_retries=0)
