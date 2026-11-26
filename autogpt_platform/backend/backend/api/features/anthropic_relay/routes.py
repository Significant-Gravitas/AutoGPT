"""Where a Claude Code block's sandbox sends its Messages API calls: the
sandbox holds a per-run token, and the user's own key is added here."""

import logging
from contextlib import AsyncExitStack

import aiohttp
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from backend.api.middleware.body_limit import request_body_limit
from backend.data.model import APIKeyCredentials
from backend.integrations.claude_code_relay import relay_grant
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName
from backend.util.anthropic_forwarding import (
    ANTHROPIC_API_URL,
    MAX_REQUEST_BYTES,
    error_body,
    forward,
    headers_back,
    sent_credentials,
    upstream_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anthropic-relay")
creds_manager = IntegrationCredentialsManager()


# No user auth: the caller is the CLI in a block's sandbox, and its run's token
# is the credential (``integrations/claude_code_relay.py``).
@router.post("/v1/messages", include_in_schema=False)
@request_body_limit(MAX_REQUEST_BYTES)
async def relay_messages(request: Request) -> Response:
    return await _relay(request, "/v1/messages")


@router.post("/v1/messages/count_tokens", include_in_schema=False)
@request_body_limit(MAX_REQUEST_BYTES)
async def relay_count_tokens(request: Request) -> Response:
    return await _relay(request, "/v1/messages/count_tokens")


async def _relay(request: Request, path: str) -> Response:
    api_key = await _granted_key(request)
    if api_key is None:
        return JSONResponse(
            error_body("authentication_error", "invalid x-api-key"), status_code=401
        )
    body = await request.body()
    resources = AsyncExitStack()
    try:
        session = await resources.enter_async_context(upstream_session())
        upstream = await resources.enter_async_context(
            forward(
                session,
                upstream_url=ANTHROPIC_API_URL,
                path=path,
                query=request.url.query,
                headers=request.headers.items(),
                body=body,
                api_key=api_key,
            )
        )
    except (aiohttp.ClientError, TimeoutError) as error:
        await resources.aclose()
        logger.warning(
            f"Anthropic relay could not reach the Anthropic API: {type(error).__name__}"
        )
        return JSONResponse(
            error_body("api_error", "Could not reach the Anthropic API"),
            status_code=502,
        )
    except BaseException:
        await resources.aclose()
        raise

    async def reply():
        try:
            async for chunk in upstream.content.iter_any():
                yield chunk
        finally:
            await resources.aclose()

    return StreamingResponse(
        reply(), status_code=upstream.status, headers=dict(headers_back(upstream))
    )


async def _granted_key(request: Request) -> str | None:
    """The key of the credential the request's token was minted for, while the
    token is live and the credential is still the user's own Anthropic key."""
    for sent in sent_credentials(request.headers):
        grant = await relay_grant(sent)
        if grant is None:
            continue
        credentials = await creds_manager.get(grant.user_id, grant.credentials_id)
        if (
            isinstance(credentials, APIKeyCredentials)
            and credentials.provider == ProviderName.ANTHROPIC
            and not is_system_credential(credentials.id)
            and not credentials.is_managed
        ):
            return credentials.api_key.get_secret_value()
        return None
    return None
