import logging
import secrets
from typing import TYPE_CHECKING, Annotated, Literal
from urllib.parse import quote, urlencode

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Path, Security, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.api.features.integrations.codex_device_page import render_device_login_page
from backend.integrations.codex.access import enforce_codex_access_http
from backend.integrations.codex.login import (
    CodexDeviceLogin,
    CodexDeviceLoginState,
    CodexLoginCoordinator,
)
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexRateLimitsSnapshot,
)
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.codex.transport import CodexTransport

CODEX_LOGIN_STATE_KEY = "codex_login_id"

logger = logging.getLogger(__name__)
router = APIRouter()
codex_login_coordinator = CodexLoginCoordinator(
    timeout_seconds=Settings().config.codex_login_timeout_seconds
)
codex_credentials_manager = IntegrationCredentialsManager()


class CodexLoginStatusResponse(BaseModel):
    status: Literal["pending", "completed", "failed", "canceled"]
    error: str | None = None


@router.get(
    "/credentials/{credential_id}/account",
    response_model=CodexAccountSnapshot,
)
async def codex_account(
    credential_id: Annotated[str, Path(min_length=1)],
    user_id: Annotated[str, Security(get_user_id)],
) -> CodexAccountSnapshot:
    await enforce_codex_access_http(user_id)
    lease = await _acquire_codex_lease(user_id, credential_id)
    try:
        return await _get_codex_transport().account(lease)
    except Exception as error:
        logger.warning("Codex account check failed with %s", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not read the connected ChatGPT account",
        ) from None
    finally:
        await lease.release()


@router.get(
    "/credentials/{credential_id}/rate-limits",
    response_model=CodexRateLimitsSnapshot,
)
async def codex_rate_limits(
    credential_id: Annotated[str, Path(min_length=1)],
    user_id: Annotated[str, Security(get_user_id)],
) -> CodexRateLimitsSnapshot:
    await enforce_codex_access_http(user_id)
    lease = await _acquire_codex_lease(user_id, credential_id)
    try:
        return await _get_codex_transport().rate_limits(lease)
    except Exception as error:
        logger.warning("Codex rate-limit check failed with %s", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not read ChatGPT usage limits",
        ) from None
    finally:
        await lease.release()


@router.get(
    "/device-login/{login_id}",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def device_login_page(
    login_id: Annotated[str, Path(min_length=1)],
    user_id: Annotated[str, Security(get_user_id)],
) -> HTMLResponse:
    attempt = await codex_login_coordinator.get(user_id, login_id)
    if attempt is None:
        raise _login_not_found()
    nonce = secrets.token_urlsafe(18)
    return HTMLResponse(render_device_login_page(nonce), headers=_page_headers(nonce))


@router.get(
    "/device-login/{login_id}/status",
    response_model=CodexLoginStatusResponse,
    include_in_schema=False,
)
async def device_login_status(
    login_id: Annotated[str, Path(min_length=1)],
    user_id: Annotated[str, Security(get_user_id)],
) -> CodexLoginStatusResponse:
    attempt = await codex_login_coordinator.get(user_id, login_id)
    if attempt is None:
        raise _login_not_found()
    return _status_response(attempt)


@router.post(
    "/device-login/{login_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    include_in_schema=False,
)
async def cancel_device_login(
    login_id: Annotated[str, Path(min_length=1)],
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    if not await codex_login_coordinator.cancel(user_id, login_id):
        raise _login_not_found()


def build_device_login_url(
    frontend_base_url: str,
    login: CodexDeviceLogin,
    state_token: str,
) -> str:
    base_url = frontend_base_url.rstrip("/")
    login_id = quote(login.login_id, safe="")
    fragment = urlencode(
        {
            "state": state_token,
            "verification_url": login.verification_url,
            "user_code": login.user_code,
        }
    )
    return (
        f"{base_url}/api/proxy/api/integrations/codex/"
        f"device-login/{login_id}#{fragment}"
    )


def build_device_login_cancel_url(
    login: CodexDeviceLogin,
) -> str:
    login_id = quote(login.login_id, safe="")
    return f"/api/proxy/api/integrations/codex/device-login/{login_id}/cancel"


async def revoke_codex_credentials(
    manager: IntegrationCredentialsManager,
    user_id: str,
    credential_id: str,
) -> bool:
    try:
        lease = await manager.acquire_lease(user_id, credential_id)
    except ValueError:
        return False
    try:
        if lease.credentials.provider != "codex":
            return False
        revoked = False
        if _is_codex_lease(lease):
            try:
                await _get_codex_transport().logout(lease)
                revoked = True
            except Exception as error:
                logger.warning("Codex logout failed with %s", type(error).__name__)
        await lease.delete()
        return revoked
    finally:
        await lease.release()


async def _acquire_codex_lease(
    user_id: str,
    credential_id: str,
) -> CredentialLease:
    try:
        lease = await codex_credentials_manager.acquire_lease(user_id, credential_id)
    except ValueError:
        raise _credential_not_found() from None
    if not _is_codex_lease(lease):
        await lease.release()
        raise _credential_not_found()
    return lease


def _is_codex_lease(lease: CredentialLease) -> bool:
    credentials = lease.credentials
    return (
        credentials.type == "oauth2"
        and credentials.provider == "codex"
        and credentials.refresh_strategy == "provider_runtime"
    )


def _get_codex_transport() -> "CodexTransport":
    from backend.integrations.codex.transport import get_codex_transport

    return get_codex_transport()


def _status_response(attempt: CodexDeviceLoginState) -> CodexLoginStatusResponse:
    if attempt.status not in ("pending", "completed", "failed", "canceled"):
        raise RuntimeError("Codex login coordinator returned an invalid status")
    return CodexLoginStatusResponse(status=attempt.status, error=attempt.error)


def _login_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Codex login not found",
    )


def _credential_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Credentials not found",
    )


def _page_headers(nonce: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Content-Security-Policy": (
            "default-src 'none'; "
            f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }


__all__ = [
    "CODEX_LOGIN_STATE_KEY",
    "CodexDeviceLogin",
    "CodexDeviceLoginState",
    "build_device_login_cancel_url",
    "build_device_login_url",
    "render_device_login_page",
    "revoke_codex_credentials",
    "router",
]
