"""List the user's connected integration credentials (metadata only, no secrets)."""

import asyncio
import logging
import weakref
from typing import Any
from urllib.parse import urlparse

from backend.api.features.integrations.router import (
    CredentialsMetaResponse,
    to_meta_response,
)
from backend.copilot.model import ChatSession
from backend.data.model import Credentials, is_sdk_default
from backend.integrations.credentials_store import (
    SYSTEM_CREDENTIAL_IDS,
    provider_matches,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.managed_credentials import (
    ensure_managed_credentials,
    get_managed_provider,
)
from backend.integrations.managed_providers import register_all

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .utils import get_user_credentials

logger = logging.getLogger(__name__)

# Mirrors the bound the integrations router puts on its first-time managed
# credential sweep, so a slow upstream can't hang the tool call.
_MANAGED_PROVISION_TIMEOUT_S = 10.0
_managed_provision_tasks_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Task[bool]]]" = (weakref.WeakKeyDictionary())


class CredentialListResponse(ToolResponseBase):
    """Response listing the user's connected credentials."""

    type: ResponseType = ResponseType.CREDENTIAL_LIST
    credentials: list[CredentialsMetaResponse] = []
    providers: list[str] = []
    count: int = 0
    # False when the managed-credential provisioning sweep timed out or
    # failed, meaning platform-managed integrations may be missing below.
    provisioning_complete: bool = True


class ListUserCredentialsTool(BaseTool):
    """Lists the integrations the user has already connected (never secrets)."""

    @property
    def name(self) -> str:
        return "list_user_credentials"

    @property
    def description(self) -> str:
        return (
            "List the integrations the user has already connected (metadata "
            "only, never secrets). Call before asking the user to sign in or "
            "connect an integration, so you only surface sign-in for "
            "integrations that are genuinely missing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional provider slug to filter by "
                        "(e.g. 'github', 'google', 'notion')."
                    ),
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        provider: str | None = None,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="missing_user_id",
                session_id=session_id,
            )

        register_all()

        # The managed-credential sweep only affects managed providers, and it
        # sits on the hot path (the model calls this before every sign-in
        # prompt). Skip it when filtering to a provider that is never managed.
        wanted_provider = provider.strip().lower() if provider else ""
        if wanted_provider and get_managed_provider(wanted_provider) is None:
            provisioning_complete = True
        else:
            provisioning_complete = await _ensure_managed_credentials_bounded(user_id)

        try:
            all_creds = await get_user_credentials(user_id)
        except Exception:
            logger.exception("Failed to list credentials for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve the user's connected credentials.",
                error="credential_lookup_failed",
                session_id=session_id,
            )

        metas = _serialize_connected_credentials(all_creds, wanted_provider)
        providers = sorted({m.provider for m in metas})

        return CredentialListResponse(
            message=_build_inventory_message(
                metas, providers, wanted_provider, provisioning_complete
            ),
            credentials=metas,
            providers=providers,
            count=len(metas),
            provisioning_complete=provisioning_complete,
            session_id=session_id,
        )


def _serialize_connected_credentials(
    all_creds: list[Credentials], wanted: str
) -> list[CredentialsMetaResponse]:
    """Strip secrets and drop non-user credentials, then apply the provider filter."""
    # System credentials (platform-provided API keys) and SDK defaults are
    # not user-connected integrations, so they'd mislead the model here. Filter
    # on the raw credentials (including the provider filter) before serializing,
    # so to_meta_response only runs on the retained set.
    metas = [
        _to_safe_meta_response(cred)
        for cred in all_creds
        if not is_sdk_default(cred.id)
        and cred.id not in SYSTEM_CREDENTIAL_IDS
        and (not wanted or provider_matches(cred.provider, wanted))
    ]

    return metas


def _to_safe_meta_response(cred: Credentials) -> CredentialsMetaResponse:
    """Serialize credential metadata without secrets embedded in URLs."""
    meta = to_meta_response(cred)
    if meta.host and provider_matches(cred.provider, "mcp"):
        try:
            host = urlparse(meta.host).hostname
        except ValueError:
            host = None
        return meta.model_copy(update={"host": host})
    return meta


def _build_inventory_message(
    metas: list[CredentialsMetaResponse],
    providers: list[str],
    wanted: str,
    provisioning_complete: bool,
) -> str:
    """Compose the model-facing summary of the connected-credential inventory."""
    if metas and wanted:
        message = (
            f"The user has {len(metas)} connected credential(s) for provider "
            f"'{wanted}'."
        )
    elif metas:
        message = (
            f"The user has {len(metas)} connected credential(s) across "
            f"{len(providers)} provider(s): {', '.join(providers)}."
        )
    elif not provisioning_complete:
        scope = f" for provider '{wanted}'" if wanted else ""
        message = (
            f"Credential discovery{scope} is incomplete because "
            "platform-managed provisioning did not complete. Do not treat "
            "the absence of credentials as authoritative; attempt the task "
            "or wait for an explicit missing-credentials response."
        )
    elif wanted:
        message = (
            f"The user has no connected credentials for provider "
            f"'{wanted}'. Use connect_integration to surface a "
            "sign-in card if this integration is needed."
        )
    else:
        message = (
            "The user has not connected any integrations yet. Use "
            "connect_integration to surface a sign-in card if one is needed."
        )

    if metas and not provisioning_complete:
        message += (
            " Note: platform-managed credential provisioning did not "
            "complete, so managed integrations may be missing from this "
            "list — do not treat their absence as authoritative."
        )

    return message


def _get_managed_provision_tasks() -> dict[str, asyncio.Task[bool]]:
    """Return the provisioning task cache for the current worker event loop."""
    loop = asyncio.get_running_loop()
    tasks = _managed_provision_tasks_by_loop.get(loop)
    if tasks is None:
        tasks = {}
        _managed_provision_tasks_by_loop[loop] = tasks
    return tasks


def _managed_provision_finished(
    user_id: str,
    task: asyncio.Task[bool],
    tasks: dict[str, asyncio.Task[bool]],
) -> None:
    if tasks.get(user_id) is task:
        tasks.pop(user_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error:
        logger.error(
            "Managed credential provisioning failed for user %s",
            user_id,
            exc_info=(type(error), error, error.__traceback__),
        )


def _get_managed_provision_task(user_id: str) -> asyncio.Task[bool]:
    tasks = _get_managed_provision_tasks()
    existing = tasks.get(user_id)
    if existing and not existing.done():
        return existing

    task = asyncio.create_task(
        ensure_managed_credentials(user_id, IntegrationCredentialsManager().store)
    )
    tasks[user_id] = task
    task.add_done_callback(
        lambda completed: _managed_provision_finished(user_id, completed, tasks)
    )
    return task


async def _ensure_managed_credentials_bounded(user_id: str) -> bool:
    """Run the managed-credential sweep; return False on timeout or failure."""
    try:
        return await asyncio.wait_for(
            asyncio.shield(_get_managed_provision_task(user_id)),
            timeout=_MANAGED_PROVISION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Managed credential sweep exceeded %.1fs for user=%s; "
            "listing without it",
            _MANAGED_PROVISION_TIMEOUT_S,
            user_id,
        )
        return False
    except Exception:
        logger.exception(
            "Managed credential provisioning failed for user %s; listing without it",
            user_id,
        )
        return False
