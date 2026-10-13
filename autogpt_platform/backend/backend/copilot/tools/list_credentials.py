"""List the user's connected integration credentials (metadata only, no secrets)."""

import asyncio
import ipaddress
import logging
import re
import weakref
from typing import Any
from urllib.parse import urlsplit

from backend.api.features.integrations.router import (
    CredentialsMetaResponse,
    to_meta_response,
)
from backend.copilot.capabilities.resolve import connection_state_from
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
from .utils import get_user_credentials, scope_credentials_to_expert

logger = logging.getLogger(__name__)

# Mirrors the bound the integrations router puts on its first-time managed
# credential sweep, so a slow upstream can't hang the tool call.
_MANAGED_PROVISION_TIMEOUT_S = 10.0
_ProvisionTasks = dict[str, "asyncio.Task[bool]"]
_managed_provision_tasks_by_loop: (
    "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _ProvisionTasks]"
) = weakref.WeakKeyDictionary()
_CONNECT_HINT = (
    "If one is needed, find it with find_capability; "
    'run_capability(id, input={"connect": true}) surfaces its sign-in card.'
)
_DNS_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOSTNAME_RE = re.compile(rf"{_DNS_LABEL}(?:\.{_DNS_LABEL})*")


class CredentialListResponse(ToolResponseBase):
    """Response listing the user's connected credentials."""

    type: ResponseType = ResponseType.CREDENTIAL_LIST
    credentials: list[CredentialsMetaResponse] = []
    providers: list[str] = []
    count: int = 0
    # False when the managed-credential provisioning sweep timed out or
    # failed, meaning platform-managed integrations may be missing below.
    provisioning_complete: bool = True
    # Expert sessions only: providers the account has connected but this
    # expert was not granted. They need a grant, not a sign-in.
    ungranted_providers: list[str] = []


class ListUserCredentialsTool(BaseTool):
    """Lists the integrations the user has already connected (never secrets)."""

    @property
    def name(self) -> str:
        return "list_user_credentials"

    @property
    def description(self) -> str:
        return (
            "List every integration the user has connected (metadata only, "
            "never secrets). For one service, find_capability's `connected` "
            "already answers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Optional provider filter, e.g. 'github'.",
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

        # The managed-credential sweep only affects managed providers and can
        # take seconds, so skip it when filtering to a provider that is never
        # managed.
        wanted_provider = provider.strip().lower() if provider else ""
        if wanted_provider and get_managed_provider(wanted_provider) is None:
            provisioning_complete = True
        else:
            provisioning_complete = await _ensure_managed_credentials_bounded(user_id)

        try:
            all_creds = await get_user_credentials(user_id)
            granted = await scope_credentials_to_expert(
                user_id, session.expert_id, all_creds
            )
        except Exception:
            logger.exception("Failed to list credentials for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve the user's connected credentials.",
                error="credential_lookup_failed",
                session_id=session_id,
            )

        metas = _serialize_connected_credentials(granted, wanted_provider)
        providers = sorted({m.provider for m in metas})
        ungranted = _ungranted_providers(all_creds, granted, wanted_provider)

        return CredentialListResponse(
            message=_build_inventory_message(
                metas, providers, wanted_provider, provisioning_complete, ungranted
            ),
            credentials=metas,
            providers=providers,
            count=len(metas),
            provisioning_complete=provisioning_complete,
            ungranted_providers=ungranted,
            session_id=session_id,
        )


def _user_connected(creds: list[Credentials], wanted: str) -> list[Credentials]:
    """Drop non-user credentials, then apply the provider filter."""
    # System credentials (platform-provided API keys) and SDK defaults are
    # not user-connected integrations, so they'd mislead the model here.
    return [
        cred
        for cred in creds
        if not is_sdk_default(cred.id)
        and cred.id not in SYSTEM_CREDENTIAL_IDS
        and (not wanted or provider_matches(cred.provider, wanted))
    ]


def _serialize_connected_credentials(
    creds: list[Credentials], wanted: str
) -> list[CredentialsMetaResponse]:
    """Strip secrets from the user-connected credentials that match *wanted*."""
    return [_to_safe_meta_response(cred) for cred in _user_connected(creds, wanted)]


def _ungranted_providers(
    all_creds: list[Credentials], granted: list[Credentials], wanted: str
) -> list[str]:
    """Providers the account holds but the session's expert was not granted.

    Uses the same connection state ``find_capability`` ranks on, so the two
    tools agree on what "connected" and "needs a grant" mean.
    """
    granted_ids = {cred.id for cred in granted}
    ungranted = [
        c for c in _user_connected(all_creds, wanted) if c.id not in granted_ids
    ]
    return sorted(connection_state_from(ungranted).providers)


def _to_safe_meta_response(cred: Credentials) -> CredentialsMetaResponse:
    """Serialize credential metadata without secrets embedded in URLs."""
    meta = to_meta_response(cred)
    if meta.host and provider_matches(cred.provider, "mcp"):
        return meta.model_copy(update={"host": _safe_server_url(meta.host)})
    return meta


def _safe_server_url(raw: str) -> str | None:
    """An MCP server URL without userinfo, query string or fragment.

    Host, port and path are kept: they are what tell two servers on one host
    apart. Anything that does not parse to a plain DNS hostname is dropped.
    """
    value = raw.strip()
    has_scheme = "://" in value
    try:
        parts = urlsplit(value if has_scheme else f"//{value}")
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not hostname or not _is_plain_host(hostname):
        return None
    if any(ch.isspace() for ch in parts.path):
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port else host
    prefix = f"{parts.scheme}://" if has_scheme and parts.scheme else ""
    return f"{prefix}{netloc}{parts.path}"


def _is_plain_host(hostname: str) -> bool:
    """A DNS name or an IP literal, nothing an embedded secret could hide in."""
    if _HOSTNAME_RE.fullmatch(hostname):
        return True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def _build_inventory_message(
    metas: list[CredentialsMetaResponse],
    providers: list[str],
    wanted: str,
    provisioning_complete: bool,
    ungranted: list[str],
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
    elif ungranted:
        message = "This expert has not been granted any matching integration."
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
            f"'{wanted}'. {_CONNECT_HINT}"
        )
    else:
        message = f"The user has not connected any integrations yet. {_CONNECT_HINT}"

    if ungranted:
        message += (
            f" Connected on the account but not granted to this expert: "
            f"{', '.join(ungranted)}. Do not ask the user to sign in to these; "
            "ask them to grant access."
        )
    if metas and not provisioning_complete:
        message += (
            " Note: platform-managed credential provisioning did not "
            "complete, so managed integrations may be missing from this "
            "list — do not treat their absence as authoritative."
        )

    return message


def _get_managed_provision_tasks() -> _ProvisionTasks:
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
    tasks: _ProvisionTasks,
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
