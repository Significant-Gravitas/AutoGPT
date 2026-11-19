"""The connected account a tool call used, for its line in the task record.

A call's credential is loaded deep inside the tool (a block's in
``execute_block``, an MCP server's in ``run_mcp_tool``), out of the
recorder's sight. ``BaseTool.execute`` opens a collector for each call, the
code that loads the credential notes the account on it, and the recorder reads
it once the call returns.

Only what the account's owner sees in their integrations list leaves here:
the service and the account's email, handle or title. Never a token, a key, a
credential id or a scope.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from backend.blocks.llm import LLM_PROVIDER_NAMES
from backend.data.model import Credentials
from backend.data.task_activity_meta import LABEL_MAX_CHARS, TaskActivityAccount
from backend.data.task_record import redact
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.provider_titles import provider_title
from backend.integrations.service_identity import (
    UNKNOWN_MCP_SERVICE,
    service_for_credential,
)

logger = logging.getLogger(__name__)

_collector: ContextVar[list[TaskActivityAccount] | None] = ContextVar(
    "call_accounts", default=None
)


@contextmanager
def collect() -> Iterator[list[TaskActivityAccount]]:
    """Gather the accounts the code inside uses, in the order it uses them."""
    accounts: list[TaskActivityAccount] = []
    token = _collector.set(accounts)
    try:
        yield accounts
    finally:
        _collector.reset(token)


def note(credentials: Credentials) -> None:
    """The running call is about to use *credentials*; a no-op outside a call.
    Never raises: the record matters less than the call it describes."""
    accounts = _collector.get()
    if accounts is None:
        return
    try:
        account = account_for(credentials)
    except Exception:
        logger.warning("Could not describe the account a call used", exc_info=True)
        return
    if account is not None and account not in accounts:
        accounts.append(account)


def account_for(credentials: Credentials) -> TaskActivityAccount | None:
    """The account as its owner knows it, or None for a model provider's key
    (the platform's own, not a connected integration) or an unknown server."""
    identity = service_for_credential(credentials)
    service = identity.service
    if service in LLM_PROVIDER_NAMES or service == UNKNOWN_MCP_SERVICE:
        return None
    name = identity.name or provider_title(service)
    # The platform's shared keys belong to no account of the user's.
    label = None if is_system_credential(credentials.id) else _label(credentials)
    if label and label.casefold() in (name.casefold(), service.casefold()):
        label = None
    return TaskActivityAccount(service=service, name=name, label=label)


def _label(credentials: Credentials) -> str | None:
    """The account's email or handle, else the title it was saved under;
    None when that text could be a secret or names the credential's id."""
    username = credentials.username if credentials.type == "oauth2" else None
    text = " ".join((username or credentials.title or "").split())
    if not text or len(text) > LABEL_MAX_CHARS or credentials.id in text:
        return None
    return text if redact(text) == text else None
