"""How the executor reports a Microsoft 365 Copilot turn whose credential
lease fails: a missing or unusable credential is "not found", and anything
else, Microsoft refusing to renew the grant included, keeps its own cause."""

import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatSession
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, code",
    [
        # Microsoft refusing to renew a valid account's grant is not a
        # missing credential, though MicrosoftAuthError is a ValueError.
        (
            MicrosoftAuthError(
                microsoft_recovery("invalid_grant", "AADSTS70043: expired")
            ),
            "microsoft_365_copilot_credential_unavailable",
        ),
        (
            ValueError("no such credential"),
            "microsoft_365_copilot_credential_not_found",
        ),
    ],
)
async def test_a_lease_failure_keeps_its_cause(failure, code):
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(side_effect=failure)
    completed = AsyncMock()

    with (
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=_session()),
        ),
        patch(
            "backend.copilot.executor.processor.IntegrationCredentialsManager",
            return_value=manager,
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.publish_chunk",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=completed,
        ),
        pytest.raises(RuntimeError) as raised,
    ):
        await CoPilotProcessor()._execute_async(
            _entry(),
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
        )

    assert str(raised.value) == code
    assert completed.await_args.kwargs["error_message"] == code


def _entry() -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="sess-microsoft",
        turn_id="turn-microsoft",
        user_id="user-1",
        message="hi",
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="cred-microsoft",
    )


def _session() -> ChatSession:
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="cred-microsoft",
    )
    session.session_id = "sess-microsoft"
    return session
