from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession, LocalExecutionTargetMetadata
from backend.copilot.response_model import StreamError
from backend.data.model import OAuth2Credentials
from backend.integrations.microsoft_365_copilot import service
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)


@pytest.mark.asyncio
async def test_a_local_pc_chat_is_refused_not_answered_in_the_cloud(mocker) -> None:
    """Microsoft 365 Copilot never sees the user's machine. Answering a Local
    PC chat here would ignore the folder the user bound it to, so the turn is
    refused before anything is sent to Microsoft."""
    client = mocker.patch.object(service, "Microsoft365CopilotClient")
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        session_id="session-1",
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="credential-1",
        execution_target=LocalExecutionTargetMetadata(
            machine_id="machine-1",
            directory_ref="directory-1",
            allowed_root="/workspace",
            root_fingerprint="a" * 64,
            root_grant="grant-1",
        ),
    )

    events = [
        event
        async for event in service.stream_chat_completion_microsoft_365(
            session_id="session-1",
            message="list my files",
            user_id="user-1",
            session=session,
            credential_lease=MagicMock(
                credentials=OAuth2Credentials(
                    id="credential-1",
                    provider="microsoft_365_copilot",
                    access_token=SecretStr("graph-token"),
                    refresh_token=SecretStr("refresh-token"),
                    scopes=Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES,
                )
            ),
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].code == "local_pc_requires_sdk_engine"
    client.assert_not_called()
