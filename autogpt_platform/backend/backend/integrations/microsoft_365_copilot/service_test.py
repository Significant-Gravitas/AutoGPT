from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.stream_kwargs = None
        self.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def create_conversation(self) -> str:
        return "conversation-1"

    async def stream_chat(self, conversation_id: str, message: str, **kwargs):
        self.stream_kwargs = {
            "conversation_id": conversation_id,
            "message": message,
            **kwargs,
        }
        yield "Hello"
        yield " world"


@pytest.mark.asyncio
async def test_service_streams_and_persists_graph_conversation(mocker) -> None:
    from backend.integrations.microsoft_365_copilot import service

    session = ChatSession.new(
        "user-1",
        dry_run=False,
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="credential-1",
    )
    session.session_id = "session-1"
    credentials = OAuth2Credentials(
        id="credential-1",
        provider="microsoft_365_copilot",
        access_token=SecretStr("graph-token"),
        refresh_token=SecretStr("refresh-token"),
        scopes=Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES,
    )
    lease = MagicMock(credentials=credentials)
    upsert = mocker.patch.object(
        service, "upsert_chat_session", new=AsyncMock(side_effect=lambda value: value)
    )
    record_usage = mocker.patch.object(
        service, "persist_and_record_usage", new=AsyncMock(return_value=4)
    )
    mocker.patch.object(service, "Microsoft365CopilotClient", _FakeClient)
    mocker.patch.object(
        service,
        "get_user_by_id",
        new=AsyncMock(return_value=SimpleNamespace(timezone="America/Chicago")),
    )

    events = [
        event
        async for event in service.stream_chat_completion_microsoft_365(
            session_id="session-1",
            message="Hi",
            is_user_message=True,
            user_id="user-1",
            session=session,
            context={"project": "launch"},
            credential_lease=lease,
        )
    ]

    assert [event.type.value for event in events] == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    key = "microsoft_365_copilot:credential-1"
    assert session.metadata.llm_provider_session_ids[key] == "conversation-1"
    assert session.messages[-1].role == "assistant"
    assert session.messages[-1].content == "Hello world"
    client = _FakeClient.instances[-1]
    assert client.access_token == "graph-token"
    assert client.stream_kwargs["timezone"] == "America/Chicago"
    assert "launch" in client.stream_kwargs["additional_context"][0]
    assert upsert.await_count == 2
    record_usage.assert_awaited_once()


def test_conversation_key_is_scoped_to_credential() -> None:
    from backend.integrations.microsoft_365_copilot.service import _conversation_key

    session = ChatSession.new(
        "user-1",
        dry_run=False,
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="credential-2",
    )

    assert _conversation_key(session) == "microsoft_365_copilot:credential-2"
