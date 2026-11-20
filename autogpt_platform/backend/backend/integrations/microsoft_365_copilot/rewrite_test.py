import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamTextDelta
from backend.data.model import OAuth2Credentials
from backend.integrations.microsoft_365_copilot import service
from backend.integrations.microsoft_365_copilot.client import Microsoft365CopilotError
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshots,fail_after_rewrite,replace_conversation",
    [
        (["Hello world", "Hello [1] world"], False, False),
        (["Hello world", "Goodbye"], False, False),
        (["Hello world", ""], False, False),
        (["Hello world", ""], False, True),
        (["Hello world", ""], True, False),
        (["Hello world", "Hello [1] world", "Hello [1] world!"], False, False),
    ],
)
async def test_final_graph_snapshot_is_persisted_without_corrupting_preview(
    mocker, snapshots: list[str], fail_after_rewrite: bool, replace_conversation: bool
) -> None:
    async def chunks():
        for text in snapshots:
            payload = {"messages": [{"id": "answer", "text": text}]}
            yield f"data: {json.dumps(payload)}\n\n".encode()
        if fail_after_rewrite:
            raise Microsoft365CopilotError("expired", status=410)

    response = MagicMock(status=200)
    response.content.iter_any = chunks
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=None)
    http_session = MagicMock()
    requests = [request]
    if replace_conversation:
        expired = MagicMock()
        expired.__aenter__.return_value = MagicMock(
            status=410, json=AsyncMock(return_value={"error": {"code": "expired"}})
        )
        create = MagicMock()
        create.__aenter__.return_value = MagicMock(
            status=201, json=AsyncMock(return_value={"id": "replacement"})
        )
        requests = [expired, create, request]
    http_session.post.side_effect = requests
    http_session.close = AsyncMock()
    mocker.patch(
        "backend.integrations.microsoft_365_copilot.client.aiohttp.ClientSession",
        return_value=http_session,
    )
    upsert = mocker.patch.object(
        service, "upsert_chat_session", new=AsyncMock(side_effect=lambda value: value)
    )
    usage = mocker.patch.object(service, "persist_and_record_usage", new=AsyncMock())
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="credential-1",
    )
    session.metadata.llm_provider_session_ids["microsoft_365_copilot:credential-1"] = (
        "conversation-1"
    )
    lease = MagicMock(
        credentials=OAuth2Credentials(
            id="credential-1",
            provider="microsoft_365_copilot",
            access_token=SecretStr("graph-token"),
            refresh_token=SecretStr("refresh-token"),
            scopes=Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES,
        )
    )

    events = [
        event
        async for event in service.stream_chat_completion_microsoft_365(
            session_id=session.session_id,
            session=session,
            message="Hi",
            credential_lease=lease,
        )
    ]

    if fail_after_rewrite:
        assert http_session.post.call_count == 1
        assert all(message.role != "assistant" for message in session.messages)
        usage.assert_not_awaited()
        assert [event.type.value for event in events][-2:] == ["error", "finish"]
        return

    assert session.messages[-1].content == snapshots[-1]
    assert upsert.await_args.args[0].messages[-1].content == snapshots[-1]
    assert usage.await_args.kwargs["completion_tokens"] == service._estimate_tokens(
        snapshots[-1]
    )
    assert (
        "".join(event.delta for event in events if isinstance(event, StreamTextDelta))
        == snapshots[0]
    )
    assert events[-1].type.value == "finish"
