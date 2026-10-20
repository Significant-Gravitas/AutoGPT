from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamError
from backend.integrations.microsoft_365_copilot import service
from backend.integrations.microsoft_365_copilot.service_test import _credential_lease


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_conversation", [False, True])
@pytest.mark.parametrize(
    "stream_bytes",
    [
        b"",
        b"data: [DONE]\n\n",
        b'data: {"messages":[]}\n\n',
        b'data: {"messages":[{"id":"answer","text":null}]}\n\n',
    ],
)
async def test_stream_with_no_updates_fails_without_persisting_an_answer(
    mocker, stream_bytes: bytes, replace_conversation: bool
) -> None:
    async def chunks():
        yield stream_bytes

    response = MagicMock(status=200)
    response.content.iter_any = chunks
    stream_request = MagicMock()
    stream_request.__aenter__.return_value = response
    requests = [stream_request]
    if replace_conversation:
        expired_request = MagicMock()
        expired_request.__aenter__.return_value = MagicMock(
            status=410, json=AsyncMock(return_value={"error": {"code": "expired"}})
        )
        create_request = MagicMock()
        create_request.__aenter__.return_value = MagicMock(
            status=201, json=AsyncMock(return_value={"id": "replacement"})
        )
        requests = [expired_request, create_request, stream_request]
    http_session = MagicMock(close=AsyncMock())
    http_session.post.side_effect = requests
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

    events = [
        event
        async for event in service.stream_chat_completion_microsoft_365(
            session_id=session.session_id,
            session=session,
            message="Hi",
            credential_lease=_credential_lease(),
        )
    ]

    assert [event.type.value for event in events] == [
        "start",
        "start-step",
        "text-start",
        "text-end",
        "finish-step",
        "error",
        "finish",
    ]
    error = next(event for event in events if isinstance(event, StreamError))
    assert error.code == "microsoft_365_copilot_request_failed"
    assert error.errorText == "Microsoft 365 Copilot returned no answer"
    assert all(message.role != "assistant" for message in session.messages)
    usage.assert_not_awaited()
    assert upsert.await_count == int(replace_conversation)
    assert http_session.post.call_count == (3 if replace_conversation else 1)
