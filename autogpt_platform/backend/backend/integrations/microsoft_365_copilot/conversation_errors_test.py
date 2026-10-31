import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from backend.integrations.microsoft_365_copilot.client import (
    Microsoft365CopilotClient,
    Microsoft365CopilotError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,decode_error",
    [
        (b'{"private": invalid}', json.JSONDecodeError),
        (b'"private\xff"', UnicodeDecodeError),
    ],
)
async def test_malformed_successful_conversation_response_is_normalized(
    payload: bytes, decode_error: type[Exception]
) -> None:
    async def decode_json():
        return json.loads(payload.decode("utf-8"))

    response = MagicMock(spec=aiohttp.ClientResponse)
    response.status = 201
    response.json = AsyncMock(side_effect=decode_json)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value.__aenter__.return_value = response
    client = Microsoft365CopilotClient("token")
    client._session = session

    with pytest.raises(
        Microsoft365CopilotError, match="invalid conversation response"
    ) as raised:
        await client.create_conversation()

    assert isinstance(raised.value.__cause__, decode_error)
    assert "private" not in str(raised.value)
