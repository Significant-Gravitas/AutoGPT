"""Exa REST helpers: the client settings, error messages and path segments."""

from unittest.mock import MagicMock, patch

import pytest

from backend.blocks.exa import _exa_rest
from backend.blocks.exa._exa_rest import error_message, exa_request, path_segment
from backend.blocks.exa._test import TEST_CREDENTIALS


def test_error_message_reads_nested_agent_api_errors():
    body = '{"error": {"type": "invalid_request", "code": "BAD_EFFORT", "message": "effort is invalid"}}'
    assert error_message(body) == "effort is invalid [BAD_EFFORT]"
    assert error_message('{"error": "Not found", "tag": "NOT_FOUND"}') == (
        "Not found [NOT_FOUND]"
    )


def test_path_segment_keeps_ids_inside_one_segment():
    assert path_segment("mon/../x?y") == "mon%2F..%2Fx%3Fy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, attempts",
    [("GET", 3), ("HEAD", 3), ("DELETE", 1), ("POST", 1), ("PATCH", 1)],
)
async def test_writes_and_deletes_are_sent_once_and_the_key_never_leaves_exa(
    method, attempts
):
    response = MagicMock(ok=True, content=b"")

    async def request(*_, **__):
        return response

    client = MagicMock(request=request)
    with patch.object(_exa_rest, "Requests", return_value=client) as requests:
        await exa_request(TEST_CREDENTIALS, method, "https://api.exa.ai/x")

    kwargs = requests.call_args.kwargs
    assert kwargs["retry_max_attempts"] == attempts
    assert "x-api-key" in kwargs["sensitive_headers"]
