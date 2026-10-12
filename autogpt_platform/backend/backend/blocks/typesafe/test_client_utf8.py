import base64
import json
import logging

import httpx2
import pytest
from typesafe_sdk import Noul

from backend.blocks.typesafe import _client
from backend.blocks.typesafe._client_test import mock_transport

BASE64_DATA_URL_PREFIX = "data:application/octet-stream;base64,"


@pytest.mark.parametrize("status", [200, 401, 502])
@pytest.mark.parametrize(
    "body",
    [b"\xff private-response \\xff", b'{"error":"private-response\xff\\xff"}'],
    ids=["non-json", "inside-json-string"],
)
async def test_malformed_utf8_preserves_exact_bytes_and_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
    body: bytes,
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, content=body, headers={"x-typesafe-request-id": "invalid-utf8-id"}
        )

    clients = mock_transport(monkeypatch, handler)
    with caplog.at_level(logging.DEBUG, logger="typesafe_sdk"):
        result = await _client.call_jev(
            "test-secret", "test-state", {"q": Noul(instructions="Yes?")}
        )
    assert result.response is not None
    assert result.response.startswith(BASE64_DATA_URL_PREFIX)
    assert (
        base64.b64decode(
            result.response.removeprefix(BASE64_DATA_URL_PREFIX), validate=True
        )
        == body
    )
    assert result.request_id == "invalid-utf8-id"
    assert json.loads(result.request)["state"] == "test-state"
    assert result.answers == {}
    assert result.input_tokens is None and result.output_tokens is None
    assert "UTF-8" in result.error and "Base64" in result.error
    assert result.latency_ms >= 0
    assert json.loads(result.model_dump_json())["response"] == result.response
    assert clients[0]._http_client.is_closed
    assert "private-response" not in caplog.text


@pytest.mark.parametrize(
    "body", [b'{"error":"literal \\xff stays literal"}', '{"error":"é"}'.encode()]
)
async def test_valid_utf8_error_response_remains_verbatim(
    monkeypatch: pytest.MonkeyPatch, body: bytes
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            502, content=body, headers={"x-typesafe-request-id": "valid-utf8-id"}
        )

    mock_transport(monkeypatch, handler)
    result = await _client.call_jev(
        "test", "test-state", {"q": Noul(instructions="Yes?")}
    )
    assert result.response == body.decode("utf-8")
    assert result.error == "Jev API request failed (HTTP 502)."
