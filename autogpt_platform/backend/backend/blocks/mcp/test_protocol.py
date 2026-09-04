"""
Unit tests for the MCP 2026-07-28 protocol helpers and the client's
dual-era behaviour (era detection, version negotiation, header mirroring,
MRTR handling).  No network: HTTP replies are faked at ``MCPClient._post``.
"""

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.blocks.mcp.protocol import (
    ERROR_HEADER_MISMATCH,
    ERROR_METHOD_NOT_FOUND,
    ERROR_MISSING_CLIENT_CAPABILITY,
    ERROR_UNSUPPORTED_PROTOCOL_VERSION,
    HEADER_METHOD,
    HEADER_NAME,
    HEADER_PROTOCOL_VERSION,
    HEADER_SESSION_ID,
    LEGACY_PROTOCOL_VERSION,
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    MODERN_PROTOCOL_VERSION,
    HeaderParam,
    InvalidHeaderAnnotation,
    MCPProtocolEra,
    MCPServerEraCache,
    MCPServerProtocol,
    build_request_meta,
    collect_header_params,
    decode_header_value,
    encode_header_value,
    era_cache,
    extract_header_params,
    is_modern_jsonrpc_error,
    name_header_for,
    negotiate_version,
)
from backend.util.request import HTTPClientError, HTTPServerError, Response

SERVER_URL = "https://mcp.example.com/mcp"


# ───────────────────────── header encoding ─────────────────────────


class TestHeaderEncoding:
    def test_plain_ascii_passthrough(self):
        assert encode_header_value("us-west1") == "us-west1"

    @pytest.mark.parametrize(
        "raw",
        ["Hello, 世界", " padded ", "line1\nline2", "=?base64?literal?=", "tab\tin"],
    )
    def test_unsafe_values_are_base64_wrapped(self, raw: str):
        encoded = encode_header_value(raw)
        assert encoded.startswith("=?base64?") and encoded.endswith("?=")
        inner = encoded[len("=?base64?") : -2]
        assert base64.b64decode(inner).decode() == raw
        assert decode_header_value(encoded) == raw

    def test_spec_example(self):
        assert encode_header_value("Hello, 世界") == "=?base64?SGVsbG8sIOS4lueVjA==?="

    def test_decode_leaves_plain_values(self):
        assert decode_header_value("plain") == "plain"

    def test_name_header_only_for_named_methods(self):
        assert name_header_for("tools/call", {"name": "get_weather"}) == "get_weather"
        assert name_header_for("resources/read", {"uri": "file:///a"}) == "file:///a"
        assert name_header_for("tools/list", None) is None
        assert name_header_for("server/discover", {"name": "x"}) is None


# ───────────────────────── request metadata ─────────────────────────


class TestRequestMeta:
    def test_meta_has_required_fields(self):
        meta = build_request_meta("2026-07-28")
        assert meta[META_PROTOCOL_VERSION] == "2026-07-28"
        assert meta[META_CLIENT_INFO]["name"] == "AutoGPT-Platform"
        assert meta[META_CLIENT_CAPABILITIES] == {}

    def test_negotiate_picks_shared_version(self):
        assert negotiate_version(["2025-11-25", "2026-07-28"]) == "2026-07-28"
        assert negotiate_version(["2025-11-25"]) is None
        assert negotiate_version("2026-07-28") is None
        assert negotiate_version(None) is None

    def test_modern_error_detection(self):
        assert is_modern_jsonrpc_error({"error": {"code": -32022}})
        assert is_modern_jsonrpc_error({"error": {"code": -32020}})
        assert not is_modern_jsonrpc_error({"error": {"code": -32601}})
        assert not is_modern_jsonrpc_error({"result": {}})
        assert not is_modern_jsonrpc_error(None)


# ───────────────────────── x-mcp-header ─────────────────────────


class TestHeaderParams:
    def test_collects_reachable_annotations(self):
        schema = {
            "type": "object",
            "properties": {
                "region": {"type": "string", "x-mcp-header": "Region"},
                "opts": {
                    "type": "object",
                    "properties": {
                        "retries": {"type": "integer", "x-mcp-header": "Retries"},
                        "dry": {"type": "boolean", "x-mcp-header": "Dry-Run"},
                    },
                },
                "query": {"type": "string"},
            },
        }
        params = collect_header_params(schema)
        assert params == [
            HeaderParam(path=("region",), header_name="Region", json_type="string"),
            HeaderParam(
                path=("opts", "retries"), header_name="Retries", json_type="integer"
            ),
            HeaderParam(
                path=("opts", "dry"), header_name="Dry-Run", json_type="boolean"
            ),
        ]

    def test_extract_formats_and_omits_missing(self):
        params = [
            HeaderParam(path=("region",), header_name="Region", json_type="string"),
            HeaderParam(
                path=("opts", "retries"), header_name="Retries", json_type="integer"
            ),
            HeaderParam(
                path=("opts", "dry"), header_name="Dry-Run", json_type="boolean"
            ),
            HeaderParam(path=("absent",), header_name="Absent", json_type="string"),
            HeaderParam(path=("nul",), header_name="Nul", json_type="string"),
        ]
        headers = extract_header_params(
            params,
            {"region": "eu-west", "opts": {"retries": 3, "dry": False}, "nul": None},
        )
        assert headers == {
            "Mcp-Param-Region": "eu-west",
            "Mcp-Param-Retries": "3",
            "Mcp-Param-Dry-Run": "false",
        }

    def test_extract_encodes_unsafe_values(self):
        params = [
            HeaderParam(path=("greeting",), header_name="Greeting", json_type="string")
        ]
        headers = extract_header_params(params, {"greeting": "Hello, 世界"})
        assert headers["Mcp-Param-Greeting"] == "=?base64?SGVsbG8sIOS4lueVjA==?="

    @pytest.mark.parametrize(
        "schema",
        [
            # empty name
            {"properties": {"a": {"type": "string", "x-mcp-header": ""}}},
            # trailing newline (``$`` would accept it, ``fullmatch`` must not)
            {"properties": {"a": {"type": "string", "x-mcp-header": "Region\n"}}},
            # annotation on the root schema itself
            {"type": "object", "x-mcp-header": "Root", "properties": {}},
            {"type": "object", "x-mcp-header": None, "properties": {}},
            # present but null
            {"properties": {"a": {"type": "string", "x-mcp-header": None}}},
            # invalid field-name characters
            {"properties": {"a": {"type": "string", "x-mcp-header": "Bad Name"}}},
            {"properties": {"a": {"type": "string", "x-mcp-header": "X\r\nY"}}},
            # number is not allowed
            {"properties": {"a": {"type": "number", "x-mcp-header": "A"}}},
            # duplicate (case-insensitive)
            {
                "properties": {
                    "a": {"type": "string", "x-mcp-header": "Region"},
                    "b": {"type": "string", "x-mcp-header": "region"},
                }
            },
            # behind items / composition / $ref
            {
                "properties": {
                    "a": {
                        "type": "array",
                        "items": {"type": "string", "x-mcp-header": "A"},
                    }
                }
            },
            {"oneOf": [{"properties": {"a": {"type": "string", "x-mcp-header": "A"}}}]},
            {"$defs": {"x": {"type": "string", "x-mcp-header": "A"}}},
        ],
    )
    def test_invalid_annotations_are_rejected(self, schema: dict[str, Any]):
        with pytest.raises(InvalidHeaderAnnotation):
            collect_header_params(schema)

    def test_schema_without_annotations(self):
        assert collect_header_params({"type": "object", "properties": {}}) == []
        assert collect_header_params(None) == []


# ───────────────────────── era cache ─────────────────────────


class TestEraCache:
    def test_set_get_forget(self):
        cache = MCPServerEraCache(ttl_seconds=60)
        value = MCPServerProtocol(
            era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
        )
        assert cache.get(SERVER_URL) is None
        cache.set(SERVER_URL, value)
        assert cache.get(SERVER_URL) == value
        cache.forget(SERVER_URL)
        assert cache.get(SERVER_URL) is None

    def test_entries_expire(self):
        cache = MCPServerEraCache(ttl_seconds=0)
        cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
            ),
        )
        assert cache.get(SERVER_URL) is None

    def test_least_recently_used_entry_is_evicted(self):
        cache = MCPServerEraCache(ttl_seconds=60, max_entries=2)
        legacy = MCPServerProtocol(
            era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
        )
        cache.set("https://a.example/mcp", legacy)
        cache.set("https://b.example/mcp", legacy)
        # Touching ``a`` makes ``b`` the oldest entry.
        assert cache.get("https://a.example/mcp") is legacy
        cache.set("https://c.example/mcp", legacy)
        assert len(cache) == 2
        assert cache.get("https://b.example/mcp") is None
        assert cache.get("https://a.example/mcp") is legacy
        assert cache.get("https://c.example/mcp") is legacy


# ───────────────────────── fake transport ─────────────────────────


class _FakeResponse(Response):
    """A canned ``Response``; decoding and status handling are inherited."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        status: int,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        text: str | None = None,
    ):
        self.status = status
        self.reason = "reason"
        self.headers = {"content-type": "application/json"}  # type: ignore[assignment]
        self.headers.update(headers or {})
        self.url = SERVER_URL
        if text is not None:
            self.content = text.encode()
        elif body is None:
            self.content = b""
        else:
            self.content = json.dumps(body).encode()


def _rpc_result(result: Any, request_id: int = 1) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(code: int, message: str = "err", data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": 1, "error": error}


DISCOVER_RESULT = {
    "resultType": "complete",
    "supportedVersions": [MODERN_PROTOCOL_VERSION],
    "capabilities": {"tools": {}},
    "_meta": {META_SERVER_INFO: {"name": "modern", "version": "1"}},
}

LEGACY_INIT_RESULT = {
    "protocolVersion": LEGACY_PROTOCOL_VERSION,
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "legacy", "version": "1"},
}


class _Transport:
    """Scripted replies keyed by JSON-RPC method, recording every request."""

    def __init__(self, script: dict[str, list[_FakeResponse]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.sent: list[tuple[dict[str, Any], dict[str, str]]] = []

    def __call__(self, payload: dict[str, Any], headers: dict[str, str]):
        self.sent.append((payload, headers))
        queue = self.script.get(payload["method"])
        if not queue:
            raise AssertionError(f"unexpected request {payload['method']}")
        # Replies are consumed in order; the last one is sticky and answers
        # every further request, so a one-entry script is "always reply X".
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def methods(self) -> list[str]:
        return [p["method"] for p, _ in self.sent]


def _client(transport: _Transport, **kwargs) -> MCPClient:
    client = MCPClient(SERVER_URL, **kwargs)
    patch.object(client, "_post", side_effect=transport).start()
    patch.object(client, "_send_notification").start()
    return client


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


# ───────────────────────── era detection ─────────────────────────


class TestEraDetection:
    async def test_modern_server_detected_via_discover(self):
        transport = _Transport(
            {"server/discover": [_FakeResponse(200, _rpc_result(DISCOVER_RESULT))]}
        )
        client = _client(transport)
        result = await client.initialize()

        assert client.era is MCPProtocolEra.MODERN
        assert client.protocol_version == MODERN_PROTOCOL_VERSION
        assert result["protocolVersion"] == MODERN_PROTOCOL_VERSION
        assert result["serverInfo"] == {"name": "modern", "version": "1"}
        assert transport.methods() == ["server/discover"]

        payload, headers = transport.sent[0]
        meta = payload["params"]["_meta"]
        assert meta[META_PROTOCOL_VERSION] == MODERN_PROTOCOL_VERSION
        assert headers[HEADER_PROTOCOL_VERSION] == MODERN_PROTOCOL_VERSION
        assert headers[HEADER_METHOD] == "server/discover"
        assert HEADER_NAME not in headers
        assert HEADER_SESSION_ID not in headers
        cached = era_cache.get(SERVER_URL)
        assert cached == MCPServerProtocol(
            era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
        )

    async def test_legacy_server_answering_method_not_found(self):
        """Legacy servers reply 200 + -32601 to an unknown method."""
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND))
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
            }
        )
        client = _client(transport)
        result = await client.initialize()

        assert client.era is MCPProtocolEra.LEGACY
        assert result == LEGACY_INIT_RESULT
        assert transport.methods() == ["server/discover", "initialize"]
        init_payload, init_headers = transport.sent[1]
        assert init_payload["params"]["protocolVersion"] == LEGACY_PROTOCOL_VERSION
        assert "_meta" not in init_payload["params"]
        assert HEADER_PROTOCOL_VERSION not in init_headers
        assert HEADER_METHOD not in init_headers

    async def test_legacy_server_answering_plain_400(self):
        """Reference legacy SDKs answer 400 'Missing session ID' with no JSON."""
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(
                        400,
                        text="Bad Request: Missing session ID",
                        headers={"content-type": "text/plain"},
                    )
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.LEGACY
        assert era_cache.get(SERVER_URL) == MCPServerProtocol(
            era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
        )

    @pytest.mark.parametrize("status", [404, 405, 406, 415])
    async def test_legacy_server_other_4xx_without_modern_body(self, status: int):
        transport = _Transport(
            {
                "server/discover": [_FakeResponse(status)],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.LEGACY

    async def test_modern_400_with_modern_error_is_not_a_fallback(self):
        """A 400 carrying a spec-reserved error code proves a modern server."""
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(400, _rpc_error(ERROR_HEADER_MISMATCH, "mismatch"))
                ],
            }
        )
        client = _client(transport)
        with pytest.raises(MCPClientError, match=r"\[-32020\]"):
            await client.initialize()
        assert transport.methods() == ["server/discover"]
        assert client.era is None

    async def test_version_negotiation_on_probe(self):
        rejected = _FakeResponse(
            400,
            _rpc_error(
                ERROR_UNSUPPORTED_PROTOCOL_VERSION,
                data={"supported": ["2099-01-01", MODERN_PROTOCOL_VERSION]},
            ),
        )
        accepted = _FakeResponse(200, _rpc_result(DISCOVER_RESULT))
        transport = _Transport({"server/discover": [rejected, accepted]})
        client = _client(transport)
        await client.initialize()

        assert client.era is MCPProtocolEra.MODERN
        assert transport.methods() == ["server/discover", "server/discover"]
        _, second_headers = transport.sent[1]
        assert second_headers[HEADER_PROTOCOL_VERSION] == MODERN_PROTOCOL_VERSION

    async def test_no_mutual_version_raises(self):
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(
                        400,
                        _rpc_error(
                            ERROR_UNSUPPORTED_PROTOCOL_VERSION,
                            data={"supported": ["2099-01-01"]},
                        ),
                    )
                ]
            }
        )
        client = _client(transport)
        with pytest.raises(MCPClientError, match="does not support any protocol"):
            await client.initialize()

    @pytest.mark.parametrize("status", [401, 403, 407, 429])
    async def test_auth_failures_propagate_without_fallback(self, status: int):
        transport = _Transport({"server/discover": [_FakeResponse(status)]})
        client = _client(transport, auth_token="bad")
        with pytest.raises(HTTPClientError) as exc:
            await client.initialize()
        assert exc.value.status_code == status
        assert transport.methods() == ["server/discover"]
        assert era_cache.get(SERVER_URL) is None

    async def test_server_error_propagates(self):
        transport = _Transport({"server/discover": [_FakeResponse(503)]})
        client = _client(transport)
        with pytest.raises(HTTPServerError):
            await client.initialize()

    async def test_discover_result_without_versions_is_still_modern(self):
        """Only a modern server answers server/discover with a *result*."""
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(
                        200,
                        _rpc_result(
                            {
                                "capabilities": {"tools": {}},
                                "_meta": {META_SERVER_INFO: {"name": "m"}},
                            }
                        ),
                    )
                ]
            }
        )
        client = _client(transport)
        result = await client.initialize()
        assert client.era is MCPProtocolEra.MODERN
        assert client.protocol_version == MODERN_PROTOCOL_VERSION
        assert result["serverInfo"] == {"name": "m"}
        assert transport.methods() == ["server/discover"]

    async def test_cached_legacy_skips_modern_probe(self):
        era_cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
            ),
        )
        transport = _Transport(
            {"initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))]}
        )
        client = _client(transport)
        await client.initialize()
        assert transport.methods() == ["initialize"]

    async def test_cached_legacy_reprobes_when_handshake_rejected(self):
        """A server that upgraded to modern-only rejects initialize; re-detect."""
        era_cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
            ),
        )
        transport = _Transport(
            {
                "initialize": [_FakeResponse(404, _rpc_error(ERROR_METHOD_NOT_FOUND))],
                "server/discover": [_FakeResponse(200, _rpc_result(DISCOVER_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.MODERN
        assert transport.methods() == ["initialize", "server/discover"]

    async def test_cached_modern_reprobes_when_discover_fails(self):
        era_cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
            ),
        )
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND)),
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND)),
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.LEGACY

    async def test_cached_modern_reprobes_after_plain_400(self):
        """A rolled-back server answers server/discover with a bare 400."""
        era_cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
            ),
        )
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(400, text="Bad Request: Missing session ID"),
                    _FakeResponse(400, text="Bad Request: Missing session ID"),
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.LEGACY
        cached = era_cache.get(SERVER_URL)
        assert cached is not None and cached.era is MCPProtocolEra.LEGACY
        assert transport.methods() == [
            "server/discover",
            "server/discover",
            "initialize",
        ]

    @pytest.mark.parametrize("status", [401, 403, 407, 429])
    async def test_cached_modern_keeps_cache_on_credential_errors(self, status):
        cached = MCPServerProtocol(
            era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
        )
        era_cache.set(SERVER_URL, cached)
        transport = _Transport({"server/discover": [_FakeResponse(status)]})
        client = _client(transport, auth_token="stale")
        with pytest.raises(HTTPClientError) as exc:
            await client.initialize()
        assert exc.value.status_code == status
        assert transport.methods() == ["server/discover"]
        assert era_cache.get(SERVER_URL) == cached

    async def test_cached_legacy_reprobes_after_jsonrpc_error(self):
        """A 200 carrying a JSON-RPC error on initialize also re-detects."""
        era_cache.set(
            SERVER_URL,
            MCPServerProtocol(
                era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
            ),
        )
        transport = _Transport(
            {
                "initialize": [_FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND))],
                "server/discover": [_FakeResponse(200, _rpc_result(DISCOVER_RESULT))],
            }
        )
        client = _client(transport)
        await client.initialize()
        assert client.era is MCPProtocolEra.MODERN
        assert transport.methods() == ["initialize", "server/discover"]

    @pytest.mark.parametrize("status", [401, 403, 407, 429])
    async def test_cached_legacy_keeps_cache_on_credential_errors(self, status):
        cached = MCPServerProtocol(
            era=MCPProtocolEra.LEGACY, protocol_version=LEGACY_PROTOCOL_VERSION
        )
        era_cache.set(SERVER_URL, cached)
        transport = _Transport({"initialize": [_FakeResponse(status)]})
        client = _client(transport, auth_token="stale")
        with pytest.raises(HTTPClientError) as exc:
            await client.initialize()
        assert exc.value.status_code == status
        assert transport.methods() == ["initialize"]
        assert era_cache.get(SERVER_URL) == cached

    async def test_send_request_detects_era_lazily(self):
        transport = _Transport(
            {
                "server/discover": [_FakeResponse(200, _rpc_result(DISCOVER_RESULT))],
                "tools/list": [
                    _FakeResponse(200, _rpc_result({"tools": [], "ttlMs": 1}))
                ],
            }
        )
        client = _client(transport)
        tools = await client.list_tools()
        assert tools == []
        assert transport.methods() == ["server/discover", "tools/list"]


class TestHttpRetries:
    """Retries are bounded, and never applied to tool execution."""

    @pytest.mark.parametrize(
        ("method", "attempts"),
        [("tools/call", 1), ("tools/list", 3), ("server/discover", 3)],
    )
    async def test_retry_budget_per_method(self, method: str, attempts: int):
        client = MCPClient(SERVER_URL)
        with patch("backend.blocks.mcp.client.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(
                return_value=_FakeResponse(200, _rpc_result({}))
            )
            await client._post({"jsonrpc": "2.0", "id": 1, "method": method}, {})
        assert MockRequests.call_args.kwargs["retry_max_attempts"] == attempts


# ───────────────────────── legacy requests ─────────────────────────


class TestLegacyRequests:
    async def test_session_id_is_captured_echoed_and_released(self):
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND))
                ],
                "initialize": [
                    _FakeResponse(
                        200,
                        _rpc_result(LEGACY_INIT_RESULT),
                        headers={HEADER_SESSION_ID: "sess-1"},
                    )
                ],
                "tools/list": [_FakeResponse(200, _rpc_result({"tools": []}))],
            }
        )
        client = _client(transport)
        await client.initialize()
        await client.list_tools()

        _, list_headers = transport.sent[-1]
        assert list_headers[HEADER_SESSION_ID] == "sess-1"
        assert HEADER_PROTOCOL_VERSION not in list_headers

        deleted: list[dict[str, str]] = []

        class _FakeRequests:
            def __init__(self, *a, **kw):
                deleted.append(kw.get("extra_headers") or {})

            async def delete(self, url):
                return _FakeResponse(200)

        with patch("backend.blocks.mcp.client.Requests", _FakeRequests):
            await client.close()
        assert deleted[0][HEADER_SESSION_ID] == "sess-1"
        assert client._session_id is None

    async def test_legacy_jsonrpc_error_raises(self):
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND))
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
                "tools/call": [_FakeResponse(200, _rpc_error(-32602, "Unknown tool"))],
            }
        )
        client = _client(transport)
        await client.initialize()
        with pytest.raises(MCPClientError, match=r"\[-32602\]: Unknown tool"):
            await client.call_tool("nope", {})

    async def test_legacy_http_error_raises_client_error(self):
        transport = _Transport(
            {
                "server/discover": [
                    _FakeResponse(200, _rpc_error(ERROR_METHOD_NOT_FOUND))
                ],
                "initialize": [_FakeResponse(200, _rpc_result(LEGACY_INIT_RESULT))],
                "tools/list": [_FakeResponse(403, text="nope")],
            }
        )
        client = _client(transport)
        await client.initialize()
        with pytest.raises(HTTPClientError) as exc:
            await client.list_tools()
        assert exc.value.status_code == 403
        assert "Body: nope" in str(exc.value)


# ───────────────────────── modern requests ─────────────────────────


def _modern_client(
    extra: dict[str, list[_FakeResponse]]
) -> tuple[MCPClient, _Transport]:
    script = {"server/discover": [_FakeResponse(200, _rpc_result(DISCOVER_RESULT))]}
    script.update(extra)
    transport = _Transport(script)
    return _client(transport), transport


class TestModernRequests:
    async def test_tools_call_headers_and_meta(self):
        client, transport = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(
                        200,
                        _rpc_result(
                            {
                                "resultType": "complete",
                                "content": [{"type": "text", "text": "ok"}],
                            }
                        ),
                    )
                ]
            }
        )
        await client.initialize()
        result = await client.call_tool("get_weather", {"city": "Oslo"})

        assert result.content == [{"type": "text", "text": "ok"}]
        payload, headers = transport.sent[-1]
        assert headers[HEADER_METHOD] == "tools/call"
        assert headers[HEADER_NAME] == "get_weather"
        assert headers[HEADER_PROTOCOL_VERSION] == MODERN_PROTOCOL_VERSION
        assert payload["params"]["name"] == "get_weather"
        assert payload["params"]["arguments"] == {"city": "Oslo"}
        assert payload["params"]["_meta"][META_PROTOCOL_VERSION] == (
            MODERN_PROTOCOL_VERSION
        )

    async def test_tool_name_needing_encoding(self):
        client, transport = _modern_client(
            {"tools/call": [_FakeResponse(200, _rpc_result({"content": []}))]}
        )
        await client.initialize()
        await client.call_tool("näme with space", {})
        _, headers = transport.sent[-1]
        assert headers[HEADER_NAME].startswith("=?base64?")
        assert decode_header_value(headers[HEADER_NAME]) == "näme with space"

    async def test_absent_result_type_is_complete(self):
        client, _ = _modern_client(
            {"tools/call": [_FakeResponse(200, _rpc_result({"content": []}))]}
        )
        await client.initialize()
        result = await client.call_tool("t", {})
        assert not result.is_error

    async def test_unknown_result_type_rejected(self):
        client, _ = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(200, _rpc_result({"resultType": "something_new"}))
                ]
            }
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match="unrecognized resultType"):
            await client.call_tool("t", {})

    async def test_x_mcp_header_params_mirrored_from_list_tools(self):
        tool = {
            "name": "execute_sql",
            "description": "",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "x-mcp-header": "Region"},
                    "query": {"type": "string"},
                },
            },
        }
        client, transport = _modern_client(
            {
                "tools/list": [_FakeResponse(200, _rpc_result({"tools": [tool]}))],
                "tools/call": [_FakeResponse(200, _rpc_result({"content": []}))],
            }
        )
        await client.initialize()
        tools = await client.list_tools()
        assert [t.name for t in tools] == ["execute_sql"]
        await client.call_tool("execute_sql", {"region": "us-west1", "query": "x"})
        _, headers = transport.sent[-1]
        assert headers["Mcp-Param-Region"] == "us-west1"

    async def test_x_mcp_header_params_from_caller_supplied_schema(self):
        client, transport = _modern_client(
            {"tools/call": [_FakeResponse(200, _rpc_result({"content": []}))]}
        )
        await client.initialize()
        schema = {
            "type": "object",
            "properties": {"region": {"type": "string", "x-mcp-header": "Region"}},
        }
        await client.call_tool("execute_sql", {"region": "eu"}, input_schema=schema)
        _, headers = transport.sent[-1]
        assert headers["Mcp-Param-Region"] == "eu"

    async def test_header_mismatch_refreshes_schema_and_retries(self):
        tool = {
            "name": "execute_sql",
            "inputSchema": {
                "type": "object",
                "properties": {"region": {"type": "string", "x-mcp-header": "Region"}},
            },
        }
        client, transport = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(400, _rpc_error(ERROR_HEADER_MISMATCH)),
                    _FakeResponse(200, _rpc_result({"content": []})),
                ],
                "tools/list": [_FakeResponse(200, _rpc_result({"tools": [tool]}))],
            }
        )
        await client.initialize()
        await client.call_tool("execute_sql", {"region": "eu"})
        assert transport.methods() == [
            "server/discover",
            "tools/call",
            "tools/list",
            "tools/call",
        ]
        _, retry_headers = transport.sent[-1]
        assert retry_headers["Mcp-Param-Region"] == "eu"
        first_id = transport.sent[1][0]["id"]
        assert transport.sent[-1][0]["id"] != first_id

    async def test_invalid_header_annotation_drops_tool(self):
        tools = [
            {"name": "good", "inputSchema": {"type": "object"}},
            {
                "name": "bad",
                "inputSchema": {
                    "type": "object",
                    "properties": {"n": {"type": "number", "x-mcp-header": "N"}},
                },
            },
        ]
        client, _ = _modern_client(
            {"tools/list": [_FakeResponse(200, _rpc_result({"tools": tools}))]}
        )
        await client.initialize()
        listed = await client.list_tools()
        assert [t.name for t in listed] == ["good"]

    async def test_input_required_with_request_state_is_retried(self):
        client, transport = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(
                        200,
                        _rpc_result(
                            {"resultType": "input_required", "requestState": "s1"}
                        ),
                    ),
                    _FakeResponse(
                        200,
                        _rpc_result({"content": [{"type": "text", "text": "done"}]}),
                    ),
                ]
            }
        )
        await client.initialize()
        result = await client.call_tool("confirm", {"a": 1})
        assert result.content[0]["text"] == "done"
        retry_payload, _ = transport.sent[-1]
        assert retry_payload["params"]["requestState"] == "s1"
        assert retry_payload["params"]["arguments"] == {"a": 1}
        assert "inputResponses" not in retry_payload["params"]

    async def test_input_required_with_input_requests_is_an_error(self):
        client, _ = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(
                        200,
                        _rpc_result(
                            {
                                "resultType": "input_required",
                                "inputRequests": {
                                    "q": {"method": "elicitation/create", "params": {}}
                                },
                            }
                        ),
                    )
                ]
            }
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match="requires interactive input"):
            await client.call_tool("ask", {})

    async def test_input_required_loop_is_bounded(self):
        # One sticky reply: the server asks for another round-trip forever.
        client, transport = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(
                        200,
                        _rpc_result(
                            {"resultType": "input_required", "requestState": "again"}
                        ),
                    )
                ]
            }
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match="kept requesting more input"):
            await client.call_tool("loop", {})
        assert transport.methods().count("tools/call") == 4

    async def test_input_required_completes_on_last_permitted_round_trip(self):
        """The reply to the final retry is inspected, not discarded."""
        interim = _FakeResponse(
            200,
            _rpc_result({"resultType": "input_required", "requestState": "again"}),
        )
        final = _FakeResponse(
            200, _rpc_result({"content": [{"type": "text", "text": "DONE"}]})
        )
        client, transport = _modern_client({"tools/call": [interim] * 3 + [final]})
        await client.initialize()
        result = await client.call_tool("slow", {})
        assert not result.is_error
        assert result.content == [{"type": "text", "text": "DONE"}]
        assert transport.methods().count("tools/call") == 4

    async def test_call_tool_rejects_invalid_annotation_in_known_schema(self):
        client, transport = _modern_client(
            {"tools/call": [_FakeResponse(200, _rpc_result({"content": []}))]}
        )
        await client.initialize()
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array", "x-mcp-header": "Tags"}},
        }
        with pytest.raises(MCPClientError, match="invalid x-mcp-header"):
            await client.call_tool("tagged", {"tags": []}, input_schema=schema)
        assert "tools/call" not in transport.methods()

    async def test_alternating_version_and_header_errors_terminate(self):
        """Each retry kind happens at most once, whatever order they arrive in."""
        client, transport = _modern_client(
            {
                "tools/list": [_FakeResponse(200, _rpc_result({"tools": []}))],
                "tools/call": [
                    _FakeResponse(400, _rpc_error(ERROR_HEADER_MISMATCH)),
                    _FakeResponse(
                        400,
                        _rpc_error(
                            ERROR_UNSUPPORTED_PROTOCOL_VERSION,
                            data={"supported": [MODERN_PROTOCOL_VERSION]},
                        ),
                    ),
                    _FakeResponse(400, _rpc_error(ERROR_HEADER_MISMATCH)),
                ],
            }
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match=str(ERROR_HEADER_MISMATCH)):
            await client.call_tool("t", {})
        assert transport.methods().count("tools/call") == 3
        assert transport.methods().count("tools/list") == 1

    @pytest.mark.parametrize("status", [401, 403, 407, 429])
    async def test_credential_errors_with_envelope_keep_http_meaning(self, status):
        """A JSON-RPC envelope on a 401 must not turn it into MCPClientError."""
        client, _ = _modern_client(
            {"tools/call": [_FakeResponse(status, _rpc_error(-32001, "Unauthorized"))]}
        )
        await client.initialize()
        with pytest.raises(HTTPClientError) as exc:
            await client.call_tool("t", {})
        assert exc.value.status_code == status

    async def test_server_error_with_envelope_keeps_http_meaning(self):
        client, _ = _modern_client(
            {"tools/call": [_FakeResponse(503, _rpc_error(-32000, "down"))]}
        )
        await client.initialize()
        with pytest.raises(HTTPServerError):
            await client.call_tool("t", {})

    async def test_cached_modern_401_with_envelope_keeps_cache(self):
        cached = MCPServerProtocol(
            era=MCPProtocolEra.MODERN, protocol_version=MODERN_PROTOCOL_VERSION
        )
        era_cache.set(SERVER_URL, cached)
        transport = _Transport(
            {"server/discover": [_FakeResponse(401, _rpc_error(-32001, "expired"))]}
        )
        client = _client(transport, auth_token="expired")
        with pytest.raises(HTTPClientError) as exc:
            await client.initialize()
        assert exc.value.status_code == 401
        assert era_cache.get(SERVER_URL) == cached

    async def test_missing_capability_error(self):
        client, _ = _modern_client(
            {
                "tools/call": [
                    _FakeResponse(
                        400,
                        _rpc_error(
                            ERROR_MISSING_CLIENT_CAPABILITY,
                            data={"requiredCapabilities": ["elicitation"]},
                        ),
                    )
                ]
            }
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match="requires client capabilities"):
            await client.call_tool("t", {})

    async def test_method_not_found_404(self):
        client, _ = _modern_client(
            {"tools/list": [_FakeResponse(404, _rpc_error(ERROR_METHOD_NOT_FOUND))]}
        )
        await client.initialize()
        with pytest.raises(MCPClientError, match="does not support tools/list"):
            await client.list_tools()

    async def test_version_renegotiated_mid_session(self):
        client, transport = _modern_client(
            {
                "tools/list": [
                    _FakeResponse(
                        400,
                        _rpc_error(
                            ERROR_UNSUPPORTED_PROTOCOL_VERSION,
                            data={"supported": [MODERN_PROTOCOL_VERSION]},
                        ),
                    ),
                    _FakeResponse(200, _rpc_result({"tools": []})),
                ]
            }
        )
        await client.initialize()
        assert await client.list_tools() == []
        assert transport.methods() == ["server/discover", "tools/list", "tools/list"]

    async def test_sse_reply_is_parsed(self):
        sse = (
            "event: message\n"
            f"data: {json.dumps(_rpc_result({'tools': [], 'resultType': 'complete'}))}\n\n"
        )
        client, _ = _modern_client(
            {
                "tools/list": [
                    _FakeResponse(
                        200, text=sse, headers={"content-type": "text/event-stream"}
                    )
                ]
            }
        )
        await client.initialize()
        assert await client.list_tools() == []

    async def test_auth_error_propagates(self):
        client, _ = _modern_client({"tools/list": [_FakeResponse(401)]})
        await client.initialize()
        with pytest.raises(HTTPClientError) as exc:
            await client.list_tools()
        assert exc.value.status_code == 401

    async def test_close_is_noop_for_stateless_server(self):
        client, _ = _modern_client({})
        await client.initialize()
        with patch("backend.blocks.mcp.client.Requests") as requests_cls:
            await client.close()
        requests_cls.assert_not_called()
