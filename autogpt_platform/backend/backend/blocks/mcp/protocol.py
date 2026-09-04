"""
MCP protocol-version helpers shared by the client and its tests.

The client speaks two *eras* of the Model Context Protocol:

* **Legacy** (``2025-03-26`` … ``2025-11-25``): an ``initialize`` handshake
  establishes a session; the server may mint an ``Mcp-Session-Id`` that is
  echoed on every later request and released with HTTP ``DELETE``.
* **Modern** (``2026-07-28`` and later): stateless.  Every request carries its
  protocol version, client identity and capabilities in ``params._meta`` and
  mirrors the method / target name into ``Mcp-Method`` / ``Mcp-Name`` headers so
  intermediaries can route without parsing the body.

Reference: https://modelcontextprotocol.io/specification/2026-07-28
"""

from __future__ import annotations

import base64
import binascii
import re
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

LEGACY_PROTOCOL_VERSION = "2025-03-26"
MODERN_PROTOCOL_VERSION = "2026-07-28"
# Modern versions we can speak, newest first.  Extend when a newer revision is
# implemented; version negotiation picks the first entry the server also lists.
SUPPORTED_MODERN_VERSIONS: tuple[str, ...] = (MODERN_PROTOCOL_VERSION,)

CLIENT_INFO: dict[str, str] = {"name": "AutoGPT-Platform", "version": "1.0.0"}
# We do not implement elicitation, sampling or roots, so a compliant modern
# server must never send us ``inputRequests`` (MRTR).  Declaring nothing is
# what makes that guarantee hold.
CLIENT_CAPABILITIES: dict[str, Any] = {}

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

HEADER_PROTOCOL_VERSION = "MCP-Protocol-Version"
HEADER_METHOD = "Mcp-Method"
HEADER_NAME = "Mcp-Name"
HEADER_PARAM_PREFIX = "Mcp-Param-"
HEADER_SESSION_ID = "Mcp-Session-Id"

# JSON-RPC error codes reserved by the 2026-07-28 specification.  Receiving any
# of these proves the peer is a modern server, even on an HTTP 400.
ERROR_HEADER_MISMATCH = -32020
ERROR_MISSING_CLIENT_CAPABILITY = -32021
ERROR_UNSUPPORTED_PROTOCOL_VERSION = -32022
MODERN_ERROR_CODES = frozenset(
    {
        ERROR_HEADER_MISMATCH,
        ERROR_MISSING_CLIENT_CAPABILITY,
        ERROR_UNSUPPORTED_PROTOCOL_VERSION,
    }
)
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602

RESULT_TYPE_COMPLETE = "complete"
RESULT_TYPE_INPUT_REQUIRED = "input_required"

# Methods whose ``params.name`` / ``params.uri`` must be mirrored into Mcp-Name.
_NAME_SOURCE_FIELD: dict[str, str] = {
    "tools/call": "name",
    "prompts/get": "name",
    "resources/read": "uri",
}


class MCPProtocolEra(str, Enum):
    LEGACY = "legacy"
    MODERN = "modern"


# ────────────────────────────── era cache ──────────────────────────────


class MCPServerProtocol(BaseModel):
    """What we last learned about a server: its era and negotiated version."""

    model_config = ConfigDict(frozen=True)

    era: MCPProtocolEra
    protocol_version: str


class MCPServerEraCache:
    """Remember which era each MCP server speaks.

    The spec asks clients to cache era detection per origin so a legacy server
    does not pay for a failed modern probe on every connection.  Entries
    expire so a server that upgrades is picked up without a restart, and
    callers drop an entry as soon as its assumption fails.

    In-memory and per process on purpose: the cost of a miss is one extra
    round-trip, which is not worth a cross-process dependency.  Keys are
    user-supplied URLs, so the cache is bounded: once full, the least
    recently used entry is evicted.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_entries: int = 1024):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[MCPServerProtocol, float]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, server_url: str) -> MCPServerProtocol | None:
        with self._lock:
            entry = self._entries.get(server_url)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at <= time.monotonic():
                del self._entries[server_url]
                return None
            self._entries.move_to_end(server_url)
            return value

    def set(self, server_url: str, value: MCPServerProtocol) -> None:
        with self._lock:
            self._entries[server_url] = (value, time.monotonic() + self.ttl_seconds)
            self._entries.move_to_end(server_url)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def forget(self, server_url: str) -> None:
        with self._lock:
            self._entries.pop(server_url, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


era_cache = MCPServerEraCache()


# ─────────────────────────── request metadata ───────────────────────────


def build_request_meta(protocol_version: str) -> dict[str, Any]:
    """The ``_meta`` block every modern request must carry."""
    return {
        META_PROTOCOL_VERSION: protocol_version,
        META_CLIENT_INFO: dict(CLIENT_INFO),
        META_CLIENT_CAPABILITIES: dict(CLIENT_CAPABILITIES),
    }


def negotiate_version(supported_by_server: Any) -> str | None:
    """Pick the newest version both sides speak, or ``None`` if there is none."""
    if not isinstance(supported_by_server, list):
        return None
    offered = {v for v in supported_by_server if isinstance(v, str)}
    for candidate in SUPPORTED_MODERN_VERSIONS:
        if candidate in offered:
            return candidate
    return None


def is_modern_jsonrpc_error(body: Any) -> bool:
    """True if *body* is a JSON-RPC error whose code the modern spec reserves."""
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    return isinstance(error, dict) and error.get("code") in MODERN_ERROR_CODES


def jsonrpc_error_code(body: Any) -> int | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, int) else None


# ─────────────────────────── header encoding ───────────────────────────

_BASE64_SENTINEL_PREFIX = "=?base64?"
_BASE64_SENTINEL_SUFFIX = "?="


def encode_header_value(value: str) -> str:
    """Encode a body value for an ``Mcp-Name`` / ``Mcp-Param-*`` header.

    Plain visible-ASCII values travel as-is.  Anything else (non-ASCII,
    control characters, leading/trailing whitespace, or a value that already
    looks like the sentinel) is Base64-wrapped as ``=?base64?…?=`` so it
    cannot be misread or used for header injection.
    """
    plain = all(0x20 <= ord(ch) <= 0x7E for ch in value)
    padded = value != value.strip()
    looks_encoded = value.startswith(_BASE64_SENTINEL_PREFIX) and value.endswith(
        _BASE64_SENTINEL_SUFFIX
    )
    if plain and not padded and not looks_encoded:
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"{_BASE64_SENTINEL_PREFIX}{encoded}{_BASE64_SENTINEL_SUFFIX}"


def decode_header_value(value: str) -> str:
    """Inverse of :func:`encode_header_value` (used by the test server)."""
    if value.startswith(_BASE64_SENTINEL_PREFIX) and value.endswith(
        _BASE64_SENTINEL_SUFFIX
    ):
        inner = value[len(_BASE64_SENTINEL_PREFIX) : -len(_BASE64_SENTINEL_SUFFIX)]
        try:
            return base64.b64decode(inner, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return value
    return value


def name_header_for(method: str, params: dict[str, Any] | None) -> str | None:
    """Value for ``Mcp-Name`` on *method*, or ``None`` when it is not required."""
    field = _NAME_SOURCE_FIELD.get(method)
    if field is None or not params:
        return None
    value = params.get(field)
    if not isinstance(value, str):
        return None
    return encode_header_value(value)


# ─────────────────────────── x-mcp-header ───────────────────────────

# RFC 7230 ``tchar``; matched with ``fullmatch`` so a trailing newline fails.
_TCHAR_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_PARAM_TYPES = {"string", "integer", "boolean"}
_SAFE_INT_MIN = -(2**53) + 1
_SAFE_INT_MAX = 2**53 - 1
_HEADER_PARAM_PROPERTY = "x-mcp-header"
_SCHEMA_ARRAY_OR_COMPOSITION_KEYS = (
    "items",
    "prefixItems",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "$ref",
)


class HeaderParam(BaseModel):
    """A tool parameter the server asked us to mirror into a header."""

    model_config = ConfigDict(frozen=True)

    path: tuple[str, ...]
    header_name: str
    json_type: str


class InvalidHeaderAnnotation(ValueError):
    """A tool's ``x-mcp-header`` annotations violate the spec."""


def collect_header_params(input_schema: Any) -> list[HeaderParam]:
    """Find every ``x-mcp-header`` annotation reachable through ``properties``.

    Raises :class:`InvalidHeaderAnnotation` if any annotation breaks the spec's
    constraints; callers must then drop the whole tool from ``tools/list``.
    """
    found: list[HeaderParam] = []
    seen_names: set[str] = set()

    def walk(schema: Any, path: tuple[str, ...]) -> None:
        if not isinstance(schema, dict):
            return
        # Presence, not truthiness: an explicit ``null`` is a malformed
        # annotation and must invalidate the tool, not be ignored.
        if _HEADER_PARAM_PROPERTY in schema:
            annotation = schema[_HEADER_PARAM_PROPERTY]
            if not path:
                raise InvalidHeaderAnnotation(
                    "x-mcp-header is not allowed on the root schema"
                )
            _validate_annotation(annotation, schema, path, seen_names)
            found.append(
                HeaderParam(path=path, header_name=annotation, json_type=schema["type"])
            )
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, sub in properties.items():
                walk(sub, path + (key,))

    walk(input_schema, ())
    _reject_unreachable_annotations(input_schema)
    return found


def _validate_annotation(
    annotation: Any,
    schema: dict[str, Any],
    path: tuple[str, ...],
    seen_names: set[str],
) -> None:
    where = ".".join(path)
    if not isinstance(annotation, str) or not annotation:
        raise InvalidHeaderAnnotation(
            f"{where}: x-mcp-header must be a non-empty string"
        )
    if not _TCHAR_RE.fullmatch(annotation):
        raise InvalidHeaderAnnotation(
            f"{where}: x-mcp-header {annotation!r} is not a valid HTTP field name"
        )
    if schema.get("type") not in _HEADER_PARAM_TYPES:
        raise InvalidHeaderAnnotation(
            f"{where}: x-mcp-header is only allowed on string/integer/boolean "
            f"parameters, got {schema.get('type')!r}"
        )
    folded = annotation.casefold()
    if folded in seen_names:
        raise InvalidHeaderAnnotation(
            f"{where}: duplicate x-mcp-header name {annotation!r}"
        )
    seen_names.add(folded)


def _reject_unreachable_annotations(schema: Any) -> None:
    """An annotation behind ``items``/``oneOf``/``$ref``/… invalidates the tool."""

    def walk(node: Any, reachable: bool) -> None:
        if isinstance(node, dict):
            if _HEADER_PARAM_PROPERTY in node and not reachable:
                raise InvalidHeaderAnnotation(
                    "x-mcp-header must be statically reachable through "
                    "`properties` only"
                )
            for key, sub in node.items():
                if key == "properties" and isinstance(sub, dict):
                    for child in sub.values():
                        walk(child, reachable)
                elif key in _SCHEMA_ARRAY_OR_COMPOSITION_KEYS:
                    walk(sub, False)
                elif isinstance(sub, (dict, list)):
                    walk(sub, False)
        elif isinstance(node, list):
            for item in node:
                walk(item, False)

    walk(schema, True)


def extract_header_params(
    header_params: list[HeaderParam], arguments: dict[str, Any] | None
) -> dict[str, str]:
    """Build ``Mcp-Param-*`` headers from call *arguments*.

    Missing or ``null`` values are simply omitted, as the spec requires.
    """
    headers: dict[str, str] = {}
    for param in header_params:
        value: Any = arguments or {}
        for key in param.path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        if param.json_type == "boolean":
            if not isinstance(value, bool):
                continue
            text = "true" if value else "false"
        elif param.json_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if not _SAFE_INT_MIN <= value <= _SAFE_INT_MAX:
                continue
            text = str(value)
        else:
            if not isinstance(value, str):
                continue
            text = value
        headers[f"{HEADER_PARAM_PREFIX}{param.header_name}"] = encode_header_value(text)
    return headers
