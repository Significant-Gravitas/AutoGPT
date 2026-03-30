"""
MCP Security (MCPS) — tool integrity hashing and optional message signing.

Tool hashing works with any MCP server and requires no extra dependencies.
MCPS message signing requires `pip install mcp-secure`.

Spec: https://datatracker.ietf.org/doc/draft-sharif-mcps-secure-mcp/
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class MCPToolIntegrityError(Exception):
    """Tool definition has changed since it was pinned."""


class MCPSignatureError(Exception):
    """MCPS message or tool signature is invalid."""


def compute_tool_hash(tool: dict[str, Any]) -> str:
    """SHA-256 fingerprint of a tool definition (name + description + inputSchema).

    Uses canonical sorted-key JSON so key ordering in server responses doesn't
    affect the result.
    """
    canonical = json.dumps(
        {
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", tool.get("input_schema", {})),
            "name": tool.get("name", ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_tool_hash(tool: dict[str, Any], expected_hash: str) -> None:
    """Raise MCPToolIntegrityError if the tool definition no longer matches the pinned hash.

    A mismatch means the tool was mutated after the hash was recorded, which may
    indicate tool poisoning or an unintended server-side change.
    """
    actual = compute_tool_hash(tool)
    if actual != expected_hash:
        raise MCPToolIntegrityError(
            f"Tool '{tool.get('name', '?')}' definition has changed since it was "
            f"pinned (expected {expected_hash[:12]}…, got {actual[:12]}…). "
            "This may indicate tool poisoning or a post-deployment mutation."
        )


@dataclass
class MCPSecurityContext:
    """ECDSA P-256 identity context for an AutoGPT Platform agent.

    Use MCPSecurityContext.generate() to create one.
    Requires the mcp-secure package (pip install mcp-secure).
    """

    private_key: str
    public_key: str
    passport_id: str
    _ta_public_key: str = field(default="", repr=False)

    @classmethod
    def generate(
        cls,
        agent_name: str = "AutoGPT-Platform",
        version: str = "1.0.0",
    ) -> "MCPSecurityContext":
        """Create a fresh context with a new ECDSA P-256 key pair and self-signed passport."""
        try:
            from mcp_secure import (  # type: ignore[import-untyped]
                create_passport,
                generate_key_pair,
                sign_passport,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The mcp-secure package is required for MCPS message signing. "
                "Install it with: pip install mcp-secure"
            ) from exc

        agent_keys = generate_key_pair()
        ta_keys = generate_key_pair()

        passport = create_passport(
            name=agent_name,
            version=version,
            public_key=agent_keys["public_key"],
            capabilities=["tools/list", "tools/call"],
        )
        signed = sign_passport(passport, ta_keys["private_key"])

        logger.debug(
            "Generated MCPS context for '%s' (passport_id=%s…)",
            agent_name,
            signed["passport_id"][:12],
        )
        return cls(
            private_key=agent_keys["private_key"],
            public_key=agent_keys["public_key"],
            passport_id=signed["passport_id"],
            _ta_public_key=ta_keys["public_key"],
        )

    def sign_outgoing(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Wrap a JSON-RPC payload in an MCPS signed envelope (nonce + timestamp + ECDSA sig)."""
        try:
            from mcp_secure import sign_message  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "mcp-secure is required for MCPS outgoing message signing"
            ) from exc

        return sign_message(payload, self.passport_id, self.private_key)

    def verify_incoming(
        self,
        response: dict[str, Any],
        server_public_key: str | None = None,
    ) -> dict[str, Any]:
        """Verify an MCPS-signed server response and return the inner payload.

        Plain JSON-RPC responses (no _mcps or signature key) pass through unchanged,
        keeping backward compatibility with non-MCPS servers.
        Raises MCPSignatureError when a signature is present but fails verification.
        """
        if "_mcps" not in response and "signature" not in response:
            return response

        try:
            from mcp_secure import verify_message  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "mcp-secure is required for MCPS response signature verification"
            ) from exc

        key = server_public_key or self.public_key
        result = verify_message(response, key)
        if not result.get("valid"):
            raise MCPSignatureError(
                f"MCPS response signature verification failed: "
                f"{result.get('reason', 'unknown reason')}"
            )
        return result.get("payload", response)

    def sign_tool_definition(self, tool: dict[str, Any]) -> str:
        """Return an ECDSA signature string for an MCP tool definition."""
        try:
            from mcp_secure import sign_tool  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "mcp-secure is required for MCPS tool definition signing"
            ) from exc

        return sign_tool(tool, self.private_key)

    def verify_tool_signature(self, tool: dict[str, Any], signature: str) -> None:
        """Raise MCPToolIntegrityError if the tool's ECDSA signature is invalid."""
        try:
            from mcp_secure import verify_tool  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "mcp-secure is required for MCPS tool signature verification"
            ) from exc

        if not verify_tool(tool, signature, self.public_key):
            raise MCPToolIntegrityError(
                f"MCPS signature invalid for tool '{tool.get('name', '?')}'"
            )
