"""Tenant-locked bridge from Autopilot to Forwarding Digital's MCP server."""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Literal

from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.helpers import parse_mcp_content
from backend.copilot.model import ChatSession
from backend.copilot.partner_context import FORWARDING_DIGITAL_PARTNER_ID

from .base import BaseTool
from .models import ErrorResponse, MCPToolOutputResponse, ToolResponseBase

_MCP_URL_ENV = "FORWARDING_DIGITAL_MCP_URL"
_MCP_SECRET_ENV = "FORWARDING_DIGITAL_MCP_SHARED_SECRET"
_TOKEN_TTL_SECONDS = 60

Report = Literal["operations_summary", "arrivals", "exceptions"]

_REPORT_TO_MCP_TOOL: dict[Report, str] = {
    "operations_summary": "get_operations_summary",
    "arrivals": "list_arrivals",
    "exceptions": "list_exceptions",
}

_REPORT_CAPABILITY: dict[Report, str] = {
    "operations_summary": "reports.read",
    "arrivals": "jobs.read",
    "exceptions": "jobs.read",
}


class ForwardingDigitalTool(BaseTool):
    """Queries partner data without allowing the model to select a tenant."""

    @property
    def name(self) -> str:
        return "query_forwarding_digital"

    @property
    def description(self) -> str:
        return (
            "Fetch authoritative freight operations data for the current "
            "Forwarding Digital customer. Use for jobs, arrivals, exceptions, "
            "documents, lanes, revenue, profit, and operational reports. The "
            "tenant is bound by the authenticated session and is not an input."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "enum": list(_REPORT_TO_MCP_TOOL),
                    "description": "The operational data to retrieve.",
                }
            },
            "required": ["report"],
            "additionalProperties": False,
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return bool(os.getenv(_MCP_URL_ENV) and os.getenv(_MCP_SECRET_ENV))

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        report: str = "operations_summary",
        **kwargs: Any,
    ) -> ToolResponseBase:
        if (
            session.metadata.source_platform != FORWARDING_DIGITAL_PARTNER_ID
            or not session.metadata.external_account_id
            or not session.organization_id
            or not user_id
        ):
            return ErrorResponse(
                message="Forwarding Digital data is not available in this session.",
                error="partner_session_required",
                session_id=session.session_id,
            )

        if report not in _REPORT_TO_MCP_TOOL:
            return ErrorResponse(
                message=f"Unknown Forwarding Digital report: {report}",
                error="invalid_report",
                session_id=session.session_id,
            )

        required_capability = _REPORT_CAPABILITY[report]
        capabilities = session.metadata.external_capabilities
        if required_capability not in capabilities:
            return ErrorResponse(
                message="You do not have access to this Forwarding Digital report.",
                error="partner_capability_required",
                session_id=session.session_id,
            )

        server_url = os.getenv(_MCP_URL_ENV, "")
        secret = os.getenv(_MCP_SECRET_ENV, "")
        if not server_url or not secret:
            return ErrorResponse(
                message="The Forwarding Digital MCP integration is not configured.",
                error="partner_mcp_not_configured",
                session_id=session.session_id,
            )

        auth_token = _create_access_token(
            secret=secret,
            user_id=user_id,
            organization_id=session.organization_id,
            external_account_id=session.metadata.external_account_id,
            capabilities=capabilities,
        )
        tool_name = _REPORT_TO_MCP_TOOL[report]
        client = MCPClient(
            server_url,
            auth_token=auth_token,
            trusted_origins=[server_url],
        )
        try:
            await client.initialize()
            result = await client.call_tool(tool_name, {})
        finally:
            await client.close()

        if result.is_error:
            return ErrorResponse(
                message="Forwarding Digital returned an MCP tool error.",
                error="partner_mcp_tool_error",
                session_id=session.session_id,
            )

        return MCPToolOutputResponse(
            message=(
                "Verified tenant-bound Forwarding Digital data. Base the answer "
                "on this result and identify the returned customer by name."
            ),
            server_url="forwarding-digital",
            tool_name=tool_name,
            result=parse_mcp_content(result.content),
            session_id=session.session_id,
        )


def _create_access_token(
    *,
    secret: str,
    user_id: str,
    organization_id: str,
    external_account_id: str,
    capabilities: list[str],
    now: int | None = None,
) -> str:
    issued_at = int(time.time()) if now is None else now
    payload = {
        "version": 1,
        "partner_id": FORWARDING_DIGITAL_PARTNER_ID,
        "user_id": user_id,
        "organization_id": organization_id,
        "external_account_id": external_account_id,
        "capabilities": sorted(set(capabilities)),
        "exp": issued_at + _TOKEN_TTL_SECONDS,
    }
    encoded_payload = _base64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    )
    signature = hmac.new(
        secret.encode(), encoded_payload.encode(), hashlib.sha256
    ).digest()
    return f"{encoded_payload}.{_base64url(signature)}"


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()
