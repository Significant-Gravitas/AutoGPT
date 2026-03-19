"""CoPilot tool for importing external workflows (n8n, Make.com, Zapier)."""

import json
import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.workflow_import.describers import describe_workflow
from backend.copilot.workflow_import.format_detector import (
    SourcePlatform,
    detect_format,
    unwrap_zapier_envelope,
)
from backend.copilot.workflow_import.url_fetcher import fetch_n8n_template

from .base import BaseTool, ToolResponseBase
from .models import ErrorResponse, ResponseType

logger = logging.getLogger(__name__)

_MAX_RAW_JSON_BYTES = 500_000  # ~500 KB


def _cap_json(workflow_json: dict[str, Any]) -> str:
    raw = json.dumps(workflow_json, default=str)
    if len(raw) > _MAX_RAW_JSON_BYTES:
        return raw[:_MAX_RAW_JSON_BYTES] + "... [truncated]"
    return raw


class ImportWorkflowResponse(ToolResponseBase):
    """Response from importing an external workflow."""

    type: ResponseType = ResponseType.WORKFLOW_IMPORTED
    source_format: str
    source_name: str
    summary: str
    raw_workflow_json: str


class ImportWorkflowTool(BaseTool):
    """Tool for importing workflows from n8n, Make.com, or Zapier."""

    @property
    def name(self) -> str:
        return "import_workflow"

    @property
    def description(self) -> str:
        return (
            "Import a workflow from n8n, Make.com, or Zapier. "
            "Accepts an n8n template URL (e.g. https://n8n.io/workflows/1234) "
            "or raw workflow JSON. Returns the parsed workflow structure "
            "including all nodes, connections, and parameters. "
            "Use this when a user wants to import/convert an external workflow "
            "into an AutoGPT agent. After getting the result, use create_agent "
            "to build the equivalent AutoGPT agent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "URL of the workflow template to import. "
                        "Currently supports n8n URLs like "
                        "https://n8n.io/workflows/1234"
                    ),
                },
                "workflow_json": {
                    "type": "string",
                    "description": (
                        "Raw workflow JSON string. Use this when the user "
                        "pastes JSON directly instead of a URL."
                    ),
                },
            },
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        url = (kwargs.get("url") or "").strip()
        workflow_json_str = (kwargs.get("workflow_json") or "").strip()

        if not url and not workflow_json_str:
            return ErrorResponse(
                message="Provide either a URL or workflow JSON to import.",
                session_id=session_id,
            )

        # Step 1: Get the raw workflow JSON
        if url:
            try:
                workflow_json = await fetch_n8n_template(url)
            except ValueError as e:
                return ErrorResponse(message=str(e), session_id=session_id)
            except RuntimeError as e:
                return ErrorResponse(message=str(e), session_id=session_id)
        else:
            try:
                workflow_json = json.loads(workflow_json_str)
            except json.JSONDecodeError as e:
                return ErrorResponse(
                    message=f"Invalid JSON: {e}",
                    session_id=session_id,
                )
            if not isinstance(workflow_json, dict):
                return ErrorResponse(
                    message="Expected a JSON object, got "
                    f"{type(workflow_json).__name__}.",
                    session_id=session_id,
                )

        # Step 2: Detect format (also unwraps Zapier {"data": [...]} envelope)
        fmt = detect_format(workflow_json)
        if fmt == SourcePlatform.UNKNOWN:
            logger.info(
                "Unknown workflow format, top-level keys: %s",
                list(workflow_json.keys())[:10],
            )
            return ErrorResponse(
                message=(
                    "Could not detect workflow format. "
                    "Supported formats: n8n, Make.com, Zapier. "
                    "Ensure you're providing a valid workflow export."
                ),
                session_id=session_id,
            )

        # Unwrap Zapier envelope so the describer receives the single Zap object.
        if fmt == SourcePlatform.ZAPIER:
            workflow_json = unwrap_zapier_envelope(workflow_json) or workflow_json

        # Step 3: Describe the workflow
        try:
            desc = describe_workflow(workflow_json, fmt)
        except ValueError as e:
            return ErrorResponse(message=str(e), session_id=session_id)

        # Build a human-readable summary
        steps_lines: list[str] = []
        for step in desc.steps:
            line = f"  {step.order}. [{step.service}] {step.action}"
            if step.typed_connections:
                conn_parts = [
                    f"step {tc.target_step} ({tc.connection_type})"
                    for tc in step.typed_connections
                ]
                line += f" → {', '.join(conn_parts)}"
            steps_lines.append(line)
        summary = "\n".join(steps_lines)

        return ImportWorkflowResponse(
            message=(
                f"Successfully parsed {fmt.value} workflow '{desc.name}' "
                f"with {len(desc.steps)} functional steps. "
                "The full workflow JSON is included below — use it with "
                "create_agent to build the equivalent AutoGPT agent."
            ),
            source_format=fmt.value,
            source_name=desc.name,
            summary=summary,
            raw_workflow_json=_cap_json(workflow_json),
            session_id=session_id,
        )
