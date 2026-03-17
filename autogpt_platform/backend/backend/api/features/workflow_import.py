"""API endpoint for importing external workflows via CoPilot."""

import logging
from typing import Any

import pydantic
from autogpt_libs.auth import requires_user
from fastapi import APIRouter, HTTPException, Security

from backend.copilot.workflow_import.converter import build_copilot_prompt
from backend.copilot.workflow_import.describers import describe_workflow
from backend.copilot.workflow_import.format_detector import (
    SourcePlatform,
    detect_format,
)
from backend.copilot.workflow_import.url_fetcher import fetch_n8n_template

logger = logging.getLogger(__name__)

router = APIRouter()


class ImportWorkflowRequest(pydantic.BaseModel):
    """Request body for importing an external workflow."""

    workflow_json: dict[str, Any] | None = None
    template_url: str | None = None

    @pydantic.model_validator(mode="after")
    def check_exactly_one_source(self) -> "ImportWorkflowRequest":
        has_json = self.workflow_json is not None
        has_url = self.template_url is not None
        if not has_json and not has_url:
            raise ValueError("Provide either 'workflow_json' or 'template_url'")
        if has_json and has_url:
            raise ValueError(
                "Provide only one of 'workflow_json' or 'template_url', not both"
            )
        return self


class ImportWorkflowResponse(pydantic.BaseModel):
    """Response from parsing an external workflow.

    Returns a CoPilot prompt that the frontend uses to redirect the user
    to CoPilot, where the agentic agent-generator handles the conversion.
    """

    copilot_prompt: str
    source_format: str
    source_name: str


@router.post(
    path="/workflow",
    summary="Import a workflow from another tool (n8n, Make.com, Zapier)",
    dependencies=[Security(requires_user)],
)
async def import_workflow(
    request: ImportWorkflowRequest,
) -> ImportWorkflowResponse:
    """Parse an external workflow and return a CoPilot prompt.

    Accepts either raw workflow JSON or a template URL (n8n only for now).
    The workflow is parsed and described, then a structured prompt is returned
    for CoPilot's agent-generator to handle the actual conversion.
    """
    # Step 1: Get the raw workflow JSON
    if request.template_url is not None:
        try:
            workflow_json = await fetch_n8n_template(request.template_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    else:
        workflow_json = request.workflow_json
        assert workflow_json is not None  # guaranteed by validator

    # Step 2: Detect format
    fmt = detect_format(workflow_json)
    logger.info(
        "Workflow format detection: result=%s, top-level keys=%s",
        fmt.value,
        list(workflow_json.keys())[:20],
    )
    if fmt == SourcePlatform.UNKNOWN:
        raise HTTPException(
            status_code=400,
            detail="Could not detect workflow format. Supported formats: "
            "n8n, Make.com, Zapier. Ensure you're uploading a valid "
            "workflow export file. "
            f"Found top-level keys: {list(workflow_json.keys())[:10]}",
        )

    # Step 3: Describe the workflow
    desc = describe_workflow(workflow_json, fmt)

    # Step 4: Build AutoPilot prompt (include raw JSON for full context)
    prompt = build_copilot_prompt(desc, workflow_json)

    return ImportWorkflowResponse(
        copilot_prompt=prompt,
        source_format=fmt.value,
        source_name=desc.name,
    )
