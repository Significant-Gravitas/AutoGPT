"""API endpoint for importing external workflows."""

import logging
from typing import Annotated, Any

import pydantic
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Security

from backend.copilot.workflow_import.converter import convert_workflow
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
    save: bool = True

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
    """Response from importing an external workflow."""

    graph: dict[str, Any]
    graph_id: str | None = None
    library_agent_id: str | None = None
    source_format: str
    source_name: str
    conversion_notes: list[str] = []


@router.post(
    path="/workflow",
    summary="Import a workflow from another tool (n8n, Make.com, Zapier)",
    dependencies=[Security(requires_user)],
)
async def import_workflow(
    request: ImportWorkflowRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> ImportWorkflowResponse:
    """Import a workflow from another automation platform and convert it to an
    AutoGPT agent.

    Accepts either raw workflow JSON or a template URL (n8n only for now).
    The workflow is parsed, described, and then converted to an AutoGPT graph
    using LLM-powered block mapping.
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
    if fmt == SourcePlatform.UNKNOWN:
        raise HTTPException(
            status_code=400,
            detail="Could not detect workflow format. Supported formats: "
            "n8n, Make.com, Zapier. Ensure you're uploading a valid "
            "workflow export file.",
        )

    # Step 3: Describe the workflow
    desc = describe_workflow(workflow_json, fmt)

    # Step 4: Convert to AutoGPT agent
    try:
        agent_json, conversion_notes = await convert_workflow(desc)
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Workflow conversion failed: {e}",
        ) from e

    # Step 5: Optionally save
    graph_id = None
    library_agent_id = None

    if request.save:
        from backend.copilot.tools.agent_generator.core import save_agent_to_library

        try:
            created_graph, library_agent = await save_agent_to_library(
                agent_json, user_id
            )
            graph_id = created_graph.id
            library_agent_id = library_agent.id
            conversion_notes.append(f"Agent saved as '{created_graph.name}'")
        except Exception as e:
            logger.error("Failed to save imported agent: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Workflow was converted but could not be saved. "
                "Please try again.",
            ) from e

    return ImportWorkflowResponse(
        graph=agent_json,
        graph_id=graph_id,
        library_agent_id=library_agent_id,
        source_format=fmt.value,
        source_name=desc.name,
        conversion_notes=conversion_notes,
    )
