"""API endpoint for importing competitor workflows."""

import logging
from typing import Annotated, Any

import pydantic
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Security

from backend.copilot.workflow_import.converter import convert_competitor_workflow
from backend.copilot.workflow_import.describers import describe_workflow
from backend.copilot.workflow_import.format_detector import (
    CompetitorFormat,
    detect_format,
)
from backend.copilot.workflow_import.url_fetcher import fetch_n8n_template

logger = logging.getLogger(__name__)

router = APIRouter()


class ImportWorkflowRequest(pydantic.BaseModel):
    """Request body for importing a competitor workflow."""

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
    """Response from importing a competitor workflow."""

    graph: dict[str, Any]
    graph_id: str | None = None
    library_agent_id: str | None = None
    source_format: str
    source_name: str
    conversion_notes: list[str] = []


@router.post(
    path="/competitor-workflow",
    summary="Import a competitor workflow (n8n, Make.com, Zapier)",
    tags=["import"],
    dependencies=[Security(requires_user)],
)
async def import_competitor_workflow(
    request: ImportWorkflowRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> ImportWorkflowResponse:
    """Import a workflow from a competitor platform and convert it to an AutoGPT agent.

    Accepts either raw workflow JSON or a template URL (n8n only for now).
    The workflow is parsed, described, and then converted to an AutoGPT graph
    using LLM-powered block mapping.
    """
    # Step 1: Get the raw workflow JSON
    if request.template_url:
        try:
            workflow_json = await fetch_n8n_template(request.template_url)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        workflow_json = request.workflow_json
        assert workflow_json is not None  # guaranteed by validator

    # Step 2: Detect format
    fmt = detect_format(workflow_json)
    if fmt == CompetitorFormat.UNKNOWN:
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
        agent_json, conversion_notes = await convert_competitor_workflow(desc)
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
            logger.error(f"Failed to save imported agent: {e}", exc_info=True)
            conversion_notes.append(
                f"Save failed: {e}. You can try saving manually from the builder."
            )

    return ImportWorkflowResponse(
        graph=agent_json,
        graph_id=graph_id,
        library_agent_id=library_agent_id,
        source_format=fmt.value,
        source_name=desc.name,
        conversion_notes=conversion_notes,
    )
