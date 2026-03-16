"""Fetch competitor workflow templates by URL."""

import logging
import re
from typing import Any

from backend.util.request import Requests

logger = logging.getLogger(__name__)

# Patterns for extracting template IDs from n8n URLs
_N8N_WORKFLOW_URL_RE = re.compile(
    r"https?://(?:www\.)?n8n\.io/workflows/(\d+)", re.IGNORECASE
)
_N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows/{id}"


async def fetch_n8n_template(url: str) -> dict[str, Any]:
    """Fetch an n8n workflow template by its URL.

    Supports URLs like:
        - https://n8n.io/workflows/1234
        - https://n8n.io/workflows/1234-some-slug

    Args:
        url: The n8n template URL.

    Returns:
        The n8n workflow JSON.

    Raises:
        ValueError: If the URL is not a valid n8n template URL.
        RuntimeError: If the fetch fails.
    """
    match = _N8N_WORKFLOW_URL_RE.match(url.strip())
    if not match:
        raise ValueError(
            "Not a valid n8n workflow URL. Expected format: "
            "https://n8n.io/workflows/<id>"
        )

    template_id = match.group(1)
    api_url = _N8N_TEMPLATES_API.format(id=template_id)

    client = Requests(raise_for_status=True)
    try:
        response = await client.get(api_url)
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch n8n template {template_id}: {e}") from e

    # n8n API wraps the workflow in a `workflow` key
    workflow = data.get("workflow", data)
    if not isinstance(workflow, dict):
        raise RuntimeError(
            f"Unexpected response format from n8n API for template {template_id}"
        )

    # Preserve the workflow name from the template metadata
    if "name" not in workflow and "name" in data:
        workflow["name"] = data["name"]

    return workflow
