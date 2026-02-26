"""Dummy Agent Generator for testing.

Returns mock responses matching the format expected from the external service.
Enable via AGENTGENERATOR_USE_DUMMY=true in settings.

WARNING: This is for testing only. Do not use in production.
"""

import asyncio
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Dummy decomposition result (instructions type)
DUMMY_DECOMPOSITION_RESULT: dict[str, Any] = {
    "type": "instructions",
    "steps": [
        {
            "description": "Get input from user",
            "action": "input",
            "block_name": "AgentInputBlock",
        },
        {
            "description": "Process the input",
            "action": "process",
            "block_name": "TextFormatterBlock",
        },
        {
            "description": "Return output to user",
            "action": "output",
            "block_name": "AgentOutputBlock",
        },
    ],
}

# Block IDs from backend/blocks/io.py
AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"
AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4"


def _generate_dummy_agent_json() -> dict[str, Any]:
    """Generate a minimal valid agent JSON for testing."""
    input_node_id = str(uuid.uuid4())
    output_node_id = str(uuid.uuid4())

    return {
        "id": str(uuid.uuid4()),
        "version": 1,
        "is_active": True,
        "name": "Dummy Test Agent",
        "description": "A dummy agent generated for testing purposes",
        "nodes": [
            {
                "id": input_node_id,
                "block_id": AGENT_INPUT_BLOCK_ID,
                "input_default": {
                    "name": "input",
                    "title": "Input",
                    "description": "Enter your input",
                    "placeholder_values": [],
                },
                "metadata": {"position": {"x": 0, "y": 0}},
            },
            {
                "id": output_node_id,
                "block_id": AGENT_OUTPUT_BLOCK_ID,
                "input_default": {
                    "name": "output",
                    "title": "Output",
                    "description": "Agent output",
                    "format": "{output}",
                },
                "metadata": {"position": {"x": 400, "y": 0}},
            },
        ],
        "links": [
            {
                "id": str(uuid.uuid4()),
                "source_id": input_node_id,
                "sink_id": output_node_id,
                "source_name": "result",
                "sink_name": "value",
                "is_static": False,
            },
        ],
    }


async def decompose_goal_dummy(
    description: str,
    context: str = "",
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return dummy decomposition result."""
    logger.info("Using dummy agent generator for decompose_goal")
    return DUMMY_DECOMPOSITION_RESULT.copy()


async def generate_agent_dummy(
    instructions: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
    operation_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return dummy agent synchronously (blocks for 30s, returns agent JSON).

    Note: operation_id and session_id parameters are ignored - we always use synchronous mode.
    """
    logger.info(
        "Using dummy agent generator (sync mode): returning agent JSON after 30s"
    )
    await asyncio.sleep(30)
    return _generate_dummy_agent_json()


async def generate_agent_patch_dummy(
    update_request: str,
    current_agent: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
    operation_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return dummy patched agent synchronously (blocks for 30s, returns patched agent JSON).

    Note: operation_id and session_id parameters are ignored - we always use synchronous mode.
    """
    logger.info(
        "Using dummy agent generator patch (sync mode): returning patched agent after 30s"
    )
    await asyncio.sleep(30)
    patched = current_agent.copy()
    patched["description"] = (
        f"{current_agent.get('description', '')} (updated: {update_request})"
    )
    return patched


async def customize_template_dummy(
    template_agent: dict[str, Any],
    modification_request: str,
    context: str = "",
) -> dict[str, Any]:
    """Return dummy customized template (returns template with updated description)."""
    logger.info("Using dummy agent generator for customize_template")
    customized = template_agent.copy()
    customized["description"] = (
        f"{template_agent.get('description', '')} (customized: {modification_request})"
    )
    return customized


async def get_blocks_dummy() -> list[dict[str, Any]]:
    """Return dummy blocks list."""
    logger.info("Using dummy agent generator for get_blocks")
    return [
        {"id": AGENT_INPUT_BLOCK_ID, "name": "AgentInputBlock"},
        {"id": AGENT_OUTPUT_BLOCK_ID, "name": "AgentOutputBlock"},
    ]


async def health_check_dummy() -> bool:
    """Always returns healthy for dummy service."""
    return True
