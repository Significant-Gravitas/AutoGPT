"""
Module for generating AI-based activity status for graph executions.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import SecretStr

from backend.blocks.llm import LlmModel, llm_call
from backend.data.block import get_block
from backend.data.execution import NodeExecutionResult
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerAsyncClient

logger = logging.getLogger(__name__)


async def generate_activity_status_for_execution(
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    execution_stats: GraphExecutionStats,
    db_client: "DatabaseManagerAsyncClient",
    node_execution_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> str:
    """
    Generate an AI-based activity status summary for a graph execution.

    This function handles all the data collection and AI generation logic,
    keeping the manager integration simple.

    Args:
        graph_exec_id: The graph execution ID
        graph_id: The graph ID
        graph_version: The graph version
        execution_stats: Execution statistics
        db_client: Database client for fetching data
        node_execution_loop: Event loop for async operations

    Returns:
        AI-generated activity status string
    """
    try:
        settings = Settings()

        # Check if we have OpenAI API key
        if not settings.secrets.openai_api_key:
            logger.debug(
                "OpenAI API key not configured, skipping activity status generation"
            )
            return "Activity status generation disabled (no API key)"

        # Get all node executions for this graph execution
        node_executions = await db_client.get_node_executions(
            graph_exec_id, include_exec_data=True
        )

        # Get graph metadata and full graph structure for name, description, and links
        graph_metadata = await db_client.get_graph_metadata(graph_id, graph_version)
        graph = await db_client.get_graph(graph_id, graph_version)

        graph_name = graph_metadata.name if graph_metadata else f"Graph {graph_id}"
        graph_description = graph_metadata.description if graph_metadata else ""
        graph_links = graph.links if graph else []

        # Build execution data summary
        execution_data = _build_execution_summary(
            node_executions, execution_stats, graph_name, graph_description, graph_links
        )

        # Prepare prompt for AI
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that analyzes agent execution data and provides clear, "
                    "concise summaries of what the agent did during its execution. "
                    "Focus on the main actions performed, the outcome, and any errors encountered. "
                    "Keep the summary to 1-2 sentences maximum. Be specific about what was accomplished.\n\n"
                    "IMPORTANT: Understand that errors in agent execution have different levels of impact:\n"
                    "- Some errors are EXPECTED or MINOR (e.g., validation failures, optional operations, "
                    "retryable network issues) and don't prevent the agent from continuing or completing its task\n"
                    "- Some errors are CRITICAL (e.g., authentication failures, missing required data, "
                    "system crashes) and cause the agent execution to stop or fail completely\n"
                    "When summarizing, distinguish between these types of errors and focus more on critical "
                    "errors that actually impacted the agent's ability to complete its intended task."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze this agent execution data for '{graph_name}' and provide a brief summary "
                    f"of what the agent did and how it ended:\n\n"
                    f"{json.dumps(execution_data, indent=2)}\n\n"
                    "Provide a concise 1-2 sentence summary focusing on:\n"
                    "1. What the agent attempted to do\n"
                    "2. Whether it succeeded or failed overall\n"
                    "3. Key errors that actually impacted the execution (distinguish between minor/expected "
                    "errors vs critical errors that stopped or significantly affected the agent's performance)"
                ),
            },
        ]

        # Create credentials for LLM call
        credentials = APIKeyCredentials(
            id="openai",
            provider="openai",
            api_key=SecretStr(settings.secrets.openai_api_key),
            title="System OpenAI",
        )

        # Make LLM call - use the event loop if provided
        if node_execution_loop:
            activity_status = await _call_llm_async(
                credentials, prompt, node_execution_loop
            )
        else:
            activity_status = await _call_llm_direct(credentials, prompt)

        logger.info(f"Generated activity status for {graph_exec_id}: {activity_status}")
        return activity_status

    except Exception as e:
        logger.error(
            f"Failed to generate activity status for execution {graph_exec_id}: {str(e)}"
        )
        return f"Failed to generate activity summary: {str(e)}"


def _build_execution_summary(
    node_executions: List[NodeExecutionResult],
    execution_stats: GraphExecutionStats,
    graph_name: str,
    graph_description: str,
    graph_links: List[Any],
) -> Dict[str, Any]:
    """Build a structured summary of execution data for AI analysis."""

    nodes = []
    node_execution_counts = {}  # Track execution count per node
    node_error_counts = {}  # Track error count per node
    input_output_data = {}
    errors = []
    node_map = {}  # Map node_id to node data for easy lookup

    # Process node executions
    for node_exec in node_executions:
        block = get_block(node_exec.block_id)
        if not block:
            logger.warning(
                f"Block {node_exec.block_id} not found for node {node_exec.node_id}"
            )
            continue

        # Track execution counts per node
        if node_exec.node_id not in node_execution_counts:
            node_execution_counts[node_exec.node_id] = 0
        node_execution_counts[node_exec.node_id] += 1

        # Track errors per node
        if (
            node_exec.status.value == "FAILED"
            if hasattr(node_exec.status, "value")
            else str(node_exec.status) == "FAILED"
        ):
            if node_exec.node_id not in node_error_counts:
                node_error_counts[node_exec.node_id] = 0
            node_error_counts[node_exec.node_id] += 1

            # Extract actual error message from output_data
            error_message = "Unknown error"
            if node_exec.output_data and isinstance(node_exec.output_data, dict):
                # Check if error is in output_data
                if "error" in node_exec.output_data:
                    error_output = node_exec.output_data["error"]
                    if isinstance(error_output, list) and error_output:
                        error_message = str(error_output[0])
                    else:
                        error_message = str(error_output)

            errors.append(
                {
                    "node_id": node_exec.node_id,
                    "block_name": block.name,
                    "error": error_message,
                    "execution_id": node_exec.node_exec_id,  # Include exec ID for debugging
                }
            )

        # Build node data (only add unique nodes)
        if node_exec.node_id not in node_map:
            node_data = {
                "node_id": node_exec.node_id,
                "block_id": node_exec.block_id,
                "block_name": block.name,
                "block_description": block.description or "",
            }
            nodes.append(node_data)
            node_map[node_exec.node_id] = node_data

        # Store input/output data for special blocks (input/output blocks)
        if block.name in ["AgentInputBlock", "AgentOutputBlock", "UserInputBlock"]:
            if node_exec.input_data:
                input_output_data[f"{node_exec.node_id}_inputs"] = dict(
                    node_exec.input_data
                )
            if node_exec.output_data:
                input_output_data[f"{node_exec.node_id}_outputs"] = dict(
                    node_exec.output_data
                )

    # Add execution and error counts to node data
    for node in nodes:
        node_id = node["node_id"]
        node["execution_count"] = node_execution_counts.get(node_id, 0)
        node["error_count"] = node_error_counts.get(node_id, 0)

    # Build node relations from graph links
    node_relations = []
    for link in graph_links:
        # Include link details with source and sink information
        relation = {
            "source_node_id": link.source_id,
            "sink_node_id": link.sink_id,
            "source_name": link.source_name,
            "sink_name": link.sink_name,
            "is_static": link.is_static if hasattr(link, "is_static") else False,
        }

        # Add block names if nodes exist in our map
        if link.source_id in node_map:
            relation["source_block_name"] = node_map[link.source_id]["block_name"]
        if link.sink_id in node_map:
            relation["sink_block_name"] = node_map[link.sink_id]["block_name"]

        node_relations.append(relation)

    # Build overall summary
    return {
        "graph_info": {"name": graph_name, "description": graph_description},
        "nodes": nodes,
        "node_relations": node_relations,
        "input_output_data": input_output_data,
        "errors": errors,
        "overall_status": {
            "total_nodes_in_graph": len(nodes),
            "total_executions": execution_stats.node_count,
            "total_errors": execution_stats.node_error_count,
            "execution_time_seconds": execution_stats.walltime,
            "has_errors": bool(
                execution_stats.error or execution_stats.node_error_count > 0
            ),
        },
    }


async def _call_llm_async(
    credentials: APIKeyCredentials,
    prompt: List[Dict[str, str]],
    event_loop: asyncio.AbstractEventLoop,
) -> str:
    """Call LLM using the provided event loop."""

    future = asyncio.run_coroutine_threadsafe(
        _call_llm_direct(credentials, prompt), event_loop
    )
    return future.result(timeout=10.0)


async def _call_llm_direct(
    credentials: APIKeyCredentials, prompt: List[Dict[str, str]]
) -> str:
    """Make direct LLM call."""

    response = await llm_call(
        credentials=credentials,
        llm_model=LlmModel.GPT4O_MINI,
        prompt=prompt,
        json_format=False,
        max_tokens=150,
        compress_prompt_to_fit=True,
    )

    if response and response.response:
        return response.response.strip()
    else:
        return "Unable to generate activity summary"
