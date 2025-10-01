"""
Module for generating AI-based activity status for graph executions.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from pydantic import SecretStr

from backend.blocks.llm import LlmModel, llm_call
from backend.data.block import get_block
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.retry import func_retry
from backend.util.settings import Settings
from backend.util.truncate import truncate

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerAsyncClient

logger = logging.getLogger(__name__)


class ErrorInfo(TypedDict):
    """Type definition for error information."""

    error: str
    execution_id: str
    timestamp: str


class InputOutputInfo(TypedDict):
    """Type definition for input/output information."""

    execution_id: str
    output_data: dict[str, Any]  # Used for both input and output data
    timestamp: str


class NodeInfo(TypedDict):
    """Type definition for node information."""

    node_id: str
    block_id: str
    block_name: str
    block_description: str
    execution_count: int
    error_count: int
    recent_errors: list[ErrorInfo]
    recent_outputs: list[InputOutputInfo]
    recent_inputs: list[InputOutputInfo]


class NodeRelation(TypedDict):
    """Type definition for node relation information."""

    source_node_id: str
    sink_node_id: str
    source_name: str
    sink_name: str
    is_static: bool
    source_block_name: NotRequired[str]  # Optional, only set if block exists
    sink_block_name: NotRequired[str]  # Optional, only set if block exists


def _truncate_uuid(uuid_str: str) -> str:
    """Truncate UUID to first segment to reduce payload size."""
    if not uuid_str:
        return uuid_str
    return uuid_str.split("-")[0] if "-" in uuid_str else uuid_str[:8]


async def generate_activity_status_for_execution(
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    execution_stats: GraphExecutionStats,
    db_client: "DatabaseManagerAsyncClient",
    user_id: str,
    execution_status: ExecutionStatus | None = None,
) -> str | None:
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
        user_id: User ID for LaunchDarkly feature flag evaluation
        execution_status: The overall execution status (COMPLETED, FAILED, TERMINATED)

    Returns:
        AI-generated activity status string, or None if feature is disabled
    """
    # Check LaunchDarkly feature flag for AI activity status generation with full context support
    if not await is_feature_enabled(Flag.AI_ACTIVITY_STATUS, user_id):
        logger.debug("AI activity status generation is disabled via LaunchDarkly")
        return None

    # Check if we have OpenAI API key
    try:
        settings = Settings()
        if not settings.secrets.openai_internal_api_key:
            logger.debug(
                "OpenAI API key not configured, skipping activity status generation"
            )
            return None

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
            node_executions,
            execution_stats,
            graph_name,
            graph_description,
            graph_links,
            execution_status,
        )

        # Prepare prompt for AI
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant summarizing what you just did for a user in simple, friendly language. "
                    "Write from the user's perspective about what they accomplished, NOT about technical execution details. "
                    "Focus on the ACTUAL TASK the user wanted done, not the internal workflow steps. "
                    "Avoid technical terms like 'workflow', 'execution', 'components', 'nodes', 'processing', etc. "
                    "Keep it to 3 sentences maximum. Be conversational and human-friendly.\n\n"
                    "IMPORTANT: Be HONEST about what actually happened:\n"
                    "- If the input was invalid/nonsensical, say so directly\n"
                    "- If the task failed, explain what went wrong in simple terms\n"
                    "- If errors occurred, focus on what the user needs to know\n"
                    "- Only claim success if the task was genuinely completed\n"
                    "- Don't sugar-coat failures or present them as helpful feedback\n\n"
                    "Understanding Errors:\n"
                    "- Node errors: Individual steps may fail but the overall task might still complete (e.g., one data source fails but others work)\n"
                    "- Graph error (in overall_status.graph_error): This means the entire execution failed and nothing was accomplished\n"
                    "- Even if execution shows 'completed', check if critical nodes failed that would prevent the desired outcome\n"
                    "- Focus on the end result the user wanted, not whether technical steps completed"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A user ran '{graph_name}' to accomplish something. Based on this execution data, "
                    f"write what they achieved in simple, user-friendly terms:\n\n"
                    f"{json.dumps(execution_data, indent=2)}\n\n"
                    "CRITICAL: Check overall_status.graph_error FIRST - if present, the entire execution failed.\n"
                    "Then check individual node errors to understand partial failures.\n\n"
                    "Write 1-3 sentences about what the user accomplished, such as:\n"
                    "- 'I analyzed your resume and provided detailed feedback for the IT industry.'\n"
                    "- 'I couldn't analyze your resume because the input was just nonsensical text.'\n"
                    "- 'I failed to complete the task due to missing API access.'\n"
                    "- 'I extracted key information from your documents and organized it into a summary.'\n"
                    "- 'The task failed to run due to system configuration issues.'\n\n"
                    "Focus on what ACTUALLY happened, not what was attempted."
                ),
            },
        ]

        # Log the prompt for debugging purposes
        logger.debug(
            f"Sending prompt to LLM for graph execution {graph_exec_id}: {json.dumps(prompt, indent=2)}"
        )

        # Create credentials for LLM call
        credentials = APIKeyCredentials(
            id="openai",
            provider="openai",
            api_key=SecretStr(settings.secrets.openai_internal_api_key),
            title="System OpenAI",
        )

        # Make LLM call using current event loop
        activity_status = await _call_llm_direct(credentials, prompt)

        logger.debug(
            f"Generated activity status for {graph_exec_id}: {activity_status}"
        )
        return activity_status

    except Exception as e:
        logger.error(
            f"Failed to generate activity status for execution {graph_exec_id}: {str(e)}"
        )
        return None


def _build_execution_summary(
    node_executions: list[NodeExecutionResult],
    execution_stats: GraphExecutionStats,
    graph_name: str,
    graph_description: str,
    graph_links: list[Any],
    execution_status: ExecutionStatus | None = None,
) -> dict[str, Any]:
    """Build a structured summary of execution data for AI analysis."""

    nodes: list[NodeInfo] = []
    node_execution_counts: dict[str, int] = {}
    node_error_counts: dict[str, int] = {}
    node_errors: dict[str, list[ErrorInfo]] = {}
    node_outputs: dict[str, list[InputOutputInfo]] = {}
    node_inputs: dict[str, list[InputOutputInfo]] = {}
    input_output_data: dict[str, Any] = {}
    node_map: dict[str, NodeInfo] = {}

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

        # Track errors per node and group them
        if node_exec.status == ExecutionStatus.FAILED:
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

            # Group errors by node_id
            if node_exec.node_id not in node_errors:
                node_errors[node_exec.node_id] = []

            node_errors[node_exec.node_id].append(
                {
                    "error": error_message,
                    "execution_id": _truncate_uuid(node_exec.node_exec_id),
                    "timestamp": node_exec.add_time.isoformat(),
                }
            )

        # Collect output samples for each node (latest executions)
        if node_exec.output_data:
            if node_exec.node_id not in node_outputs:
                node_outputs[node_exec.node_id] = []

            # Truncate output data to 100 chars to save space
            truncated_output = truncate(node_exec.output_data, 100)

            node_outputs[node_exec.node_id].append(
                {
                    "execution_id": _truncate_uuid(node_exec.node_exec_id),
                    "output_data": truncated_output,
                    "timestamp": node_exec.add_time.isoformat(),
                }
            )

        # Collect input samples for each node (latest executions)
        if node_exec.input_data:
            if node_exec.node_id not in node_inputs:
                node_inputs[node_exec.node_id] = []

            # Truncate input data to 100 chars to save space
            truncated_input = truncate(node_exec.input_data, 100)

            node_inputs[node_exec.node_id].append(
                {
                    "execution_id": _truncate_uuid(node_exec.node_exec_id),
                    "output_data": truncated_input,  # Reuse field name for consistency
                    "timestamp": node_exec.add_time.isoformat(),
                }
            )

        # Build node data (only add unique nodes)
        if node_exec.node_id not in node_map:
            node_data: NodeInfo = {
                "node_id": _truncate_uuid(node_exec.node_id),
                "block_id": _truncate_uuid(node_exec.block_id),
                "block_name": block.name,
                "block_description": block.description or "",
                "execution_count": 0,  # Will be set later
                "error_count": 0,  # Will be set later
                "recent_errors": [],  # Will be set later
                "recent_outputs": [],  # Will be set later
                "recent_inputs": [],  # Will be set later
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

    # Add execution and error counts to node data, plus limited errors and output samples
    for node in nodes:
        # Use original node_id for lookups (before truncation)
        original_node_id = None
        for orig_id, node_data in node_map.items():
            if node_data == node:
                original_node_id = orig_id
                break

        if original_node_id:
            node["execution_count"] = node_execution_counts.get(original_node_id, 0)
            node["error_count"] = node_error_counts.get(original_node_id, 0)

            # Add limited errors for this node (latest 10 or first 5 + last 5)
            if original_node_id in node_errors:
                node_error_list = node_errors[original_node_id]
                if len(node_error_list) <= 10:
                    node["recent_errors"] = node_error_list
                else:
                    # First 5 + last 5 if more than 10 errors
                    node["recent_errors"] = node_error_list[:5] + node_error_list[-5:]

            # Add latest output samples (latest 3)
            if original_node_id in node_outputs:
                node_output_list = node_outputs[original_node_id]
                # Sort by timestamp if available, otherwise take last 3
                if node_output_list and node_output_list[0].get("timestamp"):
                    node_output_list.sort(
                        key=lambda x: x.get("timestamp", ""), reverse=True
                    )
                node["recent_outputs"] = node_output_list[:3]

            # Add latest input samples (latest 3)
            if original_node_id in node_inputs:
                node_input_list = node_inputs[original_node_id]
                # Sort by timestamp if available, otherwise take last 3
                if node_input_list and node_input_list[0].get("timestamp"):
                    node_input_list.sort(
                        key=lambda x: x.get("timestamp", ""), reverse=True
                    )
                node["recent_inputs"] = node_input_list[:3]

    # Build node relations from graph links
    node_relations: list[NodeRelation] = []
    for link in graph_links:
        # Include link details with source and sink information (truncated UUIDs)
        relation: NodeRelation = {
            "source_node_id": _truncate_uuid(link.source_id),
            "sink_node_id": _truncate_uuid(link.sink_id),
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
        "overall_status": {
            "total_nodes_in_graph": len(nodes),
            "total_executions": execution_stats.node_count,
            "total_errors": execution_stats.node_error_count,
            "execution_time_seconds": execution_stats.walltime,
            "has_errors": bool(
                execution_stats.error or execution_stats.node_error_count > 0
            ),
            "graph_error": (
                str(execution_stats.error) if execution_stats.error else None
            ),
            "graph_execution_status": (
                execution_status.value if execution_status else None
            ),
        },
    }


@func_retry
async def _call_llm_direct(
    credentials: APIKeyCredentials, prompt: list[dict[str, str]]
) -> str:
    """Make direct LLM call."""

    response = await llm_call(
        credentials=credentials,
        llm_model=LlmModel.GPT4O_MINI,
        prompt=prompt,
        max_tokens=150,
        compress_prompt_to_fit=True,
    )

    if response and response.response:
        return response.response.strip()
    else:
        return "Unable to generate activity summary"
