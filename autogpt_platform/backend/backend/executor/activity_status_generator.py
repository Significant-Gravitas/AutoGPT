"""
Module for generating AI-based activity status for graph executions.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from pydantic import SecretStr

from backend.blocks.llm import AIStructuredResponseGeneratorBlock, LlmModel
from backend.data.block import get_block
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.util.feature_flag import Flag, is_feature_enabled
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


class ActivityStatusResponse(TypedDict):
    """Type definition for structured activity status response."""

    activity_status: str
    correctness_score: float


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
) -> ActivityStatusResponse | None:
    """
    Generate an AI-based activity status summary and correctness assessment for a graph execution.

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
        AI-generated activity status response with activity_status and correctness_status,
        or None if feature is disabled
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

        # Prepare prompt for AI with structured output requirements
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant analyzing what an agent execution accomplished and whether it worked correctly. "
                    "You need to provide both a user-friendly summary AND a correctness assessment.\n\n"
                    "FOR THE ACTIVITY STATUS:\n"
                    "- Write from the user's perspective about what they accomplished, NOT about technical execution details\n"
                    "- Focus on the ACTUAL TASK the user wanted done, not the internal workflow steps\n"
                    "- Avoid technical terms like 'workflow', 'execution', 'components', 'nodes', 'processing', etc.\n"
                    "- Keep it to 3 sentences maximum. Be conversational and human-friendly\n\n"
                    "FOR THE CORRECTNESS SCORE:\n"
                    "- Provide a score from 0.0 to 1.0 indicating how well the execution achieved its intended purpose\n"
                    "- Use this scoring guide:\n"
                    "  0.0-0.2: Failure - The result clearly did not meet the task requirements\n"
                    "  0.2-0.4: Poor - Major issues; only small parts of the goal were achieved\n"
                    "  0.4-0.6: Partial Success - Some objectives met, but with noticeable gaps or inaccuracies\n"
                    "  0.6-0.8: Mostly Successful - Largely achieved the intended outcome, with minor flaws\n"
                    "  0.8-1.0: Success - Fully met or exceeded the task requirements\n"
                    "- Base the score on actual outputs produced, not just technical completion\n\n"
                    "UNDERSTAND THE INTENDED PURPOSE:\n"
                    "- FIRST: Read the graph description carefully to understand what the user wanted to accomplish\n"
                    "- The graph name and description tell you the main goal/intention of this automation\n"
                    "- Use this intended purpose as your PRIMARY criteria for success/failure evaluation\n"
                    "- Ask yourself: 'Did this execution actually accomplish what the graph was designed to do?'\n\n"
                    "CRITICAL OUTPUT ANALYSIS:\n"
                    "- Check if blocks that should produce user-facing results actually produced outputs\n"
                    "- Blocks with names containing 'Output', 'Post', 'Create', 'Send', 'Publish', 'Generate' are usually meant to produce final results\n"
                    "- If these critical blocks have NO outputs (empty recent_outputs), the task likely FAILED even if status shows 'completed'\n"
                    "- Sub-agents (AgentExecutorBlock) that produce no outputs usually indicate failed sub-tasks\n"
                    "- Most importantly: Does the execution result match what the graph description promised to deliver?\n\n"
                    "SUCCESS EVALUATION BASED ON INTENTION:\n"
                    "- If the graph is meant to 'create blog posts' → check if blog content was actually created\n"
                    "- If the graph is meant to 'send emails' → check if emails were actually sent\n"
                    "- If the graph is meant to 'analyze data' → check if analysis results were produced\n"
                    "- If the graph is meant to 'generate reports' → check if reports were generated\n"
                    "- Technical completion ≠ goal achievement. Focus on whether the USER'S INTENDED OUTCOME was delivered\n\n"
                    "IMPORTANT: Be HONEST about what actually happened:\n"
                    "- If the input was invalid/nonsensical, say so directly\n"
                    "- If the task failed, explain what went wrong in simple terms\n"
                    "- If errors occurred, focus on what the user needs to know\n"
                    "- Only claim success if the INTENDED PURPOSE was genuinely accomplished AND produced expected outputs\n"
                    "- Don't sugar-coat failures or present them as helpful feedback\n"
                    "- ESPECIALLY: If the graph's main purpose wasn't achieved, this is a failure regardless of 'completed' status\n\n"
                    "Understanding Errors:\n"
                    "- Node errors: Individual steps may fail but the overall task might still complete (e.g., one data source fails but others work)\n"
                    "- Graph error (in overall_status.graph_error): This means the entire execution failed and nothing was accomplished\n"
                    "- Missing outputs from critical blocks: Even if no errors, this means the task failed to produce expected results\n"
                    "- Focus on whether the graph's intended purpose was fulfilled, not whether technical steps completed"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A user ran '{graph_name}' to accomplish something. Based on this execution data, "
                    f"provide both an activity summary and correctness assessment:\n\n"
                    f"{json.dumps(execution_data, indent=2)}\n\n"
                    "ANALYSIS CHECKLIST:\n"
                    "1. READ graph_info.description FIRST - this tells you what the user intended to accomplish\n"
                    "2. Check overall_status.graph_error - if present, the entire execution failed\n"
                    "3. Look for nodes with 'Output', 'Post', 'Create', 'Send', 'Publish', 'Generate' in their block_name\n"
                    "4. Check if these critical blocks have empty recent_outputs arrays - this indicates failure\n"
                    "5. Look for AgentExecutorBlock (sub-agents) with no outputs - this suggests sub-task failures\n"
                    "6. Count how many nodes produced outputs vs total nodes - low ratio suggests problems\n"
                    "7. MOST IMPORTANT: Does the execution outcome match what graph_info.description promised?\n\n"
                    "INTENTION-BASED EVALUATION:\n"
                    "- If description mentions 'blog writing' → did it create blog content?\n"
                    "- If description mentions 'email automation' → were emails actually sent?\n"
                    "- If description mentions 'data analysis' → were analysis results produced?\n"
                    "- If description mentions 'content generation' → was content actually generated?\n"
                    "- If description mentions 'social media posting' → were posts actually made?\n"
                    "- Match the outputs to the stated intention, not just technical completion\n\n"
                    "PROVIDE:\n"
                    "activity_status: 1-3 sentences about what the user accomplished, such as:\n"
                    "- 'I analyzed your resume and provided detailed feedback for the IT industry.'\n"
                    "- 'I couldn't complete the task because critical steps failed to produce any results.'\n"
                    "- 'I failed to generate the content you requested due to missing API access.'\n"
                    "- 'I extracted key information from your documents and organized it into a summary.'\n"
                    "- 'The task failed because the blog post creation step didn't produce any output.'\n\n"
                    "correctness_score: A float score from 0.0 to 1.0 based on how well the intended purpose was achieved:\n"
                    "- 0.0-0.2: Failure (didn't meet requirements)\n"
                    "- 0.2-0.4: Poor (major issues, minimal achievement)\n"
                    "- 0.4-0.6: Partial Success (some objectives met with gaps)\n"
                    "- 0.6-0.8: Mostly Successful (largely achieved with minor flaws)\n"
                    "- 0.8-1.0: Success (fully met or exceeded requirements)\n\n"
                    "BE CRITICAL: If the graph's intended purpose (from description) wasn't achieved, use a low score (0.0-0.4) even if status is 'completed'."
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

        # Define expected response format
        expected_format = {
            "activity_status": "A user-friendly 1-3 sentence summary of what was accomplished",
            "correctness_score": "Float score from 0.0 to 1.0 indicating how well the execution achieved its intended purpose",
        }

        # Use existing AIStructuredResponseGeneratorBlock for structured LLM call
        structured_block = AIStructuredResponseGeneratorBlock()

        # Convert credentials to the format expected by AIStructuredResponseGeneratorBlock
        credentials_input = {
            "provider": credentials.provider,
            "id": credentials.id,
            "type": credentials.type,
            "title": credentials.title,
        }

        structured_input = AIStructuredResponseGeneratorBlock.Input(
            prompt=prompt[1]["content"],  # User prompt content
            sys_prompt=prompt[0]["content"],  # System prompt content
            expected_format=expected_format,
            model=LlmModel.GPT4O_MINI,
            credentials=credentials_input,  # type: ignore
            max_tokens=150,
            retry=3,
        )

        # Execute the structured LLM call
        async for output_name, output_data in structured_block.run(
            structured_input, credentials=credentials
        ):
            if output_name == "response":
                response = output_data
                break
        else:
            raise RuntimeError("Failed to get response from structured LLM call")

        # Create typed response with validation
        correctness_score = float(response["correctness_score"])
        # Clamp score to valid range
        correctness_score = max(0.0, min(1.0, correctness_score))

        activity_response: ActivityStatusResponse = {
            "activity_status": response["activity_status"],
            "correctness_score": correctness_score,
        }

        logger.debug(
            f"Generated activity status for {graph_exec_id}: {activity_response}"
        )

        return activity_response

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
