"""
Module for generating AI-based activity status for graph executions.
"""

import json
import logging
import math
from typing import TYPE_CHECKING, Any, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam

from backend.blocks import get_block
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import GraphExecutionStats
from backend.data.platform_cost import PlatformCostEntry, usd_to_microdollars
from backend.executor.cost_tracking import schedule_platform_cost_log
from backend.util.clients import get_openai_client
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.truncate import truncate

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerAsyncClient

logger = logging.getLogger(__name__)


# OpenRouter `extra_body` flag that embeds the real generation cost on
# `response.usage.cost`. Same shape as backend/executor/simulator.py — keep
# the two in sync.
_OPENROUTER_INCLUDE_USAGE_COST: dict[str, Any] = {"usage": {"include": True}}

_MAX_JSON_RETRIES = 3
_MAX_OUTPUT_TOKENS = 150


# Default system prompt template for activity status generation
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant analyzing what an agent execution accomplished and whether it worked correctly. 
You need to provide both a user-friendly summary AND a correctness assessment.

FOR THE ACTIVITY STATUS:
- Write from the user's perspective about what they accomplished, NOT about technical execution details
- Focus on the ACTUAL TASK the user wanted done, not the internal workflow steps
- Avoid technical terms like 'workflow', 'execution', 'components', 'nodes', 'processing', etc.
- Keep it to 3 sentences maximum. Be conversational and human-friendly

FOR THE CORRECTNESS SCORE:
- Provide a score from 0.0 to 1.0 indicating how well the execution achieved its intended purpose
- Use this scoring guide:
  0.0-0.2: Failure - The result clearly did not meet the task requirements
  0.2-0.4: Poor - Major issues; only small parts of the goal were achieved
  0.4-0.6: Partial Success - Some objectives met, but with noticeable gaps or inaccuracies
  0.6-0.8: Mostly Successful - Largely achieved the intended outcome, with minor flaws
  0.8-1.0: Success - Fully met or exceeded the task requirements
- Base the score on actual outputs produced, not just technical completion

UNDERSTAND THE INTENDED PURPOSE:
- FIRST: Read the graph description carefully to understand what the user wanted to accomplish
- The graph name and description tell you the main goal/intention of this automation
- Use this intended purpose as your PRIMARY criteria for success/failure evaluation
- Ask yourself: 'Did this execution actually accomplish what the graph was designed to do?'

CRITICAL OUTPUT ANALYSIS:
- Check if blocks that should produce user-facing results actually produced outputs
- Blocks with names containing 'Output', 'Post', 'Create', 'Send', 'Publish', 'Generate' are usually meant to produce final results
- If these critical blocks have NO outputs (empty recent_outputs), the task likely FAILED even if status shows 'completed'
- Sub-agents (AgentExecutorBlock) that produce no outputs usually indicate failed sub-tasks
- Most importantly: Does the execution result match what the graph description promised to deliver?

SUCCESS EVALUATION BASED ON INTENTION:
- If the graph is meant to 'create blog posts' → check if blog content was actually created
- If the graph is meant to 'send emails' → check if emails were actually sent
- If the graph is meant to 'analyze data' → check if analysis results were produced
- If the graph is meant to 'generate reports' → check if reports were generated
- Technical completion ≠ goal achievement. Focus on whether the USER'S INTENDED OUTCOME was delivered

IMPORTANT: Be HONEST about what actually happened:
- If the input was invalid/nonsensical, say so directly
- If the task failed, explain what went wrong in simple terms
- If errors occurred, focus on what the user needs to know
- Only claim success if the INTENDED PURPOSE was genuinely accomplished AND produced expected outputs
- Don't sugar-coat failures or present them as helpful feedback
- ESPECIALLY: If the graph's main purpose wasn't achieved, this is a failure regardless of 'completed' status

Understanding Errors:
- Node errors: Individual steps may fail but the overall task might still complete (e.g., one data source fails but others work)
- Graph error (in overall_status.graph_error): This means the entire execution failed and nothing was accomplished
- Missing outputs from critical blocks: Even if no errors, this means the task failed to produce expected results
- Focus on whether the graph's intended purpose was fulfilled, not whether technical steps completed"""

# Default user prompt template for activity status generation
DEFAULT_USER_PROMPT = """A user ran '{{GRAPH_NAME}}' to accomplish something. Based on this execution data, 
provide both an activity summary and correctness assessment:

{{EXECUTION_DATA}}

ANALYSIS CHECKLIST:
1. READ graph_info.description FIRST - this tells you what the user intended to accomplish
2. Check overall_status.graph_error - if present, the entire execution failed
3. Look for nodes with 'Output', 'Post', 'Create', 'Send', 'Publish', 'Generate' in their block_name
4. Check if these critical blocks have empty recent_outputs arrays - this indicates failure
5. Look for AgentExecutorBlock (sub-agents) with no outputs - this suggests sub-task failures
6. Count how many nodes produced outputs vs total nodes - low ratio suggests problems
7. MOST IMPORTANT: Does the execution outcome match what graph_info.description promised?

INTENTION-BASED EVALUATION:
- If description mentions 'blog writing' → did it create blog content?
- If description mentions 'email automation' → were emails actually sent?
- If description mentions 'data analysis' → were analysis results produced?
- If description mentions 'content generation' → was content actually generated?
- If description mentions 'social media posting' → were posts actually made?
- Match the outputs to the stated intention, not just technical completion

PROVIDE:
activity_status: 1-3 sentences about what the user accomplished, such as:
- 'I analyzed your resume and provided detailed feedback for the IT industry.'
- 'I couldn't complete the task because critical steps failed to produce any results.'
- 'I failed to generate the content you requested due to missing API access.'
- 'I extracted key information from your documents and organized it into a summary.'
- 'The task failed because the blog post creation step didn't produce any output.'

correctness_score: A float score from 0.0 to 1.0 based on how well the intended purpose was achieved:
- 0.0-0.2: Failure (didn't meet requirements)
- 0.2-0.4: Poor (major issues, minimal achievement)
- 0.4-0.6: Partial Success (some objectives met with gaps)
- 0.6-0.8: Mostly Successful (largely achieved with minor flaws)
- 0.8-1.0: Success (fully met or exceeded requirements)

BE CRITICAL: If the graph's intended purpose (from description) wasn't achieved, use a low score (0.0-0.4) even if status is 'completed'."""


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
    model_name: str = "gpt-4o-mini",
    skip_feature_flag: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    skip_existing: bool = True,
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
        model_name: AI model to use for generation (default: gpt-4o-mini)
        skip_feature_flag: Whether to skip LaunchDarkly feature flag check
        system_prompt: Custom system prompt template (default: DEFAULT_SYSTEM_PROMPT)
        user_prompt: Custom user prompt template with placeholders (default: DEFAULT_USER_PROMPT)
        skip_existing: Whether to skip if activity_status and correctness_score already exist

    Returns:
        AI-generated activity status response with activity_status and correctness_status,
        or None if feature is disabled or skipped
    """
    # Check LaunchDarkly feature flag for AI activity status generation with full context support
    if not skip_feature_flag and not await is_feature_enabled(
        Flag.AI_ACTIVITY_STATUS, user_id
    ):
        logger.debug("AI activity status generation is disabled via LaunchDarkly")
        return None

    # Check if we should skip existing data (for admin regeneration option)
    if (
        skip_existing
        and execution_stats.activity_status
        and execution_stats.correctness_score is not None
    ):
        logger.debug(
            f"Skipping activity status generation for {graph_exec_id}: already exists"
        )
        return {
            "activity_status": execution_stats.activity_status,
            "correctness_score": execution_stats.correctness_score,
        }

    # Acquire an OpenRouter-backed OpenAI client. Activity-status generation is
    # only meaningful when we can record real USD cost in PlatformCostLog;
    # without OpenRouter the upstream platform cost would be tokens-only, so
    # we skip rather than fall back to direct-OpenAI which produces no
    # provider_cost. Same gating pattern as backend/executor/simulator.py.
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.debug(
            "OpenRouter API key not configured, skipping activity status generation"
        )
        return None

    try:
        # Get all node executions for this graph execution
        node_executions = await db_client.get_node_executions(
            graph_exec_id, include_exec_data=True
        )

        # Get graph metadata and full graph structure for name, description, and links
        graph_metadata = await db_client.get_graph_metadata(graph_id, graph_version)
        graph = await db_client.get_graph(
            graph_id=graph_id,
            version=graph_version,
            user_id=user_id,
            skip_access_check=True,
        )

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

        # Append the JSON-shape contract to the system prompt so the model
        # returns object-shaped JSON we can parse without bracketed wrappers.
        system_prompt_with_format = (
            f"{system_prompt}\n\n"
            "Return a JSON object with exactly these two keys and nothing else:\n"
            "  - activity_status: string (1-3 sentences)\n"
            "  - correctness_score: number between 0.0 and 1.0\n"
        )

        execution_data_json = json.dumps(execution_data, indent=2)
        user_prompt_content = user_prompt.replace("{{GRAPH_NAME}}", graph_name).replace(
            "{{EXECUTION_DATA}}", execution_data_json
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt_with_format},
            {"role": "user", "content": user_prompt_content},
        ]
        logger.debug(
            f"Sending prompt to LLM for graph execution {graph_exec_id}: {json.dumps(messages, indent=2)}"
        )

        # Model values arriving without a provider prefix (e.g. "gpt-4o-mini")
        # need to be remapped to OpenRouter's namespaced form. Already-prefixed
        # values (e.g. "openai/gpt-4o-mini", "anthropic/claude-...") pass through.
        or_model = model_name if "/" in model_name else f"openai/{model_name}"

        # Track the most recent attempt's usage so we can persist cost even
        # when every retry fails — the API calls were billed regardless of
        # whether parsing succeeded.
        last_error: Exception | None = None
        last_usage: CompletionUsage | None = None
        activity_response: ActivityStatusResponse | None = None
        for attempt in range(_MAX_JSON_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=or_model,
                    messages=messages,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body=_OPENROUTER_INCLUDE_USAGE_COST,
                )
                last_usage = response.usage
                if not response.choices:
                    raise ValueError("OpenRouter returned empty choices array")
                raw = response.choices[0].message.content or ""
                candidate = json.loads(raw)
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"OpenRouter returned non-object JSON: {raw[:200]}"
                    )
                if (
                    "activity_status" not in candidate
                    or "correctness_score" not in candidate
                ):
                    raise ValueError(
                        f"OpenRouter response missing required keys: {raw[:200]}"
                    )
                score = max(0.0, min(1.0, float(candidate["correctness_score"])))
                activity_response = {
                    "activity_status": str(candidate["activity_status"]),
                    "correctness_score": score,
                }
                break
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                logger.warning(
                    "activity_status: JSON parse error on attempt %d/%d: %s",
                    attempt + 1,
                    _MAX_JSON_RETRIES,
                    e,
                )

        # Persist whatever usage the API actually billed us for, even on full
        # retry-exhaustion — the spend happened either way and must show up
        # in PlatformCostLog for admin attribution.
        _persist_activity_status_cost(
            cost_usd=_extract_cost_usd(last_usage),
            input_tokens=last_usage.prompt_tokens if last_usage is not None else 0,
            output_tokens=last_usage.completion_tokens if last_usage is not None else 0,
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            model_name=or_model,
            db_client=db_client,
        )

        if activity_response is None:
            raise RuntimeError(
                f"Failed to parse OpenRouter response after {_MAX_JSON_RETRIES} attempts: {last_error}"
            )

        logger.debug(
            f"Generated activity status for {graph_exec_id}: {activity_response}"
        )
        return activity_response

    except Exception as e:
        logger.exception(
            f"Failed to generate activity status for execution {graph_exec_id}: {str(e)}"
        )
        return None


def _extract_cost_usd(usage: CompletionUsage | None) -> float | None:
    """Return the provider-reported USD cost on the response usage object.

    OpenRouter attaches a ``cost`` field to the OpenAI-compatible usage object
    when the request body includes ``usage: {"include": True}``. The typed
    ``CompletionUsage`` does not declare it, so we read it off ``model_extra``.
    Mirrors backend/executor/simulator.py::_extract_cost_usd — keep aligned.
    """
    if usage is None:
        return None
    extras = usage.model_extra or {}
    if "cost" not in extras:
        return None
    raw = extras["cost"]
    if raw is None:
        logger.error("[activity_status] usage.cost is present but null")
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.error("[activity_status] usage.cost is not numeric: %r", raw)
        return None
    if not math.isfinite(val) or val < 0:
        logger.error("[activity_status] usage.cost is non-finite or negative: %r", val)
        return None
    return val


def _persist_activity_status_cost(
    *,
    cost_usd: float | None,
    input_tokens: int,
    output_tokens: int,
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    model_name: str,
    db_client: "DatabaseManagerAsyncClient",
) -> None:
    """Schedule a PlatformCostLog entry for the activity-status LLM call.

    Mirrors ``backend.copilot.token_tracking._schedule_cost_log``: the platform
    pays for this call (platform OpenRouter key) but the user is not billed
    and not rate-limited, so we skip ``record_cost_usage`` and only write to
    ``PlatformCostLog`` for admin attribution.

    Cost-logging is best-effort: any failure here is swallowed so a transient
    DB / scheduling error never strips a successful activity-status response
    from the user.
    """
    try:
        # Skip when there is genuinely nothing to log. ``not cost_usd`` covers
        # both ``None`` and ``0.0`` so a zero-cost zero-token call doesn't
        # write an empty row that just dilutes dashboard averages.
        if not cost_usd and input_tokens == 0 and output_tokens == 0:
            return

        cost_microdollars = (
            usd_to_microdollars(cost_usd) if cost_usd is not None else None
        )
        if cost_usd is not None:
            tracking_type = "cost_usd"
            tracking_amount = float(cost_usd)
        else:
            tracking_type = "tokens"
            tracking_amount = float(input_tokens + output_tokens)

        schedule_platform_cost_log(
            db_client,
            PlatformCostEntry(
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                graph_id=graph_id,
                block_name="activity_status_generator",
                provider="open_router",
                cost_microdollars=cost_microdollars,
                input_tokens=input_tokens or None,
                output_tokens=output_tokens or None,
                model=model_name,
                tracking_type=tracking_type,
                tracking_amount=tracking_amount,
                metadata={"source": "activity_status_generator"},
            ),
        )
    except Exception:
        logger.exception(
            "Failed to persist activity-status cost for graph_exec %s; "
            "the activity status itself was returned successfully",
            graph_exec_id,
        )


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
