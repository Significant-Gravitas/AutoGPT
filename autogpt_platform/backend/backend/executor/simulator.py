"""
LLM-powered block simulator for dry-run execution.

When dry_run=True, instead of calling the real block, this module
role-plays the block's execution using an LLM.  For most blocks no real
API calls or side effects occur.  OrchestratorBlock is an exception --
it executes for real with a cheap model so the orchestrator can make LLM
calls (handled in manager.py via ``prepare_dry_run``).

The LLM simulation is grounded by:
  - Block name and description
  - Input/output schemas (from block.input_schema.jsonschema() / output_schema.jsonschema())
  - The actual input values

Inspired by https://github.com/Significant-Gravitas/agent-simulator
"""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.llm import LlmModel
from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.orchestrator import OrchestratorBlock
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)


# Use the same fast/cheap model the copilot uses for non-primary tasks.
# Overridable via ChatConfig.title_model if ChatConfig is available.
def _simulator_model() -> str:
    try:
        from backend.copilot.config import ChatConfig  # noqa: PLC0415

        model = ChatConfig().title_model
    except Exception:
        model = "openai/gpt-4o-mini"

    # get_openai_client() may return a direct OpenAI client (not OpenRouter).
    # Direct OpenAI expects bare model names ("gpt-4o-mini"), not the
    # OpenRouter-prefixed form ("openai/gpt-4o-mini").  Strip the prefix when
    # the internal OpenAI key is configured (i.e. not going through OpenRouter).
    try:
        from backend.util.settings import Settings  # noqa: PLC0415

        secrets = Settings().secrets
        # get_openai_client() uses the direct OpenAI client whenever
        # openai_internal_api_key is set, regardless of open_router_api_key.
        # Strip the provider prefix (e.g. "openai/gpt-4o-mini" -> "gpt-4o-mini")
        # so the model name is valid for the direct OpenAI API.
        if secrets.openai_internal_api_key and "/" in model:
            model = model.split("/", 1)[1]
    except Exception:
        pass

    return model


_TEMPERATURE = 0.2
_MAX_JSON_RETRIES = 5
_MAX_INPUT_VALUE_CHARS = 20000

# Cheap model used when executing OrchestratorBlock during dry-run.
DRY_RUN_MODEL = LlmModel.GPT4O_MINI


def _truncate_value(value: Any) -> Any:
    """Recursively truncate long strings anywhere in a value."""
    if isinstance(value, str):
        return (
            value[:_MAX_INPUT_VALUE_CHARS] + "... [TRUNCATED]"
            if len(value) > _MAX_INPUT_VALUE_CHARS
            else value
        )
    if isinstance(value, dict):
        return {k: _truncate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item) for item in value]
    return value


def _truncate_input_values(input_data: dict[str, Any]) -> dict[str, Any]:
    """Recursively truncate long string values so the prompt doesn't blow up."""
    return {k: _truncate_value(v) for k, v in input_data.items()}


def _describe_schema_pins(schema: dict[str, Any]) -> str:
    """Format output pins as a bullet list for the prompt."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = []
    for pin_name, pin_schema in properties.items():
        pin_type = pin_schema.get("type", "any")
        req = "required" if pin_name in required else "optional"
        lines.append(f"- {pin_name}: {pin_type} ({req})")
    return "\n".join(lines) if lines else "(no output pins defined)"


# ---------------------------------------------------------------------------
# Shared LLM call helper
# ---------------------------------------------------------------------------


async def _call_llm_for_simulation(
    system_prompt: str,
    user_prompt: str,
    *,
    label: str = "simulate",
) -> dict[str, Any]:
    """Send a simulation prompt to the LLM and return the parsed JSON dict.

    Handles client acquisition, retries on invalid JSON, and logging.

    Raises:
        RuntimeError: If no LLM client is available.
        ValueError: If all retry attempts are exhausted.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "[SIMULATOR ERROR — NOT A BLOCK FAILURE] No LLM client available "
            "(missing OpenAI/OpenRouter API key)."
        )

    model = _simulator_model()
    last_error: Exception | None = None
    for attempt in range(_MAX_JSON_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if not response.choices:
                raise ValueError("LLM returned empty choices array")
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM returned non-object JSON: {raw[:200]}")

            logger.debug(
                "simulate(%s): attempt=%d tokens=%s/%s",
                label,
                attempt + 1,
                getattr(getattr(response, "usage", None), "prompt_tokens", "?"),
                getattr(getattr(response, "usage", None), "completion_tokens", "?"),
            )
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                "simulate(%s): JSON parse error on attempt %d/%d: %s",
                label,
                attempt + 1,
                _MAX_JSON_RETRIES,
                e,
            )
        except Exception as e:
            last_error = e
            logger.error("simulate(%s): LLM call failed: %s", label, e, exc_info=True)
            break

    msg = (
        f"[SIMULATOR ERROR — NOT A BLOCK FAILURE] Failed after {_MAX_JSON_RETRIES} "
        f"attempts: {last_error}"
    )
    logger.error(
        "simulate(%s): all retries exhausted; last_error=%s", label, last_error
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_simulation_prompt(block: Any, input_data: dict[str, Any]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for block simulation."""
    input_schema = block.input_schema.jsonschema()
    output_schema = block.output_schema.jsonschema()

    input_pins = _describe_schema_pins(input_schema)
    output_pins = _describe_schema_pins(output_schema)
    output_properties = list(output_schema.get("properties", {}).keys())
    # Build a separate list for the "MUST include" instruction that excludes
    # "error" — the prompt already tells the LLM to OMIT the error pin unless
    # simulating a logical error.  Including it in "MUST include" is contradictory.
    required_output_properties = [k for k in output_properties if k != "error"]

    block_name = getattr(block, "name", type(block).__name__)
    block_description = getattr(block, "description", "No description available.")

    system_prompt = f"""You are simulating the execution of a software block called "{block_name}".

## Block Description
{block_description}

## Input Schema
{input_pins}

## Output Schema (what you must return)
{output_pins}

Your task: given the current inputs, produce realistic simulated outputs for this block.

Rules:
- Respond with a single JSON object whose keys are EXACTLY the output pin names listed above.
- Assume all credentials and authentication are present and valid. Never simulate authentication failures.
- Make the simulated outputs realistic and consistent with the inputs.
- If there is an "error" pin, OMIT it entirely unless you are simulating a logical error. Only include the "error" pin when there is a genuine error message to report.
- Do not include any extra keys beyond the output pins.

Output pin names you MUST include: {json.dumps(required_output_properties)}"""

    safe_inputs = _truncate_input_values(input_data)
    user_prompt = f"## Current Inputs\n{json.dumps(safe_inputs, indent=2)}"

    return system_prompt, user_prompt


def _build_mcp_simulation_prompt(
    input_data: dict[str, Any],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for MCP tool simulation.

    Uses the tool name, its JSON Schema, and the supplied arguments to let the
    LLM generate a realistic response.
    """
    tool_name = input_data.get("selected_tool", "unknown_tool")
    tool_description = input_data.get("tool_description", "")
    tool_schema = input_data.get("tool_input_schema", {})
    tool_arguments = input_data.get("tool_arguments", {})
    server_url = input_data.get("server_url", "")

    if tool_schema:
        schema_text = json.dumps(tool_schema, indent=2)
        if len(schema_text) > _MAX_INPUT_VALUE_CHARS:
            schema_text = schema_text[:_MAX_INPUT_VALUE_CHARS] + "... [TRUNCATED]"
    else:
        schema_text = "(none)"
    desc_line = f"\n- Description: {tool_description}" if tool_description else ""

    system_prompt = f"""You are simulating the execution of an MCP (Model Context Protocol) tool.

## Tool Details
- Tool name: {tool_name}{desc_line}
- MCP server: {server_url}

## Tool Input Schema
{schema_text}

Your task: given the tool arguments below, produce a realistic simulated output
for this MCP tool call.

Rules:
- Respond with a single JSON object with exactly two keys: "result" and "error".
- "result" should contain realistic output data that the tool would return.
- "error" should be "" (empty string) unless you are simulating a logical error.
- Assume all credentials and authentication are present and valid. Never simulate authentication failures.
- Base your response on what a tool named "{tool_name}" with the given schema would realistically return."""

    safe_args = _truncate_input_values(tool_arguments)
    user_prompt = f"## Tool Arguments\n{json.dumps(safe_args, indent=2)}"

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Public simulation functions
# ---------------------------------------------------------------------------


def prepare_dry_run(block: Any, input_data: dict[str, Any]) -> dict[str, Any] | None:
    """Prepare *input_data* for a dry-run execution of *block*.

    Returns a **modified copy** of *input_data* for blocks that should execute
    for real with cheap settings (e.g. OrchestratorBlock), or ``None`` when the
    block should be LLM-simulated instead.
    """
    if isinstance(block, OrchestratorBlock):
        # Preserve the user's configured mode: 0 means traditional (single
        # LLM call, no agent loop). Only override to 1 when the user chose
        # agent mode (non-zero) so the dry run still exercises the loop once
        # without running away.
        original = input_data.get("agent_mode_max_iterations", 0)
        max_iters = 1 if original != 0 else 0
        return {
            **input_data,
            "model": DRY_RUN_MODEL,
            "agent_mode_max_iterations": max_iters,
        }
    if isinstance(block, AgentExecutorBlock):
        # Let the AgentExecutorBlock execute for real so the sub-agent graph
        # is triggered.  The sub-agent's blocks will be individually simulated
        # because dry_run propagates via execution_context.
        return {**input_data}
    return None


async def simulate_mcp_block(
    _block: Any,
    input_data: dict[str, Any],
    simulation_context: dict[str, Any] | None = None,
) -> AsyncGenerator[tuple[str, Any]]:
    """Simulate MCP tool execution using an LLM.

    Unlike the generic ``simulate_block``, this builds a prompt grounded in
    the selected MCP tool's name and JSON Schema so the LLM can produce a
    realistic response for that specific tool.

    Yields ``(output_name, output_data)`` tuples matching the Block.execute()
    interface.
    """
    system_prompt, user_prompt = _build_mcp_simulation_prompt(input_data)
    label = input_data.get("selected_tool", "mcp_tool")

    try:
        parsed = await _call_llm_for_simulation(system_prompt, user_prompt, label=label)
        yield "result", parsed.get("result", None)
        yield "error", parsed.get("error", "")
    except (RuntimeError, ValueError) as e:
        yield "error", str(e)


async def simulate_block(
    block: Any,
    input_data: dict[str, Any],
    simulation_context: dict[str, Any] | None = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Simulate block execution using an LLM.

    For MCPToolBlock, uses a specialised prompt grounded in the tool's schema.

    Note: callers should check ``prepare_dry_run(block, input_data)`` first.
    OrchestratorBlock executes for real with a cheap model in dry-run mode
    (see manager.py).

    Yields (output_name, output_data) tuples matching the Block.execute() interface.
    On unrecoverable failure, yields a single ("error", "[SIMULATOR ERROR ...") tuple.
    """
    # MCPToolBlock gets a specialised simulation using its tool schema.
    if isinstance(block, MCPToolBlock):
        async for output in simulate_mcp_block(block, input_data):
            yield output
        return

    output_schema = block.output_schema.jsonschema()
    output_properties: dict[str, Any] = output_schema.get("properties", {})

    system_prompt, user_prompt = build_simulation_prompt(block, input_data)
    label = getattr(block, "name", "?")

    try:
        parsed = await _call_llm_for_simulation(system_prompt, user_prompt, label=label)

        # Fill missing output pins with defaults.
        # Skip empty "error" pins — an empty string means "no error" and
        # would only confuse downstream consumers (LLM, frontend).
        result: dict[str, Any] = {}
        for pin_name in output_properties:
            if pin_name in parsed:
                value = parsed[pin_name]
                # Drop empty/blank error pins: they carry no information.
                if (
                    pin_name == "error"
                    and isinstance(value, str)
                    and not value.strip()
                ):
                    continue
                result[pin_name] = value
            elif pin_name != "error":
                # Only fill non-error missing pins with None
                result[pin_name] = None

        for pin_name, pin_value in result.items():
            yield pin_name, pin_value
    except (RuntimeError, ValueError) as e:
        yield "error", str(e)
