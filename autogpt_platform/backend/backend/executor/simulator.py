"""
LLM-powered block simulator for dry-run execution.

When dry_run=True, instead of calling the real block, this module
role-plays the block's execution using an LLM.  For most blocks no real
API calls or side effects occur.  OrchestratorBlock and AgentExecutorBlock
are exceptions — they execute for real so the orchestrator can make LLM
calls and agent executors can spawn child graphs (handled in manager.py).

The LLM simulation is grounded by:
  - Block name and description
  - Input/output schemas (from block.input_schema.jsonschema() / output_schema.jsonschema())
  - The actual input values

Inspired by https://github.com/Significant-Gravitas/agent-simulator
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

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
        # Strip the provider prefix (e.g. "openai/gpt-4o-mini" → "gpt-4o-mini")
        # so the model name is valid for the direct OpenAI API.
        if secrets.openai_internal_api_key and "/" in model:
            model = model.split("/", 1)[1]
    except Exception:
        pass

    return model


_TEMPERATURE = 0.2
_MAX_JSON_RETRIES = 5
_MAX_INPUT_VALUE_CHARS = 20000


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

    Args:
        system_prompt: The system message for the LLM.
        user_prompt: The user message for the LLM.
        label: A short identifier used in log messages (e.g. block name or tool name).

    Returns:
        The parsed JSON dict from the LLM response.

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
- If there is an "error" pin, set it to "" (empty string) unless you are simulating a logical error.
- Do not include any extra keys beyond the output pins.

Output pin names you MUST include: {json.dumps(output_properties)}
"""

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
    tool_schema = input_data.get("tool_input_schema", {})
    tool_arguments = input_data.get("tool_arguments", {})
    server_url = input_data.get("server_url", "")

    schema_text = json.dumps(tool_schema, indent=2) if tool_schema else "(none)"

    system_prompt = f"""You are simulating the execution of an MCP (Model Context Protocol) tool.

## Tool Details
- Tool name: {tool_name}
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
- Base your response on what a tool named "{tool_name}" with the given schema would realistically return.
"""

    safe_args = _truncate_input_values(tool_arguments)
    user_prompt = f"## Tool Arguments\n{json.dumps(safe_args, indent=2)}"

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Public simulation functions
# ---------------------------------------------------------------------------


async def simulate_mcp_block(
    _block: Any,
    input_data: dict[str, Any],
) -> AsyncIterator[tuple[str, Any]]:
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
) -> AsyncIterator[tuple[str, Any]]:
    """Simulate block execution using an LLM.

    Yields (output_name, output_data) tuples matching the Block.execute() interface.
    On unrecoverable failure, yields a single ("error", "[SIMULATOR ERROR ...") tuple.
    """
    output_schema = block.output_schema.jsonschema()
    output_properties: dict[str, Any] = output_schema.get("properties", {})

    system_prompt, user_prompt = build_simulation_prompt(block, input_data)
    label = getattr(block, "name", "?")

    try:
        parsed = await _call_llm_for_simulation(system_prompt, user_prompt, label=label)

        # Fill missing output pins with defaults
        result: dict[str, Any] = {}
        for pin_name in output_properties:
            if pin_name in parsed:
                result[pin_name] = parsed[pin_name]
            else:
                result[pin_name] = "" if pin_name == "error" else None

        for pin_name, pin_value in result.items():
            yield pin_name, pin_value
    except (RuntimeError, ValueError) as e:
        yield "error", str(e)
