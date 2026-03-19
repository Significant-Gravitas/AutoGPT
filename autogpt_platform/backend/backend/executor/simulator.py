"""
LLM-powered block simulator for dry-run execution.

When dry_run=True, instead of calling the real block, this module
role-plays the block's execution using an LLM. No real API calls,
no side effects. The LLM is grounded by:
  - Block name and description
  - Input/output schemas (from block.input_schema.jsonschema() / output_schema.jsonschema())
  - The actual input values

Inspired by https://github.com/Significant-Gravitas/agent-simulator
"""

import json
import logging
from typing import Any, AsyncGenerator

from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

_MODEL = "openai/gpt-4o-mini"
_TEMPERATURE = 0.2
_MAX_JSON_RETRIES = 5
_MAX_INPUT_VALUE_CHARS = 20000


def _truncate_input_values(input_data: dict[str, Any]) -> dict[str, Any]:
    """Truncate long string values so the prompt doesn't blow up."""
    result = {}
    for k, v in input_data.items():
        if isinstance(v, str) and len(v) > _MAX_INPUT_VALUE_CHARS:
            result[k] = v[:_MAX_INPUT_VALUE_CHARS] + "... [TRUNCATED]"
        else:
            result[k] = v
    return result


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


async def simulate_block(
    block: Any,
    input_data: dict[str, Any],
) -> AsyncGenerator[tuple[str, Any], None]:
    """Simulate block execution using an LLM.

    Yields (output_name, output_data) tuples matching the Block.execute() interface.
    On unrecoverable failure, yields a single ("error", "[SIMULATOR ERROR ...") tuple.
    """
    client = get_openai_client()
    if client is None:
        yield (
            "error",
            "[SIMULATOR ERROR — NOT A BLOCK FAILURE] No LLM client available "
            "(missing OpenAI/OpenRouter API key).",
        )
        return

    output_schema = block.output_schema.jsonschema()
    output_properties: dict[str, Any] = output_schema.get("properties", {})

    system_prompt, user_prompt = build_simulation_prompt(block, input_data)

    last_error: Exception | None = None
    for attempt in range(_MAX_JSON_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM returned non-object JSON: {raw[:200]}")

            # Fill missing output pins with defaults
            result: dict[str, Any] = {}
            for pin_name in output_properties:
                if pin_name in parsed:
                    result[pin_name] = parsed[pin_name]
                else:
                    result[pin_name] = "" if pin_name == "error" else None

            logger.debug(
                "simulate_block: block=%s attempt=%d tokens=%s/%s",
                getattr(block, "name", "?"),
                attempt + 1,
                getattr(getattr(response, "usage", None), "prompt_tokens", "?"),
                getattr(getattr(response, "usage", None), "completion_tokens", "?"),
            )

            for pin_name, pin_value in result.items():
                yield pin_name, pin_value
            return

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                "simulate_block: JSON parse error on attempt %d/%d: %s",
                attempt + 1,
                _MAX_JSON_RETRIES,
                e,
            )
        except Exception as e:
            last_error = e
            logger.error("simulate_block: LLM call failed: %s", e, exc_info=True)
            break

    yield (
        "error",
        f"[SIMULATOR ERROR — NOT A BLOCK FAILURE] Failed after {_MAX_JSON_RETRIES} "
        f"attempts: {last_error}",
    )
