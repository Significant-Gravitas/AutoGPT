"""
LLM-powered block simulator for dry-run execution.

When dry_run=True, instead of calling the real block, this module
role-plays the block's execution using an LLM.  For most blocks no real
API calls or side effects occur.

Special cases (no LLM simulation needed):
  - OrchestratorBlock executes for real with the platform's simulation model
    (iterations capped to 1).  Uses the platform OpenRouter key so no user
    credentials are required.  Falls back to LLM simulation if the platform
    key is unavailable.
  - AgentExecutorBlock executes for real so it can spawn child graph executions
    (whose blocks are then simulated).  No credentials needed.
  - AgentInputBlock (and all subclasses) and AgentOutputBlock are pure
    passthrough -- they forward their input values directly.
  - MCPToolBlock is simulated via the generic LLM prompt (with run() source code).

OrchestratorBlock and AgentExecutorBlock are handled in manager.py via
``prepare_dry_run``.

The LLM simulation is grounded by:
  - Block name and description
  - Input/output schemas (from block.input_schema.jsonschema() / output_schema.jsonschema())
  - The block's run() source code (via inspect.getsource)
  - The actual input values

Inspired by https://github.com/Significant-Gravitas/agent-simulator
"""

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.orchestrator import OrchestratorBlock
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)


# Default simulator model — Gemini 2.5 Flash via OpenRouter (fast, cheap, good at
# JSON generation).  Configurable via ChatConfig.simulation_model
# (CHAT_SIMULATION_MODEL env var).
_DEFAULT_SIMULATOR_MODEL = "google/gemini-2.5-flash"


def _simulator_model() -> str:
    try:
        from backend.copilot.config import ChatConfig  # noqa: PLC0415

        return ChatConfig().simulation_model or _DEFAULT_SIMULATOR_MODEL
    except Exception:
        return _DEFAULT_SIMULATOR_MODEL


_TEMPERATURE = 0.2
_MAX_JSON_RETRIES = 5
_MAX_INPUT_VALUE_CHARS = 20000
_COMMON_CRED_KEYS = frozenset({"credentials", "api_key", "token", "secret"})


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
    client = get_openai_client(prefer_openrouter=True)
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

    # Include the block's run() source code so the LLM knows exactly how
    # inputs are transformed to outputs.  Truncate to avoid blowing up the
    # prompt for very large blocks.
    try:
        run_source = inspect.getsource(block.run)
        if len(run_source) > _MAX_INPUT_VALUE_CHARS:
            run_source = run_source[:_MAX_INPUT_VALUE_CHARS] + "\n# ... [TRUNCATED]"
    except (TypeError, OSError):
        run_source = ""

    implementation_section = ""
    if run_source:
        implementation_section = (
            "\n## Block Implementation (run function source code)\n"
            "```python\n"
            f"{run_source}\n"
            "```\n"
        )

    system_prompt = f"""You are simulating the execution of a software block called "{block_name}".

## Block Description
{block_description}

## Input Schema
{input_pins}

## Output Schema (what you must return)
{output_pins}
{implementation_section}
Your task: given the current inputs, produce realistic simulated outputs for this block.
{"Study the block's run() source code above to understand exactly how inputs are transformed to outputs." if run_source else "Use the block description and schemas to infer realistic outputs."}

Rules:
- Respond with a single JSON object.
- Only include output pins that have meaningful values. Omit pins with no relevant output.
- Assume all credentials and API keys are present and valid. Do not simulate auth failures.
- Generate REALISTIC, useful outputs: real-looking URLs, plausible text, valid data structures.
- Never return empty strings, null, or "N/A" for pins that should have content.
- You MAY simulate logical errors (e.g., invalid input format, unsupported operation) when the inputs warrant it — use the "error" pin for these. But do NOT simulate auth/credential errors.
- Do not include extra keys beyond the defined output pins.

Available output pins: {json.dumps(required_output_properties)}
"""

    # Strip credentials from input so the LLM doesn't see null/empty creds
    # and incorrectly simulate auth failures.  Use the block's schema to
    # detect credential fields when available, falling back to common names.
    try:
        cred_fields = set(block.input_schema.get_credentials_fields())
    except (AttributeError, TypeError):
        cred_fields = set()
    exclude_keys = cred_fields | _COMMON_CRED_KEYS
    safe_inputs = {
        k: v
        for k, v in _truncate_input_values(input_data).items()
        if k not in exclude_keys
    }
    user_prompt = f"## Current Inputs\n{json.dumps(safe_inputs, indent=2)}"

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Public simulation functions
# ---------------------------------------------------------------------------


def _get_platform_openrouter_key() -> str | None:
    """Return the platform's OpenRouter API key, or None if unavailable."""
    try:
        from backend.util.settings import Settings  # noqa: PLC0415

        key = Settings().secrets.open_router_api_key
        return key if key else None
    except Exception:
        return None


def prepare_dry_run(block: Any, input_data: dict[str, Any]) -> dict[str, Any] | None:
    """Prepare *input_data* for a dry-run execution of *block*.

    Returns a **modified copy** of *input_data* for blocks that should execute
    for real with cheap settings, or ``None`` when the block should be
    LLM-simulated instead.

    - **OrchestratorBlock** executes for real with the platform's simulation
      model (iterations capped to 1).  Uses the platform OpenRouter key so no
      user credentials are needed.  Falls back to LLM simulation if the
      platform key is unavailable.
    - **AgentExecutorBlock** executes for real so it can spawn a child graph
      execution.  The child graph inherits ``dry_run=True`` and its blocks
      are simulated.  No credentials are needed.
    """
    if isinstance(block, OrchestratorBlock):
        or_key = _get_platform_openrouter_key()
        if not or_key:
            logger.info(
                "Dry-run: no platform OpenRouter key, "
                "falling back to LLM simulation for OrchestratorBlock"
            )
            return None

        original = input_data.get("agent_mode_max_iterations", 0)
        max_iters = 1 if original != 0 else 0
        sim_model = _simulator_model()

        # Keep the original credentials dict in input_data so the block's
        # JSON schema validation passes (validate_data strips None values,
        # making the field absent and failing the "required" check).
        # The actual credentials are injected via extra_exec_kwargs in
        # manager.py using _dry_run_api_key.
        return {
            **input_data,
            "agent_mode_max_iterations": max_iters,
            "model": sim_model,
            "_dry_run_api_key": or_key,
        }

    if isinstance(block, AgentExecutorBlock):
        return {**input_data}

    return None


def get_dry_run_credentials(
    input_data: dict[str, Any],
) -> Any | None:
    """Build an ``APIKeyCredentials`` for dry-run OrchestratorBlock execution.

    Returns credentials using the platform's OpenRouter key (injected by
    ``prepare_dry_run``), or ``None`` if not a dry-run override.
    """
    api_key = input_data.pop("_dry_run_api_key", None)
    if not api_key:
        return None

    try:
        from backend.blocks.llm import APIKeyCredentials  # noqa: PLC0415
        from backend.integrations.providers import ProviderName  # noqa: PLC0415

        return APIKeyCredentials(
            id="dry-run-platform",
            provider=ProviderName.OPEN_ROUTER,
            api_key=api_key,
            title="Dry-run simulation",
            expires_at=None,
        )
    except Exception:
        logger.warning("Failed to create dry-run credentials", exc_info=True)
        return None


def _default_for_input_result(result_schema: dict[str, Any], name: str | None) -> Any:
    """Return a type-appropriate sample value for an AgentInputBlock's result pin.

    Typed subclasses (AgentNumberInputBlock, AgentDateInputBlock, etc.)
    declare a specific type/format on their ``result`` output.  When dry-run
    has no user-supplied value, this generates a fallback that matches the
    expected type so downstream validation doesn't fail with a plain string.
    """
    pin_type = result_schema.get("type", "string")
    fmt = result_schema.get("format")

    if pin_type == "integer":
        return 0
    if pin_type == "number":
        return 0.0
    if pin_type == "boolean":
        return False
    if pin_type == "array":
        return []
    if pin_type == "object":
        return {}
    if fmt == "date":
        from datetime import date as _date  # noqa: PLC0415

        return _date.today().isoformat()
    if fmt == "time":
        return "00:00:00"
    # Default: use the block's name as a sample string.
    return name or "sample input"


async def simulate_block(
    block: Any,
    input_data: dict[str, Any],
) -> AsyncGenerator[tuple[str, Any], None]:
    """Simulate block execution using an LLM.

    All block types (including MCPToolBlock) use the same generic LLM prompt
    which includes the block's run() source code for accurate simulation.

    Note: callers should check ``prepare_dry_run(block, input_data)`` first.
    OrchestratorBlock and AgentExecutorBlock execute for real in dry-run mode
    (see manager.py).

    Yields (output_name, output_data) tuples matching the Block.execute() interface.
    On unrecoverable failure, yields a single ("error", "[SIMULATOR ERROR ...") tuple.
    """
    # Input/output blocks are pure passthrough -- they just forward their
    # input values.  No LLM simulation needed.
    if isinstance(block, AgentInputBlock):
        value = input_data.get("value")
        if value is None:
            # Dry-run with no user input: use first dropdown option or name,
            # then coerce to a type-appropriate fallback so typed subclasses
            # (e.g. AgentNumberInputBlock → int, AgentDateInputBlock → date)
            # don't fail validation with a plain string.
            placeholder = input_data.get("options") or input_data.get(
                "placeholder_values"
            )
            if placeholder and isinstance(placeholder, list) and placeholder:
                value = placeholder[0]
            else:
                result_schema = (
                    block.output_schema.jsonschema()
                    .get("properties", {})
                    .get("result", {})
                )
                value = _default_for_input_result(
                    result_schema, input_data.get("name", "sample input")
                )
        yield "result", value
        return

    if isinstance(block, AgentOutputBlock):
        # Mirror AgentOutputBlock.run(): if a format string is provided,
        # apply Jinja2 formatting and yield only "output"; otherwise yield
        # both "output" (raw value) and "name".
        fmt = input_data.get("format", "")
        value = input_data.get("value")
        name = input_data.get("name", "")
        if fmt:
            try:
                from backend.util.text import TextFormatter  # noqa: PLC0415

                escape_html = input_data.get("escape_html", False)
                formatter = TextFormatter(autoescape=escape_html)
                formatted = await formatter.format_string(fmt, {name: value})
                yield "output", formatted
            except Exception as e:
                yield "output", f"Error: {e}, {value}"
        else:
            yield "output", value
            if name:
                yield "name", name
        return

    output_schema = block.output_schema.jsonschema()
    output_properties: dict[str, Any] = output_schema.get("properties", {})

    system_prompt, user_prompt = build_simulation_prompt(block, input_data)
    label = getattr(block, "name", "?")

    try:
        parsed = await _call_llm_for_simulation(system_prompt, user_prompt, label=label)

        # Track which pins were yielded so we can fill in missing required
        # ones afterwards — downstream nodes connected to unyielded pins
        # would otherwise stall in INCOMPLETE state.
        yielded_pins: set[str] = set()

        # Yield pins present in the LLM response with meaningful values.
        # We skip None and empty strings but preserve valid falsy values
        # like False, 0, and [].
        for pin_name in output_properties:
            if pin_name not in parsed:
                continue
            value = parsed[pin_name]
            if value is None or value == "":
                continue
            yield pin_name, value
            yielded_pins.add(pin_name)

        # For any required output pins the LLM omitted (excluding "error"),
        # yield a type-appropriate default so downstream nodes still fire.
        required_pins = set(output_schema.get("required", []))
        for pin_name in required_pins - yielded_pins - {"error"}:
            pin_schema = output_properties.get(pin_name, {})
            default = _default_for_schema(pin_schema)
            logger.debug(
                "simulate(%s): filling missing required pin %r with default %r",
                label,
                pin_name,
                default,
            )
            yield pin_name, default

    except (RuntimeError, ValueError) as e:
        yield "error", str(e)


def _default_for_schema(pin_schema: dict[str, Any]) -> Any:
    """Return a sensible default value for a JSON schema type."""
    pin_type = pin_schema.get("type", "string")
    if pin_type == "string":
        return ""
    if pin_type == "integer":
        return 0
    if pin_type == "number":
        return 0.0
    if pin_type == "boolean":
        return False
    if pin_type == "array":
        return []
    if pin_type == "object":
        return {}
    return ""
