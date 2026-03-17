"""LLM-powered conversion of external workflows to AutoGPT agent graphs.

Uses the CoPilot's LLM client to generate AutoGPT agent JSON from a structured
WorkflowDescription, then validates and fixes via the existing pipeline.
"""

import functools
import json
import logging
import pathlib
import threading
from typing import Any

from backend.copilot.config import ChatConfig
from backend.copilot.tools.agent_generator.blocks import get_blocks_as_dicts

from .models import WorkflowDescription

logger = logging.getLogger(__name__)

_AGENT_GUIDE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "sdk" / "agent_generation_guide.md"
)

_MAX_RETRIES = 1

# Cached LLM client — created once on first use, guarded by lock
_llm_client: Any = None
_llm_config: ChatConfig | None = None
_llm_lock = threading.Lock()


def _get_llm_client() -> tuple[Any, ChatConfig]:
    """Return a cached LangfuseAsyncOpenAI client (thread-safe)."""
    global _llm_client, _llm_config
    if _llm_client is not None:
        assert _llm_config is not None
        return _llm_client, _llm_config

    with _llm_lock:
        if _llm_client is None:
            from langfuse.openai import (
                AsyncOpenAI as LangfuseAsyncOpenAI,  # pyright: ignore[reportPrivateImportUsage]
            )

            _llm_config = ChatConfig()
            _llm_client = LangfuseAsyncOpenAI(
                api_key=_llm_config.api_key, base_url=_llm_config.base_url
            )
    assert _llm_config is not None
    return _llm_client, _llm_config


@functools.lru_cache(maxsize=1)
def _load_agent_guide() -> str:
    """Load the agent generation guide markdown (cached after first read)."""
    return _AGENT_GUIDE_PATH.read_text()


_MAX_CATALOG_BLOCKS = 200
_MAX_CATALOG_CHARS = 50_000


def _build_block_catalog(blocks: list[dict[str, Any]]) -> str:
    """Build a compact block catalog string for the LLM prompt.

    Caps at _MAX_CATALOG_BLOCKS entries / _MAX_CATALOG_CHARS to avoid
    exceeding the LLM context window.
    """
    lines: list[str] = []
    total_len = 0
    for b in blocks[:_MAX_CATALOG_BLOCKS]:
        desc = (b.get("description") or "")[:200]
        line = f"- **{b['name']}** (id: `{b['id']}`): {desc}"
        total_len += len(line) + 1
        if total_len > _MAX_CATALOG_CHARS:
            lines.append(f"... and {len(blocks) - len(lines)} more blocks")
            break
        lines.append(line)
    return "\n".join(lines)


def _build_conversion_prompt(
    desc: WorkflowDescription,
    block_catalog: str,
    agent_guide: str,
    error_feedback: str | None = None,
) -> list[dict[str, str]]:
    """Build the messages for the LLM conversion call."""
    steps_text = ""
    for step in desc.steps:
        conns = (
            f" -> connects to steps {step.connections_to}"
            if step.connections_to
            else ""
        )
        params_str = (
            f" (params: {json.dumps(step.parameters, default=str)[:300]})"
            if step.parameters
            else ""
        )
        steps_text += (
            f"  {step.order}. [{step.service}] {step.action}{params_str}{conns}\n"
        )

    system_msg = f"""You are an expert at converting automation workflows into AutoGPT agent graphs.

Your task: Convert the workflow described below into a valid AutoGPT agent JSON.

## Agent Generation Guide
{agent_guide}

## Available AutoGPT Blocks
{block_catalog}

## Instructions
1. Map each workflow step to the most appropriate AutoGPT block(s)
2. If no exact block match exists, use the closest alternative (e.g., HttpRequestBlock for generic API calls)
3. Every agent MUST have at least one AgentInputBlock and one AgentOutputBlock
4. Wire blocks together with links matching the original workflow's data flow
5. Set meaningful input_default values based on the workflow's parameters
6. Position nodes with 800+ X-unit spacing
7. Return ONLY valid JSON — no markdown fences, no explanation"""

    user_msg = f"""Convert this {desc.source_format.value} workflow to an AutoGPT agent:

**Name**: {desc.name}
**Description**: {desc.description}
**Trigger**: {desc.trigger_type or 'Manual'}

**Steps**:
{steps_text}

Generate the complete AutoGPT agent JSON with nodes and links."""

    if error_feedback:
        user_msg += f"""

IMPORTANT: Your previous attempt had validation errors. Fix them:
{error_feedback}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


async def convert_workflow(
    desc: WorkflowDescription,
) -> tuple[dict[str, Any], list[str]]:
    """Convert a WorkflowDescription into an AutoGPT agent JSON.

    Args:
        desc: Structured description of the source workflow.

    Returns:
        Tuple of (agent_json dict, conversion_notes list).

    Raises:
        ValueError: If conversion fails after retries.
    """
    client, config = _get_llm_client()

    blocks = get_blocks_as_dicts()
    block_catalog = _build_block_catalog(blocks)
    agent_guide = _load_agent_guide()
    conversion_notes: list[str] = []

    error_feedback: str | None = None

    for attempt in range(_MAX_RETRIES + 1):
        messages = _build_conversion_prompt(
            desc, block_catalog, agent_guide, error_feedback
        )

        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                max_tokens=8192,
            )
        except Exception as e:
            raise ValueError(f"LLM call failed: {e}") from e

        if not response.choices:
            raise ValueError("LLM returned no choices")
        raw_content = response.choices[0].message.content or ""

        # Strip markdown fences if present
        content = raw_content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove opening fence line (e.g. ```json)
            lines = lines[1:]
            # Find closing fence and truncate everything after it
            for idx, line in enumerate(lines):
                if line.strip() == "```":
                    lines = lines[:idx]
                    break
            content = "\n".join(lines)

        try:
            agent_json = json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < _MAX_RETRIES:
                error_feedback = f"Invalid JSON: {e}"
                conversion_notes.append(
                    f"Retry {attempt + 1}: LLM output was not valid JSON"
                )
                continue
            raise ValueError(
                f"LLM produced invalid JSON after {_MAX_RETRIES + 1} attempts: {e}"
            ) from e

        # Set metadata
        agent_json.setdefault("name", desc.name)
        agent_json.setdefault(
            "description",
            f"Imported from {desc.source_format.value}: {desc.description}",
        )
        agent_json.setdefault("version", 1)
        agent_json.setdefault("is_active", True)

        # Auto-fix (lazy import to avoid heavy module load at import time)
        try:
            from backend.copilot.tools.agent_generator.fixer import AgentFixer

            fixer = AgentFixer()
            agent_json = fixer.apply_all_fixes(agent_json, blocks)
            fixes = fixer.get_fixes_applied()
            if fixes:
                conversion_notes.append(f"Applied {len(fixes)} auto-fixes")
                logger.info("Applied %d auto-fixes to imported agent", len(fixes))
        except Exception as e:
            logger.warning("Auto-fix failed: %s", e)
            conversion_notes.append(f"Auto-fix warning: {e}")

        # Validate (lazy import to avoid heavy module load at import time)
        try:
            from backend.copilot.tools.agent_generator.validator import AgentValidator

            validator = AgentValidator()
            is_valid, _ = validator.validate(agent_json, blocks)
            if not is_valid:
                errors = validator.errors
                if attempt < _MAX_RETRIES:
                    error_feedback = "\n".join(f"- {e}" for e in errors[:5])
                    conversion_notes.append(
                        f"Retry {attempt + 1}: validation errors found"
                    )
                    continue
                # On final attempt, return with warnings rather than failing
                conversion_notes.extend(f"Validation warning: {e}" for e in errors[:5])
                conversion_notes.append("Agent may need manual fixes in the builder")
        except Exception as e:
            logger.warning("Validation exception: %s", e)
            conversion_notes.append(f"Validation could not complete: {e}")

        return agent_json, conversion_notes

    raise ValueError("Conversion failed after all retries")
