"""Build a CoPilot prompt from a WorkflowDescription.

Instead of a custom single-shot LLM conversion, we generate a structured
prompt that CoPilot's existing agentic agent-generator handles. This reuses
the multi-turn tool-use pipeline (find_block, create_agent, fixer, validator)
for reliable workflow-to-agent conversion.
"""

import json

from .models import WorkflowDescription


def build_copilot_prompt(desc: WorkflowDescription) -> str:
    """Build a CoPilot prompt from a parsed WorkflowDescription.

    The prompt describes the external workflow in enough detail for CoPilot's
    agent-generator to recreate it as an AutoGPT agent graph.

    Args:
        desc: Structured description of the source workflow.

    Returns:
        A user-facing prompt string for CoPilot.
    """
    steps_text = ""
    for step in desc.steps:
        conns = (
            f" → connects to steps {step.connections_to}" if step.connections_to else ""
        )
        params_str = ""
        if step.parameters:
            truncated = json.dumps(step.parameters, default=str)[:300]
            params_str = f" (params: {truncated})"
        steps_text += (
            f"  {step.order}. [{step.service}] {step.action}{params_str}{conns}\n"
        )

    trigger_line = f"Trigger: {desc.trigger_type}" if desc.trigger_type else ""

    return f"""I want to import a workflow from {desc.source_format.value} and recreate it as an AutoGPT agent.

**Workflow name**: {desc.name}
**Description**: {desc.description}
{trigger_line}

**Steps** (from the original {desc.source_format.value} workflow):
{steps_text}
Please build an AutoGPT agent that replicates this workflow. Map each step to the most appropriate AutoGPT block(s), wire them together, and save it.""".strip()
