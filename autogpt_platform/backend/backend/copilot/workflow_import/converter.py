"""Build an AutoPilot prompt from a WorkflowDescription.

Instead of a custom single-shot LLM conversion, we generate a structured
prompt that AutoPilot's existing agentic agent-generator handles. This reuses
the multi-turn tool-use pipeline (find_block, create_agent, fixer, validator)
for reliable workflow-to-agent conversion.
"""

from .models import WorkflowDescription


def build_copilot_prompt(desc: WorkflowDescription) -> str:
    """Build an AutoPilot prompt from a parsed WorkflowDescription.

    Args:
        desc: Structured description of the source workflow.

    Returns:
        A user-facing prompt string for AutoPilot.
    """
    steps_lines: list[str] = []
    for step in desc.steps:
        line = f"  {step.order}. [{step.service}] {step.action}"
        if step.typed_connections:
            conn_parts = []
            for tc in step.typed_connections:
                conn_parts.append(f"step {tc.target_step} ({tc.connection_type})")
            line += f" → {', '.join(conn_parts)}"
        steps_lines.append(line)
    steps_text = "\n".join(steps_lines)

    trigger_line = f"\nTrigger: {desc.trigger_type}" if desc.trigger_type else ""

    return f"""I want to import a workflow from {desc.source_format.value} and recreate it as an AutoGPT agent.

**Workflow name**: {desc.name}
**Description**: {desc.description}
{trigger_line}

**Functional steps** (non-functional nodes like sticky notes have been filtered out):
{steps_text}

Please build an AutoGPT agent that replicates this workflow. Map each step to the most appropriate AutoGPT block(s), wire them together, and save it.""".strip()
