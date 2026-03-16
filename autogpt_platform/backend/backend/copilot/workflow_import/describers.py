"""Extract structured WorkflowDescription from competitor workflow JSONs.

Each describer is a pure function that deterministically parses the competitor
format into a platform-agnostic WorkflowDescription. No LLM calls are made here.
"""

import re
from typing import Any

from .models import CompetitorFormat, StepDescription, WorkflowDescription


def describe_workflow(
    json_data: dict[str, Any], fmt: CompetitorFormat
) -> WorkflowDescription:
    """Route to the appropriate describer based on detected format."""
    describers = {
        CompetitorFormat.N8N: describe_n8n_workflow,
        CompetitorFormat.MAKE: describe_make_workflow,
        CompetitorFormat.ZAPIER: describe_zapier_workflow,
    }
    describer = describers.get(fmt)
    if not describer:
        raise ValueError(f"No describer available for format: {fmt}")
    return describer(json_data)


def describe_n8n_workflow(json_data: dict[str, Any]) -> WorkflowDescription:
    """Extract a structured description from an n8n workflow JSON."""
    nodes = json_data.get("nodes", [])
    connections = json_data.get("connections", {})

    # Build node index by name for connection resolution
    node_index: dict[str, int] = {}
    steps: list[StepDescription] = []

    for i, node in enumerate(nodes):
        node_name = node.get("name", f"Node {i}")
        node_index[node_name] = i

        node_type = node.get("type", "unknown")
        # Extract service name from type (e.g., "n8n-nodes-base.gmail" -> "Gmail")
        service = _extract_n8n_service(node_type)

        # Build action description from type and parameters
        params = node.get("parameters", {})
        action = _describe_n8n_action(node_type, node_name, params)

        # Extract key parameters (skip large/internal ones)
        clean_params = _clean_params(params)

        steps.append(
            StepDescription(
                order=i,
                action=action,
                service=service,
                parameters=clean_params,
                connections_to=[],  # filled below
            )
        )

    # Resolve connections: n8n format is {NodeName: {main: [[{node, type, index}]]}}
    for source_name, conn_data in connections.items():
        source_idx = node_index.get(source_name)
        if source_idx is None:
            continue
        main_outputs = conn_data.get("main", [])
        for output_group in main_outputs:
            if not isinstance(output_group, list):
                continue
            for conn in output_group:
                target_name = conn.get("node")
                target_idx = node_index.get(target_name)
                if target_idx is not None:
                    steps[source_idx].connections_to.append(target_idx)

    # Detect trigger type
    trigger_type = None
    if nodes:
        first_type = nodes[0].get("type", "")
        if "trigger" in first_type.lower() or "webhook" in first_type.lower():
            trigger_type = _extract_n8n_service(first_type)

    return WorkflowDescription(
        name=json_data.get("name", "Imported n8n Workflow"),
        description=_build_workflow_summary(steps),
        steps=steps,
        trigger_type=trigger_type,
        source_format=CompetitorFormat.N8N,
        raw_json=json_data,
    )


def describe_make_workflow(json_data: dict[str, Any]) -> WorkflowDescription:
    """Extract a structured description from a Make.com scenario blueprint."""
    flow = json_data.get("flow", [])
    steps: list[StepDescription] = []

    for i, module in enumerate(flow):
        module_ref = module.get("module", "unknown:unknown")
        parts = module_ref.split(":", 1)
        service = parts[0].replace("-", " ").title() if parts else "Unknown"
        action_verb = parts[1] if len(parts) > 1 else "process"

        # Build human-readable action
        action = f"{action_verb.replace(':', ' ').title()} via {service}"

        params = module.get("mapper", module.get("parameters", {}))
        clean_params = _clean_params(params) if isinstance(params, dict) else {}

        # Make.com flows are sequential by default; each step connects to next
        connections_to = [i + 1] if i < len(flow) - 1 else []

        # Check for routes (branching)
        routes = module.get("routes", [])
        if routes:
            # Routes contain nested flows; flatten as additional steps
            for route in routes:
                route_flow = route.get("flow", [])
                if route_flow:
                    next_idx = len(steps) + len(flow)
                    connections_to.append(next_idx)

        steps.append(
            StepDescription(
                order=i,
                action=action,
                service=service,
                parameters=clean_params,
                connections_to=connections_to,
            )
        )

    # Detect trigger
    trigger_type = None
    if flow:
        first_module = flow[0].get("module", "")
        if "watch" in first_module.lower() or "trigger" in first_module.lower():
            trigger_type = first_module.split(":")[0].replace("-", " ").title()

    return WorkflowDescription(
        name=json_data.get("name", "Imported Make.com Scenario"),
        description=_build_workflow_summary(steps),
        steps=steps,
        trigger_type=trigger_type,
        source_format=CompetitorFormat.MAKE,
        raw_json=json_data,
    )


def describe_zapier_workflow(json_data: dict[str, Any]) -> WorkflowDescription:
    """Extract a structured description from a Zapier Zap JSON."""
    zap_steps = json_data.get("steps", [])
    steps: list[StepDescription] = []

    for i, step in enumerate(zap_steps):
        app = step.get("app", "Unknown")
        action = step.get("action", "process")
        action_desc = f"{action.replace('_', ' ').title()} via {app}"

        params = step.get("params", step.get("inputFields", {}))
        clean_params = _clean_params(params) if isinstance(params, dict) else {}

        # Zapier zaps are linear: each step connects to next
        connections_to = [i + 1] if i < len(zap_steps) - 1 else []

        steps.append(
            StepDescription(
                order=i,
                action=action_desc,
                service=app,
                parameters=clean_params,
                connections_to=connections_to,
            )
        )

    trigger_type = None
    if zap_steps:
        trigger_type = zap_steps[0].get("app")

    return WorkflowDescription(
        name=json_data.get("name", json_data.get("title", "Imported Zapier Zap")),
        description=_build_workflow_summary(steps),
        steps=steps,
        trigger_type=trigger_type,
        source_format=CompetitorFormat.ZAPIER,
        raw_json=json_data,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_n8n_service(node_type: str) -> str:
    """Extract a human-readable service name from an n8n node type.

    Examples:
        "n8n-nodes-base.gmail" -> "Gmail"
        "@n8n/n8n-nodes-langchain.agent" -> "Langchain Agent"
        "n8n-nodes-base.httpRequest" -> "Http Request"
    """
    # Strip common prefixes
    name = node_type
    for prefix in ("n8n-nodes-base.", "@n8n/n8n-nodes-langchain.", "@n8n/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Convert camelCase to Title Case
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.replace(".", " ").replace("-", " ").title()


def _describe_n8n_action(node_type: str, node_name: str, params: dict[str, Any]) -> str:
    """Build a human-readable action description for an n8n node."""
    service = _extract_n8n_service(node_type)
    resource = params.get("resource", "")
    operation = params.get("operation", "")

    if resource and operation:
        return f"{operation.title()} {resource} via {service}"
    if operation:
        return f"{operation.title()} via {service}"
    return f"{node_name} ({service})"


def _clean_params(params: dict[str, Any], max_keys: int = 10) -> dict[str, Any]:
    """Extract key parameters, skipping large or internal values."""
    cleaned: dict[str, Any] = {}
    for key, value in list(params.items())[:max_keys]:
        if key.startswith("_") or key in ("credentials", "webhookId"):
            continue
        if isinstance(value, str) and len(value) > 500:
            cleaned[key] = value[:500] + "..."
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and len(value) <= 5:
            cleaned[key] = value
    return cleaned


def _build_workflow_summary(steps: list[StepDescription]) -> str:
    """Build a one-line summary of the workflow from its steps."""
    if not steps:
        return "Empty workflow"
    services = []
    for s in steps:
        if s.service not in services:
            services.append(s.service)
    service_chain = " -> ".join(services[:6])
    if len(services) > 6:
        service_chain += f" (and {len(services) - 6} more)"
    return f"Workflow with {len(steps)} steps: {service_chain}"
