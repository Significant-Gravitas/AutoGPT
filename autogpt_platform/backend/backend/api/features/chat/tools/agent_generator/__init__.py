"""Agent generator package - Creates agents from natural language."""

from .core import (
    AgentGeneratorNotConfiguredError,
    decompose_goal,
    generate_agent,
    generate_agent_patch,
    get_agent_as_json,
    json_to_graph,
    save_agent_to_library,
)
from .service import health_check as check_external_service_health
from .service import is_external_service_configured

__all__ = [
    # Core functions
    "decompose_goal",
    "generate_agent",
    "generate_agent_patch",
    "save_agent_to_library",
    "get_agent_as_json",
    "json_to_graph",
    # Exceptions
    "AgentGeneratorNotConfiguredError",
    # Service
    "is_external_service_configured",
    "check_external_service_health",
]
