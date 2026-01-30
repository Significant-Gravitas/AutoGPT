"""Agent generator package - Creates agents from natural language."""

from .core import (  # Types; Exceptions; Functions
    AgentGeneratorNotConfiguredError,
    AgentSummary,
    DecompositionResult,
    DecompositionStep,
    LibraryAgentSummary,
    MarketplaceAgentSummary,
    decompose_goal,
    enrich_library_agents_from_steps,
    extract_search_terms_from_steps,
    generate_agent,
    generate_agent_patch,
    get_agent_as_json,
    get_all_relevant_agents_for_generation,
    get_library_agents_for_generation,
    json_to_graph,
    save_agent_to_library,
    search_marketplace_agents_for_generation,
)
from .errors import get_user_message_for_error
from .service import health_check as check_external_service_health
from .service import is_external_service_configured

__all__ = [
    # Types
    "AgentSummary",
    "DecompositionResult",
    "DecompositionStep",
    "LibraryAgentSummary",
    "MarketplaceAgentSummary",
    # Core functions
    "decompose_goal",
    "generate_agent",
    "generate_agent_patch",
    "save_agent_to_library",
    "get_agent_as_json",
    "get_library_agents_for_generation",
    "get_all_relevant_agents_for_generation",
    "search_marketplace_agents_for_generation",
    "enrich_library_agents_from_steps",
    "extract_search_terms_from_steps",
    "json_to_graph",
    # Exceptions
    "AgentGeneratorNotConfiguredError",
    # Service
    "is_external_service_configured",
    "check_external_service_health",
    # Error handling
    "get_user_message_for_error",
]
