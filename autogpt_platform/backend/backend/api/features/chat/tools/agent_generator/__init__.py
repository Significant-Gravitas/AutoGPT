"""Agent generator package - Creates agents from natural language."""

from .core import (
    apply_agent_patch,
    decompose_goal,
    generate_agent,
    generate_agent_patch,
    get_agent_as_json,
    save_agent_to_library,
)
from .fixer import apply_all_fixes
from .service import health_check as check_external_service_health
from .service import is_external_service_configured
from .utils import get_blocks_info
from .validator import validate_agent

__all__ = [
    # Core functions
    "decompose_goal",
    "generate_agent",
    "generate_agent_patch",
    "apply_agent_patch",
    "save_agent_to_library",
    "get_agent_as_json",
    # Fixer
    "apply_all_fixes",
    # Validator
    "validate_agent",
    # Utils
    "get_blocks_info",
    # Service
    "is_external_service_configured",
    "check_external_service_health",
]
