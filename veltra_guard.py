"""
Veltra AI - Pre-Execution Security Guardrail Integration for AutoGPT
Sub-3ms execution firewall for agent loops.
"""

import logging
try:
    from veltra import VeltraGuard
    VELTRA_AVAILABLE = True
except ImportError:
    VELTRA_AVAILABLE = False

logger = logging.getLogger(__name__)

class AutoGPTSecurityGuard:
    def __init__(self, api_key: str = None):
        if not VELTRA_AVAILABLE:
            logger.warning("Veltra SDK not installed. Run 'pip install veltra-agent'")
            self.guard = None
        else:
            self.guard = VeltraGuard(api_key=api_key)

    def validate_command(self, payload: str) -> tuple[bool, str]:
        """
        Validates shell/Python payloads before AutoGPT execution.
        Returns (is_safe, reason).
        """
        if not self.guard:
            return True, "Veltra guardrail disabled."
            
        # Fast sub-3ms local execution check
        is_safe, reason = self.guard.check(payload)
        if not is_safe:
            logger.warning(f"[Veltra Security Block]: {reason}")
        return is_safe, reason
