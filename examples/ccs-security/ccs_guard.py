"""CCS Security Guard for AutoGPT code/shell execution.

Integrates CCS runtime verification into AutoGPT's CodeExecutorComponent,
providing security verification for execute_shell() and execute_python_code().

Usage:
    from examples.ccs_security.ccs_guard import CCSAutoGPTGuard

    guard = CCSAutoGPTGuard()
    # Wrap shell command before execution
    allowed, reason = guard.verify_shell("curl https://example.com")
    if allowed:
        result = code_executor.execute_shell("curl https://example.com")
"""
import logging
from typing import Optional, Tuple

try:
    from ccs_verifier import Verifier, Command
    from ccs_verifier.builtin_rules import RCERule, SSRFRule, CredentialLeakRule
    CCS_AVAILABLE = True
except ImportError:
    CCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CCSAutoGPTGuard:
    """CCS security layer for AutoGPT command execution.

    Provides in-process verification (~7.5μs P50) for shell commands
    and Python code execution in AutoGPT's CodeExecutorComponent.

    Complements AutoGPT's existing allowlist/denylist with semantic
    security analysis (intent-level detection, not just pattern matching).
    """

    def __init__(self, agent_id: str = "autogpt-agent"):
        self.agent_id = agent_id
        self._verifier: Optional[Verifier] = None

    @property
    def verifier(self) -> Optional[Verifier]:
        if not CCS_AVAILABLE:
            logger.warning("ccs-verifier not installed. Install: pip install ccs-verifier")
            return None
        if self._verifier is None:
            self._verifier = Verifier(rules=[RCERule(), SSRFRule(), CredentialLeakRule()])
        return self._verifier

    def verify_shell(self, command_line: str) -> Tuple[bool, str]:
        """Verify a shell command through CCS.

        Args:
            command_line: Shell command to verify.

        Returns:
            (allowed, reason) tuple.
        """
        if self.verifier is None:
            return True, "CCS not available"
        cmd = Command(
            agent_id=self.agent_id,
            tool="shell",
            params={"command": command_line},
        )
        result = self.verifier.verify(cmd)
        if result.verdict.value == "deny":
            reason = getattr(result, "reason", "policy violation") or "policy violation"
            logger.warning(f"[CCS] Shell command denied: {command_line[:80]} | reason={reason}")
            return False, reason
        return True, "allowed"

    def verify_python(self, code: str) -> Tuple[bool, str]:
        """Verify Python code execution through CCS.

        Args:
            code: Python code to verify.

        Returns:
            (allowed, reason) tuple.
        """
        if self.verifier is None:
            return True, "CCS not available"
        cmd = Command(
            agent_id=self.agent_id,
            tool="python_execute",
            params={"code": code},
        )
        result = self.verifier.verify(cmd)
        if result.verdict.value == "deny":
            reason = getattr(result, "reason", "policy violation") or "policy violation"
            logger.warning(f"[CCS] Python code denied | reason={reason}")
            return False, reason
        return True, "allowed"

    def patch_executor(self, executor) -> None:
        """Monkey-patch an AutoGPT CodeExecutorComponent with CCS guards.

        Args:
            executor: CodeExecutorComponent instance to patch.
        """
        guard = self
        original_validate = executor.validate_command

        def enhanced_validate(command_line: str):
            # First check CCS
            allowed, reason = guard.verify_shell(command_line)
            if not allowed:
                from forge.utils.exceptions import OperationNotAllowedError
                raise OperationNotAllowedError(f"[CCS Security] {reason}")
            # Then run original validation
            return original_validate(command_line)

        executor.validate_command = enhanced_validate
        logger.info("[CCS] AutoGPT CodeExecutorComponent patched with CCS security")
