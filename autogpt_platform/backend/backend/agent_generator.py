"""Agent Generator process for running the agent-generator service as a submodule."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.util.process import AppProcess

logger = logging.getLogger(__name__)

# Path to agent-generator submodule: autogpt_platform/agent-generator
# __file__ = .../autogpt_platform/backend/backend/agent_generator.py
# .parent.parent.parent = .../autogpt_platform
AGENT_GENERATOR_SUBMODULE_PATH = Path(__file__).parent.parent.parent / "agent-generator"


def is_agent_generator_available() -> bool:
    """Check if the agent-generator submodule is present and initialized."""
    pyproject_path = AGENT_GENERATOR_SUBMODULE_PATH / "pyproject.toml"
    return pyproject_path.exists()


class AgentGeneratorProcess(AppProcess):
    """
    Process to run the agent-generator service from the submodule.

    This allows running the agent-generator alongside the main AutoGPT backend
    for local development and testing.

    Prerequisites:
        cd autogpt_platform/agent-generator && poetry install
    """

    _subprocess: subprocess.Popen | None = None

    @property
    def service_name(self) -> str:
        return "AgentGenerator"

    def run(self):
        """Run the agent-generator service."""
        if not is_agent_generator_available():
            logger.warning(
                f"[{self.service_name}] Submodule not found at {AGENT_GENERATOR_SUBMODULE_PATH}. "
                "Run 'git submodule update --init autogpt_platform/agent-generator' to initialize."
            )
            return

        logger.info(f"[{self.service_name}] Starting agent-generator service...")

        # Try to run using the submodule's venv first, fall back to poetry
        venv_python = AGENT_GENERATOR_SUBMODULE_PATH / ".venv" / "bin" / "python"

        env = os.environ.copy()
        env.setdefault("PORT", "8009")
        env.setdefault("HOST", "0.0.0.0")

        if venv_python.exists():
            # Run directly with venv python
            self._subprocess = subprocess.Popen(
                [
                    str(venv_python),
                    "-m",
                    "autogpt_agent_generator.app",
                ],
                cwd=AGENT_GENERATOR_SUBMODULE_PATH,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        else:
            # Fall back to poetry (will install deps if needed)
            logger.info(f"[{self.service_name}] No venv found, using poetry run...")
            self._subprocess = subprocess.Popen(
                ["poetry", "run", "app"],
                cwd=AGENT_GENERATOR_SUBMODULE_PATH,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

        # Wait for the process to complete
        self._subprocess.wait()

    def cleanup(self):
        """Terminate the agent-generator subprocess."""
        if self._subprocess and self._subprocess.poll() is None:
            logger.info(f"[{self.service_name}] Terminating subprocess...")
            self._subprocess.terminate()
            try:
                self._subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.service_name}] Force killing subprocess...")
                self._subprocess.kill()
