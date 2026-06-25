"""Entry point for running the CoPilot Executor service.

Usage:
    python -m backend.copilot.executor
"""

from backend.app import run_processes

from .manager import CoPilotExecutor


def main():
    """Run the CoPilot Executor service."""
    run_processes(CoPilotExecutor())


if __name__ == "__main__":
    main()
