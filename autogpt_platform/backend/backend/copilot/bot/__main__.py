"""Entry point for running the CoPilot Chat Bridge service.

Usage:
    poetry run copilot-bot
    python -m backend.copilot.bot
"""

from backend.app import run_processes

from .app import CoPilotChatBridge


def main():
    """Run the CoPilot Chat Bridge service."""
    run_processes(CoPilotChatBridge())


if __name__ == "__main__":
    main()
