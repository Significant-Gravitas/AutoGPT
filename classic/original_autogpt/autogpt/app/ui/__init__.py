"""UI providers for AutoGPT.

This package provides UI abstractions for the AutoGPT interaction loop,
using Rich for beautiful terminal output.
"""

from .protocol import ApprovalResult, UIProvider

__all__ = ["UIProvider", "ApprovalResult", "create_ui_provider"]


def create_ui_provider(plain_output: bool = False) -> UIProvider:
    """Create a UI provider.

    Args:
        plain_output: Whether to use plain output (no spinners/colors).

    Returns:
        A UIProvider instance.
    """
    from .terminal.provider import TerminalUIProvider

    return TerminalUIProvider(plain_output=plain_output)
