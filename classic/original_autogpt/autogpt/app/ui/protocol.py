"""Protocol defining the UI provider interface for AutoGPT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from forge.permissions import ApprovalScope

if TYPE_CHECKING:
    from forge.models.utils import ModelWithSummary


class MessageLevel(str, Enum):
    """Log message severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ApprovalResult:
    """Result of a command approval prompt.

    Attributes:
        scope: The approval scope chosen by the user.
        feedback: Optional user feedback if they typed text instead.
    """

    scope: ApprovalScope
    feedback: str | None = None


class UIProvider(ABC):
    """Abstract base class for UI providers.

    UI providers handle all user interaction in the AutoGPT interaction loop,
    including displaying thoughts, prompting for input, and showing results.
    """

    @abstractmethod
    @asynccontextmanager
    async def show_spinner(self, message: str) -> AsyncIterator[None]:
        """Show a spinner/loading indicator.

        Args:
            message: The message to display alongside the spinner.

        Yields:
            None
        """
        yield

    @abstractmethod
    async def prompt_input(self, prompt: str, default: str = "") -> str:
        """Prompt the user for text input.

        Args:
            prompt: The prompt message to display.
            default: Default value if user just presses Enter.

        Returns:
            The user's input string.
        """
        pass

    @abstractmethod
    async def prompt_permission(
        self, cmd: str, args_str: str, args: dict[str, Any]
    ) -> ApprovalResult:
        """Prompt user for command permission.

        Args:
            cmd: Command name.
            args_str: Formatted arguments string.
            args: Full arguments dictionary.

        Returns:
            ApprovalResult with the user's choice and optional feedback.
        """
        pass

    @abstractmethod
    async def display_thoughts(
        self,
        ai_name: str,
        thoughts: "str | ModelWithSummary",
        speak_mode: bool = False,
    ) -> None:
        """Display the agent's thoughts.

        Args:
            ai_name: The name of the AI agent.
            thoughts: The agent's thoughts (string or structured).
            speak_mode: Whether to use text-to-speech.
        """
        pass

    @abstractmethod
    async def display_command(self, name: str, arguments: dict[str, Any]) -> None:
        """Display the next command to be executed.

        Args:
            name: The command name.
            arguments: The command arguments.
        """
        pass

    @abstractmethod
    async def display_result(
        self, result: str, is_error: bool = False, title: str = "SYSTEM:"
    ) -> None:
        """Display a command result.

        Args:
            result: The result message.
            is_error: Whether this is an error result.
            title: The title to show with the result.
        """
        pass

    @abstractmethod
    async def display_message(
        self,
        message: str,
        level: MessageLevel = MessageLevel.INFO,
        title: str | None = None,
        preserve_color: bool = False,
    ) -> None:
        """Display a general message.

        Args:
            message: The message content.
            level: The message severity level.
            title: Optional title/prefix for the message.
            preserve_color: Whether to preserve ANSI color codes in the message.
        """
        pass

    @abstractmethod
    async def display_agent_selection(self, agents: list[str]) -> str:
        """Display existing agents and let user select one.

        Args:
            agents: List of existing agent IDs.

        Returns:
            The selected agent ID or empty string for new agent.
        """
        pass

    @abstractmethod
    async def confirm(self, message: str, default: bool = True) -> bool:
        """Ask user for yes/no confirmation.

        Args:
            message: The confirmation prompt.
            default: Default value if user just presses Enter.

        Returns:
            True if user confirms, False otherwise.
        """
        pass

    @abstractmethod
    async def prompt_finish_continuation(
        self,
        summary: str,
        suggested_next_task: Optional[str] = None,
    ) -> str:
        """Display task completion and prompt for next task.

        Args:
            summary: The completion summary from the agent.
            suggested_next_task: Optional suggested follow-up task.

        Returns:
            User's input for next task, or empty string to exit.
        """
        pass

    async def startup(self) -> None:
        """Called when the UI is starting up.

        Override to perform any initialization.
        """
        pass

    async def shutdown(self) -> None:
        """Called when the UI is shutting down.

        Override to perform any cleanup.
        """
        pass
