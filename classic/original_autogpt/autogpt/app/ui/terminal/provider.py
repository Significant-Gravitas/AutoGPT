"""Terminal-based UI provider for AutoGPT.

This provider wraps the existing terminal-based interaction behavior,
providing a seamless migration path while maintaining backward compatibility.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import click
from colorama import Fore, Style

from forge.logging.utils import print_attribute
from forge.permissions import ApprovalScope

from ..protocol import ApprovalResult, MessageLevel, UIProvider

if TYPE_CHECKING:
    from forge.models.utils import ModelWithSummary


class TerminalUIProvider(UIProvider):
    """Terminal-based UI provider using colorama and click.

    This provider maintains backward compatibility with the original
    AutoGPT terminal interface.
    """

    def __init__(self, plain_output: bool = False):
        """Initialize the terminal UI provider.

        Args:
            plain_output: If True, disable spinners and fancy output.
        """
        self.plain_output = plain_output
        self.logger = logging.getLogger(__name__)
        self._spinner = None

    @asynccontextmanager
    async def show_spinner(self, message: str) -> AsyncIterator[None]:
        """Show a spinner/loading indicator.

        Args:
            message: The message to display alongside the spinner.
        """
        from autogpt.app.spinner import Spinner

        spinner = Spinner(message, plain_output=self.plain_output)
        spinner.start()
        try:
            yield
        finally:
            spinner.stop()

    async def prompt_input(self, prompt: str, default: str = "") -> str:
        """Prompt the user for text input.

        Args:
            prompt: The prompt message to display.
            default: Default value if user just presses Enter.

        Returns:
            The user's input string.
        """
        try:
            return click.prompt(
                text=prompt, prompt_suffix=" ", default=default, show_default=False
            )
        except KeyboardInterrupt:
            self.logger.info("You interrupted AutoGPT")
            self.logger.info("Quitting...")
            raise SystemExit(0)

    async def prompt_permission(
        self, cmd: str, args_str: str, args: dict[str, Any]
    ) -> ApprovalResult:
        """Prompt user for command permission.

        Uses an interactive selector with arrow keys and Tab-to-add-context.

        Args:
            cmd: Command name.
            args_str: Formatted arguments string.
            args: Full arguments dictionary.

        Returns:
            ApprovalResult with the user's choice and optional feedback.
        """
        from ..rich_select import RichSelect

        # Map choices to approval scopes
        choices = [
            "Once",
            "Always (this agent)",
            "Always (all agents)",
            "Deny",
        ]

        scope_map = {
            0: ApprovalScope.ONCE,
            1: ApprovalScope.AGENT,
            2: ApprovalScope.WORKSPACE,
            3: ApprovalScope.DENY,
        }

        try:
            selector = RichSelect(
                choices=choices,
                title="Approve command execution?",
                subtitle=f"{cmd}({args_str})",
            )
            result = selector.run()

            scope = scope_map.get(result.index, ApprovalScope.DENY)

            # If feedback was provided (via Tab context or inline typing)
            if result.has_feedback:
                return ApprovalResult(scope=scope, feedback=result.feedback)

            return ApprovalResult(scope=scope)

        except KeyboardInterrupt:
            self.logger.info("Command approval interrupted")
            return ApprovalResult(scope=ApprovalScope.DENY)

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
        from autogpt.agents.prompt_strategies.one_shot import AssistantThoughts

        from forge.models.utils import ModelWithSummary

        thoughts_text = self._remove_ansi_escape(
            thoughts.reasoning
            if isinstance(thoughts, AssistantThoughts)
            else (
                thoughts.summary()
                if isinstance(thoughts, ModelWithSummary)
                else thoughts
            )
        )
        print_attribute(
            f"{ai_name.upper()} THOUGHTS", thoughts_text, title_color=Fore.YELLOW
        )

        if isinstance(thoughts, AssistantThoughts):
            if assistant_thoughts_plan := self._remove_ansi_escape(
                "\n".join(f"- {p}" for p in thoughts.plan)
            ):
                print_attribute("PLAN", "", title_color=Fore.YELLOW)
                # If it's a list, join it into a string
                if isinstance(assistant_thoughts_plan, list):
                    assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
                elif isinstance(assistant_thoughts_plan, dict):
                    assistant_thoughts_plan = str(assistant_thoughts_plan)

                # Split the input_string using the newline character and dashes
                lines = assistant_thoughts_plan.split("\n")
                for line in lines:
                    line = line.lstrip("- ")
                    self.logger.info(
                        line.strip(), extra={"title": "- ", "title_color": Fore.GREEN}
                    )
            print_attribute(
                "CRITICISM",
                self._remove_ansi_escape(thoughts.self_criticism),
                title_color=Fore.YELLOW,
            )

    async def display_command(self, name: str, arguments: dict[str, Any]) -> None:
        """Display the next command to be executed.

        Args:
            name: The command name.
            arguments: The command arguments.
        """
        print()
        safe_name = self._remove_ansi_escape(name)
        self.logger.info(
            f"COMMAND = {Fore.CYAN}{safe_name}{Style.RESET_ALL}  "
            f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            extra={
                "title": "NEXT ACTION:",
                "title_color": Fore.CYAN,
                "preserve_color": True,
            },
        )

    async def display_result(
        self, result: str, is_error: bool = False, title: str = "SYSTEM:"
    ) -> None:
        """Display a command result.

        Args:
            result: The result message.
            is_error: Whether this is an error result.
            title: The title to show with the result.
        """
        if is_error:
            self.logger.warning(result)
        else:
            self.logger.info(result, extra={"title": title, "title_color": Fore.YELLOW})

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
            preserve_color: Whether to preserve ANSI color codes.
        """
        extra: dict[str, Any] = {}
        if title:
            extra["title"] = title
        if preserve_color:
            extra["preserve_color"] = True

        if level == MessageLevel.DEBUG:
            self.logger.debug(message, extra=extra if extra else None)
        elif level == MessageLevel.INFO:
            self.logger.info(message, extra=extra if extra else None)
        elif level == MessageLevel.WARNING:
            self.logger.warning(message, extra=extra if extra else None)
        elif level == MessageLevel.ERROR:
            self.logger.error(message, extra=extra if extra else None)
        elif level == MessageLevel.SUCCESS:
            extra["color"] = Fore.GREEN
            self.logger.info(message, extra=extra)

    async def display_agent_selection(self, agents: list[str]) -> str:
        """Display existing agents and let user select one.

        Args:
            agents: List of existing agent IDs.

        Returns:
            The selected agent ID or empty string for new agent.
        """
        print(
            "Existing agents\n---------------\n"
            + "\n".join(f"{i} - {agent_id}" for i, agent_id in enumerate(agents, 1))
        )
        selection = await self.prompt_input(
            "Enter the number or name of the agent to run,"
            " or hit enter to create a new one:"
        )

        # Check if input is a number
        import re

        if re.match(r"^\d+$", selection.strip()):
            idx = int(selection)
            if 0 < idx <= len(agents):
                return agents[idx - 1]

        # Check if input matches an agent name
        if selection in agents:
            return selection

        return ""

    async def confirm(self, message: str, default: bool = True) -> bool:
        """Ask user for yes/no confirmation.

        Args:
            message: The confirmation prompt.
            default: Default value if user just presses Enter.

        Returns:
            True if user confirms, False otherwise.
        """
        suffix = " [Y/n]" if default else " [y/N]"
        response = await self.prompt_input(message + suffix)

        if response == "":
            return default
        return response.lower() in ("y", "yes")

    def _remove_ansi_escape(self, s: str) -> str:
        """Remove ANSI escape sequences from a string."""
        return s.replace("\x1B", "")

    async def prompt_finish_continuation(
        self,
        summary: str,
        suggested_next_task: Optional[str] = None,
    ) -> str:
        """Display task completion and prompt for next task."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Build panel content
        content = Text()
        content.append(summary, style="green")

        if suggested_next_task:
            content.append("\n\n")
            content.append("Suggested: ", style="bold yellow")
            content.append(suggested_next_task, style="italic")

        panel = Panel(
            content,
            title="[bold green]Task Completed[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        console.print()
        console.print(panel)
        console.print()

        return await self.prompt_input("Enter next task (or press Enter to exit):")
