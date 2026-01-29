"""Rich-based interactive selector with inline feedback option.

A custom selector using Rich for display and raw terminal input for interaction.
No prompt_toolkit dependency - works within existing async event loops.
"""

import sys
import termios
import tty
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class SelectionResult:
    """Result of a selection with optional feedback."""

    choice: str
    index: int
    feedback: str | None = None

    @property
    def has_feedback(self) -> bool:
        return self.feedback is not None and self.feedback.strip() != ""


def _getch() -> str:
    """Read a single character from stdin without echo."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences (arrow keys)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return f"\x1b[{ch3}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class RichSelect:
    """Interactive selector with Rich formatting and inline feedback."""

    FEEDBACK_INDEX = -1

    def __init__(
        self,
        choices: list[str],
        title: str = "Select an option",
        subtitle: str | None = None,
        default_index: int = 0,
        show_feedback_option: bool = True,
        feedback_placeholder: str = "Type feedback here...",
    ):
        self.choices = choices
        self.title = title
        self.subtitle = subtitle
        self.selected_index = default_index
        self.show_feedback_option = show_feedback_option
        self.feedback_placeholder = feedback_placeholder
        self.feedback_buffer = ""
        self.adding_context = False  # Tab mode for adding context to current selection
        self.context_buffer = ""
        self.console = Console()

    @property
    def _total_options(self) -> int:
        return len(self.choices) + (1 if self.show_feedback_option else 0)

    @property
    def _on_feedback_option(self) -> bool:
        return self.show_feedback_option and self.selected_index == len(self.choices)

    def _render(self) -> str:
        """Render the selector as a plain string."""
        lines = []

        # Regular choices
        for i, choice in enumerate(self.choices):
            if i == self.selected_index:
                if self.adding_context:
                    # Show choice + context being typed
                    ctx = self.context_buffer
                    lines.append(
                        f"  \033[1;32m❯ {choice}\033[0m \033[33m+ {ctx}\033[5m█\033[0m"
                    )
                else:
                    lines.append(f"  \033[1;32m❯ {choice}\033[0m")
            else:
                lines.append(f"    \033[2m{choice}\033[0m")

        # Feedback option - inline text input
        if self.show_feedback_option:
            feedback_idx = len(self.choices)
            if self.selected_index == feedback_idx:
                if self.feedback_buffer:
                    # Show typed text with cursor
                    lines.append(f"  \033[1;33m❯ {self.feedback_buffer}\033[5m█\033[0m")
                else:
                    # Show placeholder as shadow text with cursor
                    ph = self.feedback_placeholder
                    lines.append(f"  \033[1;33m❯ \033[2;33m{ph}\033[0;5;33m█\033[0m")
            else:
                if self.feedback_buffer:
                    # Show typed text (not selected)
                    lines.append(f"    \033[2;33m{self.feedback_buffer}\033[0m")
                else:
                    # Show placeholder (not selected)
                    lines.append(f"    \033[2m{self.feedback_placeholder}\033[0m")

        # Help text
        lines.append("")
        # ANSI: \033[2m=dim, \033[1;36m=bold cyan, \033[0;2m=reset+dim, \033[0m=reset
        if self.adding_context:
            lines.append(
                "  \033[2mType context, \033[1;36mEnter\033[0;2m confirm, "
                "\033[1;36mEsc\033[0;2m cancel\033[0m"
            )
        elif self._on_feedback_option:
            lines.append(
                "  \033[1;36m↑↓\033[0;2m move  \033[1;36mEnter\033[0;2m send  "
                "\033[1;36mEsc\033[0;2m clear  \033[2mjust start typing...\033[0m"
            )
        else:
            lines.append(
                "  \033[1;36m↑↓\033[0;2m move  \033[1;36mEnter\033[0;2m select  "
                "\033[1;36mTab\033[0;2m +context  \033[1;36m1-5\033[0;2m quick\033[0m"
            )

        return "\n".join(lines)

    def _clear_lines(self, n: int):
        """Clear n lines above cursor."""
        for _ in range(n):
            sys.stdout.write("\033[A")  # Move up
            sys.stdout.write("\033[2K")  # Clear line
        sys.stdout.flush()

    def run(self) -> SelectionResult:
        """Run the interactive selector."""
        # Print header with Rich
        header = Text()
        header.append(f"{self.title}", style="bold cyan")
        if self.subtitle:
            header.append(f"\n{self.subtitle}", style="dim")

        self.console.print()
        self.console.print(Panel(header, border_style="cyan", padding=(0, 1)))
        self.console.print()

        num_lines = self._total_options + 2  # options + blank + help

        # Initial render
        output = self._render()
        print(output, flush=True)

        while True:
            ch = _getch()

            # Handle context input mode (Tab on regular option)
            if self.adding_context:
                if ch == "\r" or ch == "\n":  # Enter - confirm with context
                    self._clear_lines(num_lines)
                    choice = self.choices[self.selected_index]
                    context = (
                        self.context_buffer if self.context_buffer.strip() else None
                    )
                    if context:
                        result_text = (
                            f"  \033[1;32m✓\033[0m \033[32m{choice}\033[0m "
                            f"\033[33m+ {context}\033[0m"
                        )
                    else:
                        result_text = f"  \033[1;32m✓\033[0m \033[32m{choice}\033[0m"
                    print(result_text)
                    print()
                    return SelectionResult(
                        choice=choice, index=self.selected_index, feedback=context
                    )
                elif ch == "\x1b":  # Escape - cancel context
                    self.adding_context = False
                    self.context_buffer = ""
                elif ch == "\x7f" or ch == "\x08":  # Backspace
                    self.context_buffer = self.context_buffer[:-1]
                elif ch == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt()
                elif ch.isprintable():
                    self.context_buffer += ch

            # Navigation (when not in context mode)
            elif ch == "\x1b[A":  # Up arrow
                self.selected_index = (self.selected_index - 1) % self._total_options
            elif ch == "\x1b[B":  # Down arrow
                self.selected_index = (self.selected_index + 1) % self._total_options
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt()

            # Tab - add context to current selection (not on feedback option)
            elif ch == "\t" and not self._on_feedback_option:
                self.adding_context = True
                self.context_buffer = ""

            # Enter key
            elif ch == "\r" or ch == "\n":
                if self._on_feedback_option and self.feedback_buffer.strip():
                    # Submit feedback
                    self._clear_lines(num_lines)
                    fb = self.feedback_buffer
                    result_text = f"  \033[1;32m✓\033[0m \033[33mFeedback: {fb}\033[0m"
                    print(result_text)
                    print()
                    return SelectionResult(
                        choice="feedback",
                        index=self.FEEDBACK_INDEX,
                        feedback=self.feedback_buffer,
                    )
                elif not self._on_feedback_option:
                    # Select regular option
                    self._clear_lines(num_lines)
                    choice = self.choices[self.selected_index]
                    result_text = f"  \033[1;32m✓\033[0m \033[32m{choice}\033[0m"
                    print(result_text)
                    print()
                    return SelectionResult(
                        choice=choice,
                        index=self.selected_index,
                        feedback=None,
                    )
                # On feedback option with no text - do nothing (need to type something)

            # Escape key
            elif ch == "\x1b":
                if self._on_feedback_option and self.feedback_buffer:
                    # Clear feedback buffer
                    self.feedback_buffer = ""
                else:
                    # Exit with first option
                    self._clear_lines(num_lines)
                    choice = self.choices[0]
                    result_text = f"  \033[1;32m✓\033[0m \033[32m{choice}\033[0m"
                    print(result_text)
                    print()
                    return SelectionResult(choice=choice, index=0, feedback=None)

            # Quick select numbers
            elif ch in "12345" and not self._on_feedback_option:
                idx = int(ch) - 1
                if idx < len(self.choices):
                    self._clear_lines(num_lines)
                    choice = self.choices[idx]
                    result_text = f"  \033[1;32m✓\033[0m \033[32m{choice}\033[0m"
                    print(result_text)
                    print()
                    return SelectionResult(choice=choice, index=idx, feedback=None)
                elif idx == len(self.choices) and self.show_feedback_option:
                    # Jump to feedback option
                    self.selected_index = idx

            # Backspace (when on feedback option)
            elif (ch == "\x7f" or ch == "\x08") and self._on_feedback_option:
                self.feedback_buffer = self.feedback_buffer[:-1]

            # Printable character - if on feedback option, type directly
            elif ch.isprintable():
                if self._on_feedback_option:
                    self.feedback_buffer += ch

            # Re-render
            self._clear_lines(num_lines)
            output = self._render()
            print(output, flush=True)


def select(
    choices: list[str],
    title: str = "Select an option",
    subtitle: str | None = None,
    default_index: int = 0,
    show_feedback_option: bool = True,
) -> SelectionResult:
    """Convenience function to run a selection."""
    selector = RichSelect(
        choices=choices,
        title=title,
        subtitle=subtitle,
        default_index=default_index,
        show_feedback_option=show_feedback_option,
    )
    return selector.run()


if __name__ == "__main__":
    result = select(
        choices=["Once", "Always (this agent)", "Always (all agents)", "Deny"],
        title="Approve command execution?",
        subtitle="execute_python_code(code='print(hello)')",
    )
    print(f"Result: choice={result.choice}, index={result.index}")
    if result.has_feedback:
        print(f"Feedback: {result.feedback}")
