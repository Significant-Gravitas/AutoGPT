"""Main Settings UI class - tabbed settings browser."""

from __future__ import annotations

import sys
import termios
import tty
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .categories import CATEGORIES
from .env_file import get_default_env_path, load_env_file, save_env_file
from .introspection import get_complete_settings
from .validators import validate_setting
from .widgets import (
    prompt_boolean,
    prompt_float,
    prompt_numeric,
    prompt_secret_input,
    prompt_selection,
    prompt_text_input,
)


def _getch() -> str:
    """Read a single character from stdin without echo."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences (arrow keys, etc.)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                # Handle Shift+Tab (reverse tab)
                if ch3 == "Z":
                    return "shift_tab"
                return f"\x1b[{ch3}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class SettingsUI:
    """Interactive tabbed settings browser using Rich."""

    def __init__(self):
        self.console = Console()
        self.all_settings = get_complete_settings()
        self.categories = [
            cat for cat in CATEGORIES if cat.get_settings(self.all_settings)
        ]
        self.current_tab = 0
        self.selected_index = 0
        self.values: dict[str, str] = {}
        self.original_values: dict[str, str] = {}
        self.env_path: Path = get_default_env_path()
        self.has_unsaved_changes = False

    def run(self, env_path: Path | None = None) -> None:
        """Run the interactive settings browser.

        Args:
            env_path: Optional path to .env file. Uses default if not specified.
        """
        if env_path:
            self.env_path = env_path

        # Load existing values
        self.values = load_env_file(self.env_path)
        self.original_values = self.values.copy()

        # Main loop
        try:
            while True:
                self._render()
                if not self._handle_input():
                    break
        except KeyboardInterrupt:
            self._cleanup()
            self.console.print("\n[yellow]Cancelled[/yellow]")

    def _render(self) -> None:
        """Render the UI."""
        # Clear screen
        self.console.clear()

        # Header with file path
        header = Text()
        header.append("AutoGPT Config", style="bold cyan")
        header.append(f"  ({self.env_path})", style="dim")
        if self.has_unsaved_changes:
            header.append("  *", style="bold yellow")

        self.console.print()
        self.console.print(Panel(header, border_style="cyan", padding=(0, 1)))
        self.console.print()

        # Tab bar
        self._render_tabs()
        self.console.print()

        # Current category settings
        self._render_settings()
        self.console.print()

        # Help text
        self._render_help()

    def _render_tabs(self) -> None:
        """Render the tab bar."""
        tabs = Text()
        tabs.append("  ")

        for i, cat in enumerate(self.categories):
            is_active = i == self.current_tab
            label = f"{i + 1} {cat.name}"

            if is_active:
                tabs.append("[", style="bold cyan")
                tabs.append(label, style="bold cyan")
                tabs.append("]", style="bold cyan")
            else:
                tabs.append(f" {label} ", style="dim")

            tabs.append(" ")

        self.console.print(tabs)

        # Underline for active tab
        underline = Text()
        underline.append("  ")
        for i, cat in enumerate(self.categories):
            label = f"{i + 1} {cat.name}"
            if i == self.current_tab:
                underline.append("═" * (len(label) + 2), style="bold cyan")
            else:
                underline.append(" " * (len(label) + 2), style="dim")
            underline.append(" ")
        self.console.print(underline)

    def _render_settings(self) -> None:
        """Render settings for the current category."""
        if not self.categories:
            self.console.print("  [dim]No settings available[/dim]")
            return

        category = self.categories[self.current_tab]
        settings = category.get_settings(self.all_settings)

        if not settings:
            self.console.print(f"  [dim]No settings in {category.name}[/dim]")
            return

        for i, setting in enumerate(settings):
            is_selected = i == self.selected_index
            value = self.values.get(setting.env_var, "")
            display_value = setting.get_display_value(value or None)

            # Determine if value has changed from original
            changed = value != self.original_values.get(setting.env_var, "")

            line = Text()
            if is_selected:
                line.append("  ❯ ", style="bold green")
                line.append(setting.env_var, style="bold green")
            else:
                line.append("    ", style="dim")
                line.append(setting.env_var, style="dim")

            # Pad to align values
            padding = 30 - len(setting.env_var)
            line.append(" " * max(padding, 1))

            # Value
            if display_value == "[not set]":
                line.append(display_value, style="dim italic")
            elif changed:
                line.append(display_value, style="yellow")
            else:
                line.append(display_value, style="white")

            self.console.print(line)

    def _render_help(self) -> None:
        """Render help text at the bottom."""
        help_text = Text()
        help_text.append("  ")
        help_text.append("←→", style="bold cyan")
        help_text.append("/", style="dim")
        help_text.append("Tab", style="bold cyan")
        help_text.append("/", style="dim")
        help_text.append("1-9", style="bold cyan")
        help_text.append(" category  ", style="dim")
        help_text.append("↑↓", style="bold cyan")
        help_text.append(" navigate  ", style="dim")
        help_text.append("Enter", style="bold cyan")
        help_text.append(" edit  ", style="dim")
        help_text.append("S", style="bold cyan")
        help_text.append(" save  ", style="dim")
        help_text.append("Q", style="bold cyan")
        help_text.append(" quit", style="dim")
        self.console.print(help_text)

    def _handle_input(self) -> bool:
        """Handle keyboard input.

        Returns:
            True to continue, False to exit
        """
        ch = _getch()

        # Tab - next category
        if ch == "\t":
            self.current_tab = (self.current_tab + 1) % len(self.categories)
            self.selected_index = 0
            return True

        # Shift+Tab - previous category
        if ch == "shift_tab":
            self.current_tab = (self.current_tab - 1) % len(self.categories)
            self.selected_index = 0
            return True

        # Number keys 1-9 - jump to category
        if ch in "123456789":
            idx = int(ch) - 1
            if idx < len(self.categories):
                self.current_tab = idx
                self.selected_index = 0
            return True

        # Arrow up
        if ch == "\x1b[A":
            category = self.categories[self.current_tab]
            settings = category.get_settings(self.all_settings)
            if settings:
                self.selected_index = (self.selected_index - 1) % len(settings)
            return True

        # Arrow down
        if ch == "\x1b[B":
            category = self.categories[self.current_tab]
            settings = category.get_settings(self.all_settings)
            if settings:
                self.selected_index = (self.selected_index + 1) % len(settings)
            return True

        # Arrow left - previous category
        if ch == "\x1b[D":
            self.current_tab = (self.current_tab - 1) % len(self.categories)
            self.selected_index = 0
            return True

        # Arrow right - next category
        if ch == "\x1b[C":
            self.current_tab = (self.current_tab + 1) % len(self.categories)
            self.selected_index = 0
            return True

        # Enter - edit selected setting
        if ch in ("\r", "\n"):
            self._edit_current_setting()
            return True

        # S - save
        if ch in ("s", "S"):
            self._save_settings()
            return True

        # Q - quit
        if ch in ("q", "Q"):
            if self.has_unsaved_changes:
                return self._confirm_quit()
            return False

        # Ctrl+C
        if ch == "\x03":
            raise KeyboardInterrupt()

        return True

    def _edit_current_setting(self) -> None:
        """Edit the currently selected setting."""
        if not self.categories:
            return

        category = self.categories[self.current_tab]
        settings = category.get_settings(self.all_settings)

        if not settings or self.selected_index >= len(settings):
            return

        setting = settings[self.selected_index]
        current_value = self.values.get(setting.env_var, "")

        # Clear screen for edit mode
        self.console.clear()
        self.console.print()

        new_value: Any = None

        if setting.field_type == "secret":
            masked = setting.get_display_value(current_value or None)
            new_value = prompt_secret_input(
                self.console,
                label=setting.env_var,
                description=setting.description,
                current_masked=masked if masked != "[not set]" else "",
                env_var=setting.env_var,
            )
            # Keep current value if empty input
            if not new_value and current_value:
                return

        elif setting.field_type == "choice":
            default_idx = 0
            if current_value and current_value in setting.choices:
                default_idx = setting.choices.index(current_value)
            new_value = prompt_selection(
                label=setting.env_var,
                choices=setting.choices,
                description=setting.description,
                default_index=default_idx,
                env_var=setting.env_var,
            )

        elif setting.field_type == "bool":
            current_bool = (
                current_value.lower() in ("true", "1", "yes")
                if current_value
                else False
            )
            result = prompt_boolean(
                self.console,
                label=setting.env_var,
                description=setting.description,
                default=current_bool,
                env_var=setting.env_var,
            )
            new_value = "true" if result else "false"

        elif setting.field_type == "int":
            current_int = (
                int(current_value)
                if current_value and current_value.isdigit()
                else None
            )
            result = prompt_numeric(
                self.console,
                label=setting.env_var,
                description=setting.description,
                default=current_int,
                env_var=setting.env_var,
            )
            new_value = str(result) if result is not None else ""

        elif setting.field_type == "float":
            try:
                current_float = float(current_value) if current_value else None
            except ValueError:
                current_float = None
            result = prompt_float(
                self.console,
                label=setting.env_var,
                description=setting.description,
                default=current_float,
                env_var=setting.env_var,
            )
            new_value = str(result) if result is not None else ""

        else:  # str
            new_value = prompt_text_input(
                self.console,
                label=setting.env_var,
                description=setting.description,
                default=current_value,
                env_var=setting.env_var,
            )

        # Validate the new value
        if new_value:
            is_valid, error = validate_setting(setting.env_var, new_value)
            if not is_valid:
                self.console.print(f"\n[red]Validation error: {error}[/red]")
                self.console.print("[dim]Press any key to continue...[/dim]")
                _getch()
                return
            elif error:  # Warning
                self.console.print(f"\n[yellow]{error}[/yellow]")

        # Update value
        if new_value != current_value:
            self.values[setting.env_var] = new_value
            self.has_unsaved_changes = True

    def _save_settings(self) -> None:
        """Save settings to .env file."""
        try:
            save_env_file(self.env_path, self.values, CATEGORIES)
            self.original_values = self.values.copy()
            self.has_unsaved_changes = False

            self.console.clear()
            self.console.print()
            self.console.print(
                Panel(
                    f"[green]Settings saved to {self.env_path}[/green]",
                    border_style="green",
                )
            )
            self.console.print("\n[dim]Press any key to continue...[/dim]")
            _getch()

        except Exception as e:
            self.console.print(f"\n[red]Error saving settings: {e}[/red]")
            self.console.print("[dim]Press any key to continue...[/dim]")
            _getch()

    def _confirm_quit(self) -> bool:
        """Confirm quitting with unsaved changes.

        Returns:
            True to continue (not quit), False to quit
        """
        self.console.clear()
        self.console.print()
        self.console.print(
            Panel(
                "[yellow]You have unsaved changes![/yellow]\n\n"
                "Press [bold]S[/bold] to save, [bold]Q[/bold] to quit without saving, "
                "or any other key to cancel",
                border_style="yellow",
            )
        )

        ch = _getch()
        if ch in ("s", "S"):
            self._save_settings()
            return False
        elif ch in ("q", "Q"):
            return False
        return True

    def _cleanup(self) -> None:
        """Clean up terminal state."""
        # Terminal should be restored by _getch's finally block
        pass
