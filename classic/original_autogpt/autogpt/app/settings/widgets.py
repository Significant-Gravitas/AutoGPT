"""Reusable Rich input widgets for settings UI."""

from __future__ import annotations

from autogpt.app.ui.rich_select import RichSelect
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text


def prompt_text_input(
    console: Console,
    label: str,
    description: str = "",
    default: str = "",
    env_var: str = "",
) -> str:
    """Prompt for text input with Rich styling.

    Args:
        console: Rich Console instance
        label: Setting name/label
        description: Help text for the setting
        default: Default value
        env_var: Environment variable name

    Returns:
        User input string
    """
    header = Text()
    header.append(f"{label}\n", style="bold cyan")
    if description:
        header.append(f"{description}\n", style="dim")
    if env_var:
        header.append(f"ENV: {env_var}", style="dim italic")

    console.print()
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))

    prompt_text = "Value"
    if default:
        prompt_text += f" [{default}]"

    result = Prompt.ask(f"  {prompt_text}", default=default, console=console)
    return result


def prompt_secret_input(
    console: Console,
    label: str,
    description: str = "",
    current_masked: str = "",
    env_var: str = "",
) -> str:
    """Prompt for secret/password input with Rich styling.

    Args:
        console: Rich Console instance
        label: Setting name/label
        description: Help text for the setting
        current_masked: Current masked value for display
        env_var: Environment variable name

    Returns:
        User input string (unmasked)
    """
    header = Text()
    header.append(f"{label}\n", style="bold cyan")
    if description:
        header.append(f"{description}\n", style="dim")
    if env_var:
        header.append(f"ENV: {env_var}\n", style="dim italic")
    if current_masked:
        header.append(f"Current: {current_masked}", style="yellow")

    console.print()
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))
    console.print("  [dim](Leave empty to keep current value)[/dim]")

    result = Prompt.ask("  Value", password=True, console=console)
    return result


def prompt_selection(
    label: str,
    choices: list[str],
    description: str = "",
    default_index: int = 0,
    env_var: str = "",
) -> str:
    """Prompt for selection from choices using RichSelect.

    Args:
        label: Setting name/label
        choices: List of choices
        description: Help text for the setting
        default_index: Index of default choice
        env_var: Environment variable name

    Returns:
        Selected choice string
    """
    subtitle = description
    if env_var:
        subtitle = (
            f"{description}\nENV: {env_var}" if description else f"ENV: {env_var}"
        )

    selector = RichSelect(
        choices=choices,
        title=label,
        subtitle=subtitle,
        default_index=default_index,
        show_feedback_option=False,
    )
    result = selector.run()
    return result.choice


def prompt_boolean(
    console: Console,
    label: str,
    description: str = "",
    default: bool = False,
    env_var: str = "",
) -> bool:
    """Prompt for boolean input with Rich styling.

    Args:
        console: Rich Console instance
        label: Setting name/label
        description: Help text for the setting
        default: Default value
        env_var: Environment variable name

    Returns:
        Boolean value
    """
    header = Text()
    header.append(f"{label}\n", style="bold cyan")
    if description:
        header.append(f"{description}\n", style="dim")
    if env_var:
        header.append(f"ENV: {env_var}", style="dim italic")

    console.print()
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))

    return Confirm.ask("  Enable", default=default, console=console)


def prompt_numeric(
    console: Console,
    label: str,
    description: str = "",
    default: int | None = None,
    env_var: str = "",
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Prompt for numeric input with Rich styling.

    Args:
        console: Rich Console instance
        label: Setting name/label
        description: Help text for the setting
        default: Default value
        env_var: Environment variable name
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Integer value or None if empty
    """
    header = Text()
    header.append(f"{label}\n", style="bold cyan")
    if description:
        header.append(f"{description}\n", style="dim")
    if env_var:
        header.append(f"ENV: {env_var}\n", style="dim italic")

    constraints = []
    if min_value is not None:
        constraints.append(f"min: {min_value}")
    if max_value is not None:
        constraints.append(f"max: {max_value}")
    if constraints:
        header.append(f"({', '.join(constraints)})", style="dim")

    console.print()
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))

    prompt_text = "Value"
    if default is not None:
        prompt_text += f" [{default}]"
    else:
        prompt_text += " [empty to skip]"

    while True:
        result = Prompt.ask(f"  {prompt_text}", console=console)

        # Handle empty input
        if not result.strip():
            return default

        # Try to parse as integer
        try:
            value = int(result)

            # Validate range
            if min_value is not None and value < min_value:
                console.print(f"  [red]Value must be at least {min_value}[/red]")
                continue
            if max_value is not None and value > max_value:
                console.print(f"  [red]Value must be at most {max_value}[/red]")
                continue

            return value
        except ValueError:
            console.print("  [red]Please enter a valid number[/red]")


def prompt_float(
    console: Console,
    label: str,
    description: str = "",
    default: float | None = None,
    env_var: str = "",
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Prompt for float input with Rich styling.

    Args:
        console: Rich Console instance
        label: Setting name/label
        description: Help text for the setting
        default: Default value
        env_var: Environment variable name
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Float value or None if empty
    """
    header = Text()
    header.append(f"{label}\n", style="bold cyan")
    if description:
        header.append(f"{description}\n", style="dim")
    if env_var:
        header.append(f"ENV: {env_var}\n", style="dim italic")

    constraints = []
    if min_value is not None:
        constraints.append(f"min: {min_value}")
    if max_value is not None:
        constraints.append(f"max: {max_value}")
    if constraints:
        header.append(f"({', '.join(constraints)})", style="dim")

    console.print()
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))

    prompt_text = "Value"
    if default is not None:
        prompt_text += f" [{default}]"
    else:
        prompt_text += " [empty to skip]"

    while True:
        result = Prompt.ask(f"  {prompt_text}", console=console)

        # Handle empty input
        if not result.strip():
            return default

        # Try to parse as float
        try:
            value = float(result)

            # Validate range
            if min_value is not None and value < min_value:
                console.print(f"  [red]Value must be at least {min_value}[/red]")
                continue
            if max_value is not None and value > max_value:
                console.print(f"  [red]Value must be at most {max_value}[/red]")
                continue

            return value
        except ValueError:
            console.print("  [red]Please enter a valid number[/red]")
