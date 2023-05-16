from pathlib import Path

import click
import yaml

from autogpt.core.runner.app_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.app_lib.utils import coroutine
from autogpt.core.runner.cli_app.main import run_auto_gpt


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


autogpt.add_command(make_settings)


@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@coroutine
async def run(settings_file: str) -> None:
    """Run the Auto-GPT agent."""
    click.echo("Running Auto-GPT agent...")
    settings_file = Path(settings_file)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())
    await run_auto_gpt(settings)


if __name__ == "__main__":
    autogpt()
