from pathlib import Path

import click
import yaml

from autogpt.core.runner.cli_app.main import run_auto_gpt
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.client_lib.utils import coroutine, handle_exceptions


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
@click.option(
    "--pdb",
    is_flag=True,
    help="Drop into a debugger if an error is raised.",
)
@coroutine
async def run(settings_file: str, pdb: bool) -> None:
    """Run the AutoGPT agent."""
    click.echo("Running AutoGPT agent...")
    settings_file: Path = Path(settings_file)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())
    main = handle_exceptions(run_auto_gpt, with_debugger=pdb)
    await main(settings)


if __name__ == "__main__":
    autogpt()
