import pathlib

import click
import yaml
from agent_protocol import Agent as AgentProtocol

from autogpt.core.runner.cli_web_app.server.api import task_handler
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.client_lib.utils import coroutine


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


autogpt.add_command(make_settings)


@autogpt.command()
@click.option(
    "port",
    "--port",
    default=8080,
    help="The port of the webserver.",
    type=click.INT,
)
def server(port: int) -> None:
    """Run the AutoGPT runner httpserver."""
    click.echo("Running AutoGPT runner httpserver...")
    AgentProtocol.handle_task(task_handler).start(port)


@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@coroutine
async def client(settings_file) -> None:
    """Run the AutoGPT runner client."""
    settings_file = pathlib.Path(settings_file)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())

    settings
    # TODO: Call the API server with the settings and task,
    #   using the Python API client for agent protocol.


if __name__ == "__main__":
    autogpt()
