import contextlib
import pathlib
import shlex
import subprocess
import sys
import time

import click
import requests
import uvicorn
import yaml

from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
    status,
)
from autogpt.core.runner.client_lib.utils import coroutine


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


autogpt.add_command(make_settings)
autogpt.add_command(status)


@autogpt.command()
@click.option(
    "host",
    "--host",
    default="localhost",
    help="The host for the webserver.",
    type=click.STRING,
)
@click.option(
    "port",
    "--port",
    default=8080,
    help="The port of the webserver.",
    type=click.INT,
)
def server(host: str, port: int) -> None:
    """Run the Auto-GPT runner httpserver."""
    click.echo("Running Auto-GPT runner httpserver...")
    uvicorn.run(
        "autogpt.core.runner.cli_web_app.server.api:app",
        workers=1,
        host=host,
        port=port,
        reload=True,
    )


@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@coroutine
async def client(settings_file) -> None:
    """Run the Auto-GPT runner client."""
    settings_file = pathlib.Path(settings_file)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())

    from autogpt.core.runner.cli_web_app.client.client import run

    with autogpt_server():
        run()


@contextlib.contextmanager
def autogpt_server():
    host = "localhost"
    port = 8080
    cmd = shlex.split(
        f"{sys.executable} autogpt/core/runner/cli_web_app/cli.py server --host {host} --port {port}"
    )
    server_process = subprocess.Popen(
        args=cmd,
    )
    started = False

    while not started:
        try:
            requests.get(f"http://{host}:{port}")
            started = True
        except requests.exceptions.ConnectionError:
            time.sleep(0.2)
    yield server_process
    server_process.terminate()


if __name__ == "__main__":
    autogpt()
