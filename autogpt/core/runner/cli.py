import asyncio
import contextlib
import functools
import pathlib
import shlex
import subprocess
import sys
import time

import click
import requests
import uvicorn
import yaml

DEFAULT_SETTINGS_FILE = str(pathlib.Path("~/auto-gpt/settings.yaml").expanduser())


def coroutine(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


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
        "autogpt.core.runner.httpserver:app",
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

    from autogpt.core.runner.client import run_auto_gpt

    with autogpt_server():
        await run_auto_gpt(settings)


@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
def config(settings_file: str) -> None:
    from autogpt.core.runner.settings import make_default_settings

    make_default_settings(pathlib.Path(settings_file))


@autogpt.command()
@click.option("-d", "--detailed", is_flag=True, help="Show detailed status.")
def status(detailed: bool):
    import importlib
    import pkgutil

    import autogpt.core
    from autogpt.core.status import print_status

    status_list = []
    for loader, package_name, is_pkg in pkgutil.iter_modules(autogpt.core.__path__):
        if is_pkg:
            subpackage = importlib.import_module(
                f"{autogpt.core.__name__}.{package_name}"
            )
            if hasattr(subpackage, "status"):
                status_list.append(subpackage.status)

    print_status(status_list, detailed)


@contextlib.contextmanager
def autogpt_server():
    host = "localhost"
    port = 8080
    cmd = shlex.split(
        f"{sys.executable} autogpt/core/runner/cli.py server --host {host} --port {port}"
    )
    server_process = subprocess.Popen(
        args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
