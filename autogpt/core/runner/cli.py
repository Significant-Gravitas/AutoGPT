import contextlib
import shlex
import subprocess
import sys
import time

import click
import requests
import uvicorn


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
def client() -> None:
    """Run the Auto-GPT runner client."""
    import autogpt.core.runner.client

    with autogpt_server():
        autogpt.core.runner.client.run()


@autogpt.command()
@click.option("--pdb", is_flag=True, help="Run the agent with pdb.")
def test(pdb):
    from autogpt.core.runner.settings import make_default_settings
    from autogpt.core.runner.utils import handle_exceptions

    main = handle_exceptions(make_default_settings, with_debugger=pdb)
    main()


# @v2.command()
# @click.option("-a", "--is-async", is_flag=True, help="Run the agent asynchronously.")
# def run(is_async: bool):
#     print("Running v2 agent...")
#     print(f"Is async: {is_async}")


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
        f"{sys.executable} autogpt/core/runner/cli.py httpserver --host {host} --port {port}"
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
            time.sleep(0.1)
    yield server_process
    server_process.terminate()


if __name__ == "__main__":
    autogpt()
