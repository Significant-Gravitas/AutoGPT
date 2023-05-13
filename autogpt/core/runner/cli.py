import shlex
import subprocess
import sys
import time

import click


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


@autogpt.command()
def httpserver() -> None:
    """Run the Auto-GPT runner httpserver."""
    import uvicorn

    print("Running Auto-GPT runner httpserver...")
    uvicorn.run(
        "autogpt.core.runner.server:app",
        workers=1,
        host="localhost",
        port=8080,
        reload=True,
    )


@autogpt.command()
def client() -> None:
    """Run the Auto-GPT runner client."""
    import autogpt.core.runner.client

    # print("Running Auto-GPT runner client...")
    # autogpt.core.runner.client.run()

    cmd = f"{sys.executable} autogpt/core/runner/cli.py httpserver"
    cmds = shlex.split(cmd)
    print("cmds:", cmds)

    process = subprocess.Popen(
        args=cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)  # wait for server to start

    autogpt.core.runner.client.run()


# @v2.command()
# @click.option("-a", "--is-async", is_flag=True, help="Run the agent asynchronously.")
# def run(is_async: bool):
#     print("Running v2 agent...")
#     print(f"Is async: {is_async}")


# @v2.command()
# @click.option("-d", "--detailed", is_flag=True, help="Show detailed status.")
# def status(detailed: bool):
#     import importlib
#     import pkgutil

#     import autogpt.core
#     from autogpt.core.status import print_status

#     status_list = []
#     for loader, package_name, is_pkg in pkgutil.iter_modules(autogpt.core.__path__):
#         if is_pkg:
#             subpackage = importlib.import_module(
#                 f"{autogpt.core.__name__}.{package_name}"
#             )
#             if hasattr(subpackage, "status"):
#                 status_list.append(subpackage.status)

#     print_status(status_list, detailed)


# @click.group()
# def runner() -> None:
#     """Auto-GPT Runner commands"""
#     pass


# @runner.command()
# def server() -> None:
#     """Run the Auto-GPT runner server."""
#     import autogpt.core.messaging.base
#     import autogpt.core.runner.server

#     print("Running Auto-GPT runner server...")
#     msg = autogpt.core.messaging.base.Message(
#         {
#             "message": "Translated user input into objective prompt.",
#             "objective_prompt": "test auto-gpt",
#         },
#         None,
#     )
#     autogpt.core.runner.server.launch_agent("msg")


if __name__ == "__main__":
    autogpt()
