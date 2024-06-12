"""
The command line interface for the agent server
"""

from multiprocessing import freeze_support
from multiprocessing.spawn import freeze_support as freeze_support_spawn

import click


@click.group()
def main():
    """AutoGPT Server CLI Tool"""


@main.command()
def background() -> None:
    """
    Command to run the server in the background. Used by the run command
    """
    from autogpt_server.app import background_process

    background_process()


@main.command()
def start():
    """
    Starts the server in the background and saves the PID
    """
    import os
    import pathlib
    import subprocess
    import psutil

    # Define the path for the new directory and file
    home_dir = pathlib.Path.home()
    new_dir = home_dir / ".config" / "agpt"
    file_path = new_dir / "running.tmp"

    # Create the directory if it does not exist
    os.makedirs(new_dir, exist_ok=True)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            pid = int(file.read())
            if psutil.pid_exists(pid):
                print("Server is already running")
                exit(1)
            else:
                print("PID does not exist deleting file")
                os.remove(file_path)

    sp = subprocess.Popen(
        ["poetry", "run", "python", "autogpt_server/cli.py", "background"],
        stdout=subprocess.DEVNULL,  # Redirect standard output to devnull
        stderr=subprocess.DEVNULL,  # Redirect standard error to devnull
    )
    print(f"Server running in process: {sp.pid}")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(sp.pid))


@main.command()
def stop():
    """
    Stops the server
    """
    import os
    import pathlib
    import subprocess

    home_dir = pathlib.Path.home()
    new_dir = home_dir / ".config" / "agpt"
    file_path = new_dir / "running.tmp"
    if not file_path.exists():
        print("Server is not running")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        pid = file.read()
    os.remove(file_path)

    subprocess.Popen(["kill", pid])
    print("Server Stopped")


@click.group()
def test():
    """
    Group for test commands
    """


@test.command()
def event():
    """
    Send an event to the running server
    """
    print("Event sent")


main.add_command(test)


def start_cli() -> None:
    """
    Entry point into the cli
    """
    freeze_support()
    freeze_support_spawn()
    main()


if __name__ == "__main__":
    start_cli()
