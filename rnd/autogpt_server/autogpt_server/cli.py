"""
The command line interface for the agent server
"""

import os
import pathlib

import click
import psutil

from autogpt_server import app
from autogpt_server.util.process import AppProcess


def get_pid_path() -> pathlib.Path:
    home_dir = pathlib.Path.home()
    new_dir = home_dir / ".config" / "agpt"
    file_path = new_dir / "running.tmp"
    return file_path


def get_pid() -> int | None:
    file_path = get_pid_path()
    if not file_path.exists():
        return None

    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "r", encoding="utf-8") as file:
        pid = file.read()
    try:
        return int(pid)
    except ValueError:
        return None


def write_pid(pid: int):
    file_path = get_pid_path()
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(pid))


class MainApp(AppProcess):
    def run(self):
        app.main(silent=True)


@click.group()
def main():
    """AutoGPT Server CLI Tool"""
    pass


@main.command()
def start():
    """
    Starts the server in the background and saves the PID
    """
    # Define the path for the new directory and file
    pid = get_pid()
    if pid and psutil.pid_exists(pid):
        print("Server is already running")
        exit(1)
    elif pid:
        print("PID does not exist deleting file")
        os.remove(get_pid_path())

    print("Starting server")
    pid = MainApp().start(background=True, silent=True)
    print(f"Server running in process: {pid}")

    write_pid(pid)
    print("done")
    os._exit(status=0)


@main.command()
def stop():
    """
    Stops the server
    """
    pid = get_pid()
    if not pid:
        print("Server is not running")
        return

    os.remove(get_pid_path())
    process = psutil.Process(int(pid))
    for child in process.children(recursive=True):
        child.terminate()
    process.terminate()

    print("Server Stopped")


@click.group()
def test():
    """
    Group for test commands
    """
    pass


@test.command()
def event():
    """
    Send an event to the running server
    """
    print("Event sent")


if __name__ == "__main__":
    main()
