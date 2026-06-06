"""CLI utilities for backend development & administration"""

import os
import pathlib

import click
import psutil

from backend.util.process import AppProcess

from .chat import chat
from .test import test


@click.group()
def main():
    """AutoGPT Server CLI Tool"""
    pass


main.add_command(chat)
main.add_command(test)


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


class MainApp(AppProcess):
    def run(self):
        from backend import app

        app.main(silent=True)


def get_pid() -> int | None:
    file_path = get_pid_path()
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        pid = file.read()
    try:
        return int(pid)
    except ValueError:
        return None


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
    try:
        process = psutil.Process(int(pid))
    except psutil.NoSuchProcess:
        print("Server Stopped")
        return
    for child in process.children(recursive=True):
        child.terminate()
    process.terminate()

    print("Server Stopped")


def write_pid(pid: int):
    file_path = get_pid_path()
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(pid))


def get_pid_path() -> pathlib.Path:
    home_dir = pathlib.Path.home()
    new_dir = home_dir / ".config" / "agpt"
    file_path = new_dir / "running.tmp"
    return file_path


@main.command()
def gen_encrypt_key():
    """
    Generate a new encryption key
    """
    from cryptography.fernet import Fernet

    print(Fernet.generate_key().decode())


if __name__ == "__main__":
    main()
