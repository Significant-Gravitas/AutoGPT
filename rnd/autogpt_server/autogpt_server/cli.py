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
        app.main(silent=True) # type: ignore


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
    pid = MainApp().start(background=True, silent=True) # type: ignore
    print(f"Server running in process: {pid}")

    write_pid(pid)
    print("done")
    os._exit(status=0) # type: ignore


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


@test.command()
@click.argument("server_address")
def websocket(server_address: str):
    """
    Tests the websocket connection.
    """
    import asyncio
    import websockets
    from autogpt_server.server.ws_api import Methods, WsMessage, ExecutionSubscription

    async def send_message(server_address: str):
        uri = f"ws://{server_address}"
        async with websockets.connect(uri) as websocket:
            try:
                await websocket.send(
                    WsMessage(
                        method=Methods.SUBSCRIBE,
                        data=ExecutionSubscription(
                            channel="test", graph_id="asdasd", run_id="asdsa"
                        ).model_dump(),
                    ).model_dump_json()
                )
                while True:
                    response = await websocket.recv()
                    print(f"Response from server: {response}")
                    await websocket.close()
            except InterruptedError:
                exit(0)

    asyncio.run(send_message(server_address))
    print("Testing WS")


main.add_command(test)

if __name__ == "__main__":
    main()
