"""Services as upstream's entry points build them: `run_processes(X())`, with
X an object that has `start(background=False)`, `cleanup()` and
`service_name`. Each leaves marker files so a test can see what happened."""

import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path

from backend.app import run_processes


def mark(name: str, content: str = "") -> None:
    (Path(os.environ["FAKE_MARKS"]) / name).write_text(content)


def marked(name: str) -> bool:
    return (Path(os.environ["FAKE_MARKS"]) / name).exists()


class Service:
    def __init__(self, name: str) -> None:
        self.name = name
        self.done = threading.Event()

    @property
    def service_name(self) -> str:
        return self.name.capitalize()

    def start(self, background: bool = False) -> int:
        # What AppProcess.execute_run_command does first; off the main thread
        # it raises unless the host has made it thread-aware.
        signal.signal(signal.SIGTERM, lambda signum, frame: None)
        mark(f"{self.name}.environment", " ".join(sorted(os.environ)))
        mark(f"{self.name}.started", threading.current_thread().name)
        mark(f"{self.name}.argv", " ".join(sys.argv))
        try:
            self.run()
        finally:
            mark(f"{self.name}.terminated")
        return 0

    def run(self) -> None:
        self.done.wait()

    def cleanup(self) -> None:
        mark(f"{self.name}.cleaned")
        self.done.set()


class Quitter(Service):
    def run(self) -> None:
        time.sleep(1)


class Stubborn(Service):
    def cleanup(self) -> None:
        mark(f"{self.name}.cleaning")
        time.sleep(60)


class Careless(Service):
    def cleanup(self) -> None:
        raise RuntimeError("the broker is already gone")


class Connector(Service):
    def run(self) -> None:
        mark("database.connected")
        super().run()


class Forever(Service):
    """As the two executors: run() never returns, cleanup or not."""

    def run(self) -> None:
        while True:
            time.sleep(1e5)

    def cleanup(self) -> None:
        mark(f"{self.name}.cleaned")


class Renamed:
    """An upstream that renamed `cleanup`."""

    service_name = "Renamed"

    def __init__(self, name: str) -> None:
        self.service = Service(name)

    def start(self, background: bool = False) -> int:
        return self.service.start(background)

    def shutdown(self) -> None:
        self.service.cleanup()


class Web(Service):
    """As the two API servers: run() is a call to uvicorn.run."""

    def run(self) -> None:
        import uvicorn

        port = int(os.environ[f"FAKE_PORT_{self.name.upper()}"])
        uvicorn.run(web_app(self.name), host="127.0.0.1", port=port, log_config=None)


def web_app(name: str):
    async def app(scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.shutdown":
                    mark(f"{name}.lifespan-shutdown")
                await send({"type": message["type"] + ".complete"})
                if message["type"] == "lifespan.shutdown":
                    return
        body = str(id(asyncio.get_running_loop())).encode()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body})

    return app


def one() -> None:
    run_processes(Service("one"))


def two() -> None:
    run_processes(Service("two"))


def quitter() -> None:
    run_processes(Quitter("quitter"))


def stubborn() -> None:
    run_processes(Stubborn("stubborn"))


def careless() -> None:
    run_processes(Careless("careless"))


def connector() -> None:
    run_processes(Connector("connector"))


def forever() -> None:
    run_processes(Forever("forever"))


def renamed() -> None:
    run_processes(Renamed("renamed"))


def web_a() -> None:
    run_processes(Web("a"))


def web_b() -> None:
    run_processes(Web("b"))


def hands_nothing_over() -> None:
    """An upstream whose entry point no longer goes through run_processes."""
    mark("nothing.called")


def serves_itself() -> None:
    """An upstream whose entry point runs its service in the foreground."""
    mark("itself.started")
    time.sleep(1e5)
