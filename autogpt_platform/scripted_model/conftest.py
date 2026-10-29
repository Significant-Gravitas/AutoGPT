import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripted_model.server import ScriptedModelServer


class RunningServer:
    def __init__(self, server: ScriptedModelServer, scripts_dir: Path) -> None:
        self.server = server
        self.scripts_dir = scripts_dir
        host, port = server.server_address[:2]
        self.url = f"http://{host}:{port}"

    def add_script(self, name: str, script: dict) -> None:
        (self.scripts_dir / f"{name}.json").write_text(json.dumps(script))

    def journal(self, **query: str) -> list[dict]:
        return self.server.journal.find(**query)


@pytest.fixture
def scripted_model(tmp_path: Path) -> Iterator[RunningServer]:
    server = ScriptedModelServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RunningServer(server, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
