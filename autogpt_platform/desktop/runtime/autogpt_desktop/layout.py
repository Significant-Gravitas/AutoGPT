"""Where things are: the read-only bundle shipped with the app, and the
per-user data directory that survives upgrades.

The bundle mirrors the appliance image (single-container/Dockerfile) with
each Linux path replaced by a directory under the runtime root:

    runtime/
      manifest.json         how the shell starts this package
      python/               relocatable CPython with the backend installed
      backend/              backend source tree (cwd for every backend service)
      frontend/             Next standalone server
      postgres/             PostgreSQL + pgvector
      valkey/               Valkey server
      erlang/, rabbitmq/    RabbitMQ and the Erlang runtime it needs
      erlang-patches/       inet_udp built to bind loopback (build/erlang_patches.py)
      prisma/               Prisma CLI and engines
      tools/bin/            programs the backend runs by name (ffmpeg)
      assets/               00-init.sql
"""

from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

WINDOWS = sys.platform == "win32"
EXE = ".exe" if WINDOWS else ""
SCRIPT = ".bat" if WINDOWS else ""


@dataclass(frozen=True)
class Bundle:
    root: Path

    @classmethod
    def locate(cls) -> Bundle:
        configured = os.environ.get("AUTOGPT_DESKTOP_RUNTIME")
        root = Path(configured) if configured else Path(__file__).resolve().parents[1]
        return cls(root)

    @property
    def python(self) -> Path:
        return Path(sys.executable)

    @property
    def backend_dir(self) -> Path:
        return self.root / "backend"

    @property
    def frontend_server(self) -> Path:
        return self.root / "frontend" / "server.js"

    def postgres_bin(self, name: str) -> Path:
        return self.root / "postgres" / "bin" / f"{name}{EXE}"

    @property
    def valkey_server(self) -> Path:
        return self.root / "valkey" / f"valkey-server{EXE}"

    @property
    def erlang_home(self) -> Path:
        return self.root / "erlang"

    @property
    def rabbitmq_home(self) -> Path:
        return self.root / "rabbitmq"

    @property
    def erlang_patches(self) -> Path:
        return self.root / "erlang-patches"

    def rabbitmq_script(self, name: str) -> Path:
        return self.rabbitmq_home / "sbin" / f"{name}{SCRIPT}"

    @property
    def prisma_cli(self) -> Path:
        return self.root / "prisma" / "node_modules" / "prisma" / "build" / "index.js"

    def prisma_engine(self, name: str) -> Path:
        return self.root / "prisma" / f"{name}{EXE}"

    @property
    def tools_bin(self) -> Path:
        return self.root / "tools" / "bin"

    @property
    def ffmpeg(self) -> Path:
        return self.tools_bin / f"ffmpeg{EXE}"

    @property
    def init_sql(self) -> Path:
        return self.root / "assets" / "00-init.sql"

    def node_command(self) -> list[str]:
        """Node runs the frontend and the Prisma CLI. Inside the desktop app
        that is Electron's own Node (the shell exports its executable with
        ELECTRON_RUN_AS_NODE); headless runs fall back to a bundled node."""
        configured = os.environ.get("AUTOGPT_DESKTOP_NODE")
        if configured:
            return [configured]
        return [str(self.root / "node" / f"node{EXE}")]


@dataclass(frozen=True)
class DataDir:
    root: Path

    @classmethod
    def locate(cls) -> DataDir:
        configured = os.environ.get("AUTOGPT_DESKTOP_DATA_DIR")
        if not configured:
            raise RuntimeError("AUTOGPT_DESKTOP_DATA_DIR is not set")
        return cls(Path(configured))

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def runtime_env(self) -> Path:
        return self.config / "runtime.env"

    @property
    def ports_file(self) -> Path:
        return self.config / "ports.json"

    @property
    def postgres(self) -> Path:
        return self.root / "postgres"

    @property
    def valkey(self) -> Path:
        return self.root / "valkey"

    @property
    def rabbitmq(self) -> Path:
        return self.root / "rabbitmq"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def run(self) -> Path:
        return self.root / "run"

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def workspaces(self) -> Path:
        return self.root / "workspaces"

    @property
    def store_media(self) -> Path:
        return self.root / "store-media"

    @property
    def backend_cache(self) -> Path:
        return self.root / "cache" / "backend"

    @property
    def next_cache(self) -> Path:
        return self.root / "cache" / "next"

    @property
    def frontend_home(self) -> Path:
        return self.root / "frontend-home"

    def prepare(self) -> None:
        # Private to the user: it holds the database, API keys and logs, and
        # on Linux ~/.local/share is usually readable by everyone.
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not WINDOWS:
            self.root.chmod(0o700)
        for path in (
            self.config,
            self.postgres.parent,
            self.valkey,
            self.rabbitmq,
            self.logs,
            self.run,
            self.home,
            self.workspaces,
            self.store_media,
            self.backend_cache,
            self.next_cache,
            self.frontend_home,
        ):
            path.mkdir(parents=True, exist_ok=True)


def write_private(path: Path, content: str) -> None:
    """Write a file only its owner can read, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    os.replace(temporary, path)
