"""Third-party runtimes the desktop bundle is assembled from.

Every entry is an upstream release pinned by digest, taken from the
publisher's own checksum list where there is one (python-build-standalone,
nodejs.org, GitHub release assets, hex.pm, valkey.io, postgresql.org) and
from the download itself for the two GitHub source archives. A version bump
changes the URL and the digest together. Nothing here is fetched on a
user's machine: the build downloads it once and the installer carries it.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

PYTHON_VERSION = "3.13.16"
PYTHON_BUILD = "20261001"
NODE_VERSION = "24.21.0"
POSTGRES_BUNDLE = "v0.3.1"  # PostgreSQL 18.6 + pgvector 0.8.6 + pg_trgm
# Linux builds PostgreSQL from source (see build_runtime.step_postgres). 16 is
# the newest major whose tarball still ships the generated parser, so the
# build needs only gcc and make.
POSTGRES_SOURCE_VERSION = "16.12"
PGVECTOR_VERSION = "0.8.7"
ERLANG_VERSION = "27.3.4.18"  # RabbitMQ 4.3 supports Erlang 27.x-28.x
RABBITMQ_VERSION = "4.3.6"  # matches single-container/Dockerfile
VALKEY_VERSION = "8.1.10"
# The Claude Code CLI that the locked claude-agent-sdk is built with, and
# carries in its macOS and Linux wheels. That SDK version has no Windows
# wheel: the Windows bundle installs the same SDK from source and gets this,
# the same CLI, from Anthropic's release bucket (claude_cli.py). The build
# stops, on every platform, when the lock moves to an SDK built with another
# CLI; bump it then.
CLAUDE_CLI_VERSION = "2.1.284"
REDIS_WINDOWS_VERSION = "8.10.2"


@dataclass(frozen=True)
class Artifact:
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]


def platform_key() -> str:
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[
        machine
    ]
    return f"{sys.platform}-{arch}"


_PBS = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PYTHON_BUILD}/cpython-{PYTHON_VERSION}+{PYTHON_BUILD}-{{triple}}-install_only_stripped.tar.gz"
)
_NODE = f"https://nodejs.org/dist/v{NODE_VERSION}"
_POSTGRES = (
    "https://github.com/boomship/postgres-vector-embedded/releases/download/"
    f"{POSTGRES_BUNDLE}/postgres-lite-{{target}}.tar.gz"
)
_RABBITMQ = (
    "https://github.com/rabbitmq/rabbitmq-server/releases/download/"
    f"v{RABBITMQ_VERSION}"
)

ARTIFACTS: dict[str, dict[str, Artifact]] = {
    "win32-x64": {
        "python": Artifact(
            _PBS.format(triple="x86_64-pc-windows-msvc"),
            "c402eb9a35aba90de319bbc87d23e6d887eca1551a05cfa80775f92dcf3ebc08",
        ),
        "node": Artifact(
            f"{_NODE}/win-x64/node.exe",
            "ba4e6d110e8c1592a1ecd390f6b05f3da124b13871a5be62b341a07a853c6c32",
        ),
        "postgres": Artifact(
            _POSTGRES.format(target="win32-x64"),
            "920bebd11a2cc336096b3df382ed02c21df050d18a25420bf9049e50dd5bb38e",
        ),
        "erlang": Artifact(
            "https://github.com/erlang/otp/releases/download/"
            f"OTP-{ERLANG_VERSION}/otp_win64_{ERLANG_VERSION}.zip",
            "2af5d35d521211c16e9fa31b23f76e44b7b3284a27c6c1295fbeb382d2ff211f",
        ),
        "rabbitmq": Artifact(
            f"{_RABBITMQ}/rabbitmq-server-windows-{RABBITMQ_VERSION}.zip",
            "48161b084f1257edc112b24fccda378cc3e83c8e4b94984801cd6651817981f1",
        ),
        # Checksum from the release's own manifest.json.
        "claude-cli": Artifact(
            "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/"
            f"claude-code-releases/{CLAUDE_CLI_VERSION}/win32-x64/claude.exe",
            "0416631e846f743110da5282409776fa1313e65f33a588aae066eaf8db0fda7d",
        ),
        # Valkey publishes no Windows build; build/valkey-windows.sh makes
        # one under MSYS2. This is the stand-in for a developer machine
        # without MSYS2: the redis-windows project's build of Redis. It is
        # not for release (not BSD, and Valkey cannot read the files it
        # writes), so build_runtime.py takes it only when asked to.
        "valkey": Artifact(
            "https://github.com/redis-windows/redis-windows/releases/download/"
            f"{REDIS_WINDOWS_VERSION}/Redis-{REDIS_WINDOWS_VERSION}-Windows-x64-msys2.zip",
            "7c8cebd50347eaa1d9e784da842ed47a4f33394637835531d6614777b950ee85",
        ),
    },
    "darwin-arm64": {
        "python": Artifact(
            _PBS.format(triple="aarch64-apple-darwin"),
            "d00669acb53c1b014f1fcf5eaea740d8e45c3aff4b1e0244dc0f8697fb211a82",
        ),
        "node": Artifact(
            f"{_NODE}/node-v{NODE_VERSION}-darwin-arm64.tar.gz",
            "bed7eea5325e1108f32ce5228ddd6a5f0f08a499ee42aa7442aea583702f6057",
        ),
        "postgres": Artifact(
            _POSTGRES.format(target="darwin-arm64"),
            "5ced879c2707bc69149bb1f7bf158b2ddd9804e140ae0f408b9fbcb1c81adf73",
        ),
        "erlang": Artifact(
            "https://github.com/erlef/otp_builds/releases/download/"
            f"OTP-{ERLANG_VERSION}/OTP-{ERLANG_VERSION}-macos-arm64.tar.gz",
            "b5940c8d6c96a9eb74c976644bb4ec97190e29784f3ae10225561081699e83e0",
        ),
        "rabbitmq": Artifact(
            f"{_RABBITMQ}/rabbitmq-server-generic-unix-{RABBITMQ_VERSION}.tar.xz",
            "aa0ff5de9ff0136dc42eba21dc3a973cc3b286437cdba0f45556e890509c1614",
        ),
        # No macOS binaries upstream; compiled from this source tarball.
        "valkey": Artifact(
            f"https://github.com/valkey-io/valkey/archive/refs/tags/{VALKEY_VERSION}.tar.gz",
            "c74e50cd83f6d398a3dc570e04ac2fe538249585d021f76ee2449bbf9ebd04ed",
        ),
    },
    "linux-x64": {
        "python": Artifact(
            _PBS.format(triple="x86_64-unknown-linux-gnu"),
            "ffcb50e716789d1a6e1db5e745d4d194ac8ed9b015ccf5eabcaccb179a25e4a8",
        ),
        "node": Artifact(
            f"{_NODE}/node-v{NODE_VERSION}-linux-x64.tar.gz",
            "6e1db87ef58b8819e5d5402eff1536491b18edd8eb7bee5ef7897876e88dc5ff",
        ),
        # The prebuilt Linux bundle needs glibc 2.38 (Ubuntu 24.04); built
        # from source here it runs wherever the build machine's glibc does.
        "postgres": Artifact(
            "https://ftp.postgresql.org/pub/source/"
            f"v{POSTGRES_SOURCE_VERSION}/postgresql-{POSTGRES_SOURCE_VERSION}.tar.bz2",
            "b253ee949303ef5df00e24002600da4fb37e5ccfafa78718c6ea6a936b4d97f1",
        ),
        "pgvector": Artifact(
            f"https://github.com/pgvector/pgvector/archive/refs/tags/v{PGVECTOR_VERSION}.tar.gz",
            "cac0b10c360f05b2d521200105ba3697e773d4cd3731f5a915a7e37ebe0bea85",
        ),
        "erlang": Artifact(
            f"https://builds.hex.pm/builds/otp/amd64/ubuntu-22.04/OTP-{ERLANG_VERSION}.tar.gz",
            "72ceab002e9943b1d8f6bf4e50478ed4d16e5542e1bf8daa2382b8dbdb234966",
        ),
        "rabbitmq": Artifact(
            f"{_RABBITMQ}/rabbitmq-server-generic-unix-{RABBITMQ_VERSION}.tar.xz",
            "aa0ff5de9ff0136dc42eba21dc3a973cc3b286437cdba0f45556e890509c1614",
        ),
        "valkey": Artifact(
            f"https://download.valkey.io/releases/valkey-{VALKEY_VERSION}-jammy-x86_64.tar.gz",
            "23400087a684b6e87f66332e0a836a2674c24c7c482361766777149499c08e50",
        ),
    },
}
