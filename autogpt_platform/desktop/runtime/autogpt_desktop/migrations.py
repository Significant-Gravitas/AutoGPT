"""The Prisma CLI, which applies the database migrations.

Before any command the CLI makes sure it has a query engine and a schema
engine, and fetches what it does not find from binaries.prisma.sh into its
own package directory: a download nobody pinned, written into the bundle, on
the user's machine. By default the query engine it wants is the Node-API
library for whatever platform it detects (on a Linux with OpenSSL 1.1 and 3
both installed it picks 1.1), which the bundle does not carry.

An engine named in the environment is taken as it is, with no look at the
package directory, the download cache or the network. The bundle has the two
engines as executables (build_runtime.py, step_prisma), so the CLI is told to
use the executable query engine and given both paths. `migrate deploy` only
ever runs the schema engine.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from contextlib import closing

from autogpt_desktop.layout import Bundle
from autogpt_desktop.process import run_tool


class IncompatibleSchemaError(RuntimeError):
    pass


def bundled_history(bundle: Bundle) -> dict[str, set[str]]:
    directory = bundle.backend_dir / "migrations"
    if not directory.is_dir():
        raise IncompatibleSchemaError(
            "This AutoGPT bundle has no database migration history."
        )
    history = {}
    for path in sorted(directory.glob("*/migration.sql")):
        content = path.read_bytes()
        unix = content.replace(b"\r\n", b"\n")
        history[path.parent.name] = {
            hashlib.sha256(value).hexdigest()
            for value in (content, unix, unix.replace(b"\n", b"\r\n"))
        }
    return history


def require_compatible_history(bundle: Bundle, connect) -> None:
    with closing(connect()) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT to_regclass('platform._prisma_migrations') IS NOT NULL")
        if not cursor.fetchone()[0]:
            return
        cursor.execute(
            "SELECT migration_name, checksum FROM platform._prisma_migrations "
            "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL"
        )
        applied = cursor.fetchall()
    known = bundled_history(bundle)
    incompatible = [
        name for name, checksum in applied if checksum not in known.get(name, set())
    ]
    if incompatible:
        raise IncompatibleSchemaError(
            "This AutoGPT version cannot open this database: it lacks or changes "
            f"applied migrations ({', '.join(sorted(set(incompatible)))}). "
            "Install the version last used with this data, or restore a complete backup "
            "made before upgrading. The database has not been migrated."
        )


# What the CLI reads to decide what it needs and where it is. The build checks
# that the bundled CLI still reads each of them (build_runtime.py), and the
# smoke test runs the migrations with no way out to the network.
ENGINE_TYPE = "PRISMA_CLI_QUERY_ENGINE_TYPE"
QUERY_ENGINE = "PRISMA_QUERY_ENGINE_BINARY"
SCHEMA_ENGINE = "PRISMA_SCHEMA_ENGINE_BINARY"
READ_BY_THE_CLI = (ENGINE_TYPE, QUERY_ENGINE, SCHEMA_ENGINE)

MISSING_LIBRARY = re.compile(r"error while loading shared libraries: (\S+?):")
# What the bundled engines are linked against; the build refuses any other
# (build_runtime.py, step_seal).
OPENSSL_LIBRARIES = ("libssl.so.3", "libcrypto.so.3")
OPENSSL_MISSING = (
    "AutoGPT needs OpenSSL 3 (libssl.so.3), and this system does not have it. "
    "Install it (`sudo apt install libssl3` on Debian and Ubuntu, "
    "`sudo dnf install openssl-libs` on Fedora) and start AutoGPT again."
)


def environment(bundle: Bundle, env: dict[str, str]) -> dict[str, str]:
    """`env` (the backend's) over the user's own, minus any Prisma settings
    the user has exported for their own projects: a mirror, extra binary
    targets or an engine path would each send the CLI somewhere else."""
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PRISMA_")
    }
    return {
        **inherited,
        **env,
        ENGINE_TYPE: "binary",
        QUERY_ENGINE: str(bundle.prisma_engine("query-engine")),
        SCHEMA_ENGINE: str(bundle.prisma_engine("schema-engine")),
        "ELECTRON_RUN_AS_NODE": "1",
        "CHECKPOINT_DISABLE": "1",
        "PRISMA_HIDE_UPDATE_MESSAGE": "1",
    }


def require_engines(bundle: Bundle) -> None:
    """Linux only: the engines are linked against the system's OpenSSL 3.
    Without it the CLI reports that an engine "could not be started", which
    tells the user nothing; say what is missing instead."""
    if not sys.platform.startswith("linux"):
        return
    engine = bundle.prisma_engine("schema-engine")
    try:
        result = run_tool([str(engine), "--version"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return  # not what this check is about; the migration will report it
    problem = engine_problem(result.returncode, result.stderr.decode(errors="replace"))
    if problem:
        raise RuntimeError(problem)


def engine_problem(code: int, stderr: str) -> str | None:
    """What to tell the user when an engine does not run, if it is for want
    of a system library."""
    missing = MISSING_LIBRARY.search(stderr)
    if code == 0 or not missing:
        return None
    library = missing.group(1)
    if library in OPENSSL_LIBRARIES:
        return OPENSSL_MISSING
    # Any other library, another OpenSSL included, is named as the loader
    # named it: advice to install OpenSSL 3 would not bring it.
    return f"AutoGPT cannot run its database tools: this system has no {library}."
