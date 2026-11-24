"""One shim process's use of the stored sign-in, for tests/test_stored_sign_in.py
to run as real processes:

    python -m tests.stored_sign_in_process <platform url> <folder> store <access> <refresh>
    python -m tests.stored_sign_in_process <platform url> <folder> refresh <refused access>
    python -m tests.stored_sign_in_process <platform url> <folder> read

`refresh` is the daemon's own refresh after a 401 on the access token named.
Each prints its answer as one line of JSON.

The keychain is FileKeyring below, which the test names in
PYTHON_KEYRING_BACKEND: one SQLite file the processes share, at
SIGN_IN_TEST_KEYCHAIN, so none of them reads or writes the keychain of
whoever runs the tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import keyring.errors
from keyring.backend import KeyringBackend


class FileKeyring(KeyringBackend):
    """A keychain several processes share: SQLite does its own locking."""

    priority = 1  # type: ignore[assignment]

    @contextlib.contextmanager
    def _passwords(self):
        connection = sqlite3.connect(os.environ["SIGN_IN_TEST_KEYCHAIN"], timeout=60)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS passwords "
                    "(service TEXT, username TEXT, password TEXT, PRIMARY KEY (service, username))"
                )
                yield connection
        finally:
            connection.close()

    def get_password(self, service: str, username: str) -> str | None:
        with self._passwords() as passwords:
            row = passwords.execute(
                "SELECT password FROM passwords WHERE service = ? AND username = ?",
                (service, username),
            ).fetchone()
        return row[0] if row else None

    def set_password(self, service: str, username: str, password: str) -> None:
        with self._passwords() as passwords:
            passwords.execute(
                "INSERT OR REPLACE INTO passwords VALUES (?, ?, ?)", (service, username, password)
            )

    def delete_password(self, service: str, username: str) -> None:
        with self._passwords() as passwords:
            removed = passwords.execute(
                "DELETE FROM passwords WHERE service = ? AND username = ?", (service, username)
            ).rowcount
        if not removed:
            raise keyring.errors.PasswordDeleteError("not found")


def main(platform_url: str, folder: str, action: str, *arguments: str) -> None:
    from autogpt_local_executor import auth
    from autogpt_local_executor.audit import AuditWriter
    from autogpt_local_executor.auth import KeychainTokenStore
    from autogpt_local_executor.config import ShimConfig
    from autogpt_local_executor.daemon import ShimDaemon

    # What the process logs goes to stderr, where the test reads it.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    # The test holds the first exchange's answer until the second process is
    # up, and decides when either has waited too long: a slow start of the
    # second must not run the first one's exchange, or its wait, out of time.
    auth.EXCHANGE_SECONDS = auth.TURN_WAIT_SECONDS = 600.0
    config = ShimConfig(
        platform_url=platform_url,
        session_id="session-1",
        allowed_root=Path(folder),
        audit_log_path=Path(folder) / "audit.log",
    )
    store = KeychainTokenStore(config)
    answer: dict[str, object]
    if action == "store":
        asyncio.run(store.store_tokens(*arguments))
        answer = {"stored": True}
    elif action == "refresh":
        daemon = ShimDaemon(config, token_store=store, audit=MagicMock(spec=AuditWriter))
        try:
            token = asyncio.run(
                daemon._get_access_token(refresh_on_fail=True, rejected_token=arguments[0])
            )
        except Exception as exc:
            answer = {"error": type(exc).__name__}
        else:
            answer = {"token": token}
    else:
        answer = {
            "access_token": asyncio.run(store.get_access_token()),
            "refresh_token": asyncio.run(store.get_refresh_token()),
        }
    print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
