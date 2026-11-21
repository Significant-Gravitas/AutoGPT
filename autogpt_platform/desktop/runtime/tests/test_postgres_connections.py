import contextlib
import os
import socketserver
import struct
import sys
import threading

import psycopg2
import pytest
from autogpt_desktop import postgres
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.supervisor import Stack

PASSWORD = "synthetic-desktop-connection-test"


def message(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack("!I", len(body) + 4) + body


@contextlib.contextmanager
def postgres_server():
    startups = []
    errors = []

    class Server(socketserver.ThreadingTCPServer):
        def handle_error(self, request, client_address):
            errors.append(sys.exception())

    class Handler(socketserver.StreamRequestHandler):
        timeout = 5

        def handle(self):
            while header := self.rfile.read(4):
                length = struct.unpack("!I", header)[0]
                assert 8 <= length <= 65536
                packet = self.rfile.read(length - 4)
                protocol = struct.unpack("!I", packet[:4])[0]
                if protocol in (80877103, 80877104):
                    self.wfile.write(b"N")
                    continue
                assert protocol == 196608
                fields = packet[4:].rstrip(b"\0").split(b"\0")
                startup = dict(zip(fields[::2], fields[1::2], strict=True))
                startups.append(startup)
                if startup.get(b"options"):
                    self.wfile.write(
                        message(b"E", b"SFATAL\0C42704\0Munknown test option\0\0")
                    )
                    return
                self.wfile.write(message(b"R", struct.pack("!I", 3)))
                assert self.rfile.read(1) == b"p"
                size = struct.unpack("!I", self.rfile.read(4))[0]
                password = self.rfile.read(size - 4)
                if password != PASSWORD.encode() + b"\0":
                    self.wfile.write(
                        message(b"E", b"SFATAL\0C28P01\0Mpassword rejected\0\0")
                    )
                    return
                self.wfile.write(
                    message(b"R", struct.pack("!I", 0))
                    + message(b"S", b"server_version\00017.0\0")
                    + message(b"S", b"client_encoding\0UTF8\0")
                    + message(b"S", b"DateStyle\0ISO, MDY\0")
                    + message(b"S", b"standard_conforming_strings\0on\0")
                    + message(b"Z", b"I")
                )
                assert self.rfile.read(5) == b"X\0\0\0\4"
                return

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01)
        )
        thread.start()
        try:
            yield server.server_address[1], startups
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert not errors


@pytest.fixture(autouse=True)
def isolated_pg_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PG"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("caller", ["readiness", "supervisor"])
@pytest.mark.parametrize(
    "ambient",
    [
        {},
        {"PGSSLMODE": "require"},
        {"PGSSLMODE": "verify-full"},
        {"PGHOSTADDR": "not-an-ip-address"},
        {"PGGSSENCMODE": "require"},
        {"PGCHANNELBINDING": "require"},
        {"PGOPTIONS": "-c desktop_nonexistent_setting=1"},
        {
            "PGSSLMODE": "require",
            "PGHOSTADDR": "not-an-ip-address",
            "PGGSSENCMODE": "require",
            "PGCHANNELBINDING": "require",
            "PGOPTIONS": "-c desktop_nonexistent_setting=1",
            "PGUSER": "wrong-user",
            "PGDATABASE": "wrong-database",
            "PGPASSWORD": "wrong-password",
        },
    ],
)
def test_local_connections_override_host_ssl_and_options(
    monkeypatch, tmp_path, caller, ambient
):
    for key, value in ambient.items():
        monkeypatch.setenv(key, value)
    environment = dict(os.environ)
    with postgres_server() as (port, startups):
        if caller == "readiness":
            assert postgres.is_ready(port, PASSWORD)
        else:
            stack = Stack(Bundle(tmp_path / "runtime"), DataDir(tmp_path / "data"))
            with contextlib.closing(stack.database_connector(port, PASSWORD)()) as conn:
                assert conn.info.host == "127.0.0.1"
                assert conn.info.port == port
                assert conn.info.dbname == "postgres"
                assert not conn.info.ssl_in_use
        assert len(startups) == 1
        assert startups[0][b"user"] == b"postgres"
        assert startups[0][b"database"] == b"postgres"
        assert not startups[0].get(b"options")
    assert dict(os.environ) == environment


@pytest.mark.parametrize("caller", ["readiness", "supervisor"])
def test_a_wrong_local_password_is_still_rejected(tmp_path, caller):
    with postgres_server() as (port, startups):
        if caller == "readiness":
            assert not postgres.is_ready(port, "wrong-password")
        else:
            stack = Stack(Bundle(tmp_path / "runtime"), DataDir(tmp_path / "data"))
            with pytest.raises(psycopg2.OperationalError, match="password rejected"):
                stack.database_connector(port, "wrong-password")()
        assert len(startups) == 1
