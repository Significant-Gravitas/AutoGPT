"""SendEmailBlock only sends a saved SMTP login to the server its scope names;
a login with no scope is only used in a graph version the running user saved.

It connects only to the addresses the server's name was checked to resolve to,
requires TLS valid for that name, and talks to the server in a worker thread
within a deadline."""

import asyncio
import contextlib
import datetime
import re
import select
import smtplib
import socket
import ssl
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from pydantic import SecretStr

from backend.blocks.email_block import (
    SendEmailBlock,
    SMTPConfig,
    UserPasswordCredentials,
)
from backend.blocks.helpers.smtp import BoundedSMTP, SMTPDeadline, verified_tls_context


def _make_credentials(scope: str | None) -> UserPasswordCredentials:
    return UserPasswordCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="smtp",
        username=SecretStr("user@example.com"),
        password=SecretStr("app-password"),
        scope=scope,
        title="test creds",
    )


def _make_input(smtp_server: str, smtp_port: int = 587) -> SendEmailBlock.Input:
    return SendEmailBlock.Input.model_validate(
        {
            "to_email": "recipient@example.com",
            "subject": "Subject",
            "body": "Body",
            "config": {"smtp_server": smtp_server, "smtp_port": smtp_port},
            "credentials": {
                "provider": "smtp",
                "id": "01234567-89ab-cdef-0123-456789abcdef",
                "type": "user_password",
                "title": "test creds",
            },
        }
    )


async def _run(
    input_data: SendEmailBlock.Input,
    credentials: UserPasswordCredentials,
    *,
    owns_graph: bool = False,
    **run_kwargs: Any,
) -> tuple[dict[str, Any], MagicMock]:
    block = SendEmailBlock()
    send_email = MagicMock(return_value="Email sent successfully")
    db = MagicMock(is_graph_version_owner=AsyncMock(return_value=owns_graph))
    outputs: dict[str, Any] = {}
    with (
        patch.object(SendEmailBlock, "send_email", send_email),
        patch(
            "backend.blocks.email_block.resolve_and_check_blocked",
            new=AsyncMock(return_value=["203.0.113.7"]),
        ),
        patch(
            "backend.blocks.helpers.host_bound_login.get_database_manager_async_client",
            return_value=db,
        ),
    ):
        async for name, value in block.run(
            input_data, credentials=credentials, **run_kwargs
        ):
            outputs[name] = value
    return outputs, send_email


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,smtp_server,smtp_port",
    [
        ("smtp://smtp.example.com", "203.0.113.7", 587),
        (None, "smtp.example.com", 587),
        ("", "smtp.example.com", 587),
        ("smtp.example.com", "smtp.example.com", 587),
        ("smtp://smtp.example.com:465", "smtp.example.com", 587),
        ("smtp://smtp.example.com", "sub.smtp.example.com", 587),
        ("smtp://smtp.example.com", "mail.example.org", 587),
        ("postgresql://smtp.example.com", "smtp.example.com", 587),
    ],
)
async def test_login_not_sent_to_another_server(
    scope: str | None, smtp_server: str, smtp_port: int
):
    outputs, send_email = await _run(
        _make_input(smtp_server, smtp_port), _make_credentials(scope)
    )
    error = outputs["error"]
    if scope is None:
        assert "saved without a server" in error
    else:
        assert f"different server than this workflow's ('{smtp_server}'" in error
    assert "so they were not sent" in error
    assert not re.search(r"\badd\b", error, re.IGNORECASE)
    assert "status" not in outputs
    send_email.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,smtp_server,smtp_port",
    [
        ("smtp://smtp.example.com", "smtp.example.com", 587),
        ("smtp://SMTP.Example.com:587", "smtp.example.com", 587),
        ("smtp://203.0.113.7", "203.0.113.7", 2525),
    ],
)
async def test_login_sent_to_its_own_server(
    scope: str, smtp_server: str, smtp_port: int
):
    outputs, send_email = await _run(
        _make_input(smtp_server, smtp_port), _make_credentials(scope)
    )
    assert outputs == {"status": "Email sent successfully"}
    send_email.assert_called_once()


_RUN_BY_USER = {"user_id": "user-1", "graph_id": "graph-1", "graph_version": 3}


@pytest.mark.asyncio
async def test_login_without_scope_used_on_own_graph():
    """A login with no scope is used in a graph version the running user saved."""
    outputs, send_email = await _run(
        _make_input("smtp.example.com"),
        _make_credentials(None),
        owns_graph=True,
        **_RUN_BY_USER,
    )
    assert outputs == {"status": "Email sent successfully"}
    send_email.assert_called_once()


@pytest.mark.asyncio
async def test_login_without_scope_not_used_on_another_users_graph():
    """A login with no scope is not used in a graph version another user saved."""
    outputs, send_email = await _run(
        _make_input("203.0.113.7"),
        _make_credentials(None),
        owns_graph=False,
        **_RUN_BY_USER,
    )
    assert "saved without a server" in outputs["error"]
    send_email.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["smtp://smtp.example.com", ""])
async def test_login_with_scope_ignores_graph_ownership(scope: str):
    """A login with a scope goes only where its scope says, even on the user's
    own graph; only a login with no scope at all is a legacy one."""
    outputs, send_email = await _run(
        _make_input("203.0.113.7"),
        _make_credentials(scope),
        owns_graph=True,
        **_RUN_BY_USER,
    )
    assert "different server" in outputs["error"]
    send_email.assert_not_called()


def _send_to_local_server(serve) -> Exception | str:
    """Call send_email against a server on localhost that handles one connection."""
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]

        def accept_one():
            connection, _ = listener.accept()
            with connection:
                serve(connection)

        server = threading.Thread(target=accept_one, daemon=True)
        server.start()
        try:
            result: Exception | str = SendEmailBlock.send_email(
                SMTPConfig(smtp_server="127.0.0.1", smtp_port=port),
                "recipient@example.com",
                "Subject",
                "Body",
                _make_credentials(None),
                addresses=["127.0.0.1"],
                deadline=SMTPDeadline(30),
            )
        except (smtplib.SMTPException, OSError) as exc:
            result = exc
        server.join(timeout=10)
        return result


def test_a_reply_that_never_ends_is_cut_off():
    sent = 0

    def never_finish_the_greeting(connection: socket.socket):
        nonlocal sent
        line = b"220-" + b"A" * 8000 + b"\r\n"
        try:
            while sent < 100 * BoundedSMTP.MAX_REPLY_BYTES:
                connection.sendall(line)
                sent += len(line)
        except OSError:
            pass

    result = _send_to_local_server(never_finish_the_greeting)

    assert isinstance(result, smtplib.SMTPServerDisconnected)
    # Socket buffers hold a few MB more than the client read.
    assert sent < 20 * BoundedSMTP.MAX_REPLY_BYTES


def _no_starttls(connection: socket.socket):
    connection.sendall(b"220 mail.example.com ESMTP\r\n")
    connection.recv(1024)  # EHLO
    connection.sendall(b"250-mail.example.com\r\n250-SIZE 1000\r\n250 8BITMIME\r\n")
    connection.recv(1024)  # QUIT
    connection.sendall(b"221 Bye\r\n")


def test_an_ordinary_multiline_reply_is_still_read():
    result = _send_to_local_server(_no_starttls)

    assert isinstance(result, smtplib.SMTPNotSupportedError)


PUBLIC_ADDRESS = "203.0.113.25"  # a documentation address: allowed, routes nowhere
SERVER_NAME = "smtp.example.test"


class _Server:
    """Serves each connection to a port on 127.0.0.1 with ``serve``, in a thread."""

    def __init__(self, serve: Callable[[socket.socket], None]):
        self._serve = serve
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.settimeout(0.05)
        self.port: int = self._listener.getsockname()[1]
        self.connections = 0
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._closing.set()
        self._thread.join(timeout=30)
        self._listener.close()

    def _accept(self) -> None:
        while not self._closing.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            self.connections += 1
            with connection, contextlib.suppress(OSError):
                connection.settimeout(20)
                self._serve(connection)


@contextlib.contextmanager
def _dns(answers: dict[str, list[str]], servers: dict[str, int]) -> Iterator[list[str]]:
    """Name resolution for one test: each name in ``answers`` resolves to its next
    answer, the last one repeating, and each address in ``servers`` reaches the
    test server on that port of 127.0.0.1. Yields the names looked up."""
    real_getaddrinfo = socket.getaddrinfo
    lookups: list[str] = []

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if host in answers:
            address = answers[host][min(lookups.count(host), len(answers[host]) - 1)]
            lookups.append(host)
            tcp = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            return [(*tcp, "", (address, port or 0))]
        if host in servers:
            return real_getaddrinfo("127.0.0.1", servers[host], *args, **kwargs)
        return real_getaddrinfo(host, port, *args, **kwargs)

    with patch("socket.getaddrinfo", getaddrinfo):
        yield lookups


@contextlib.contextmanager
def _smtp_server(serve: Callable[[socket.socket], None]) -> Iterator[_Server]:
    """A test server that ``SERVER_NAME`` resolves to."""
    with (
        _Server(serve) as server,
        _dns({SERVER_NAME: [PUBLIC_ADDRESS]}, {PUBLIC_ADDRESS: server.port}),
    ):
        yield server


async def _send(smtp_server: str, smtp_port: int) -> dict[str, Any]:
    """Run the block against ``smtp_server`` on ``smtp_port``; return its outputs."""
    outputs: dict[str, Any] = {}
    with patch.object(SendEmailBlock, "ALLOWED_SMTP_PORTS", {smtp_port}):
        async for name, value in SendEmailBlock().run(
            _make_input(smtp_server, smtp_port),
            credentials=_make_credentials(f"smtp://{smtp_server}"),
        ):
            outputs[name] = value
    return outputs


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Condition not met within 10 seconds")


@pytest.mark.asyncio
async def test_connects_only_to_the_address_that_was_checked():
    """A name that resolves to another address the second time is resolved once,
    and the connection goes to the address that was checked."""
    with _Server(_no_starttls) as checked, _Server(_no_starttls) as other:
        answers = {"changing.example": [PUBLIC_ADDRESS, "127.0.0.1"]}
        with _dns(answers, {PUBLIC_ADDRESS: checked.port}) as lookups:
            outputs = await _send("changing.example", other.port)

    assert (checked.connections, other.connections) == (1, 0)
    assert lookups == ["changing.example"]
    assert "STARTTLS not supported" in outputs["error"]


def test_each_checked_address_is_tried_in_turn():
    """When one checked address refuses the connection, the next one is tried."""
    with socket.create_server(("127.0.0.1", 0)) as unused:
        refusing_port = unused.getsockname()[1]
    refusing_address = "203.0.113.26"

    with (
        _Server(_no_starttls) as server,
        _dns({}, {refusing_address: refusing_port, PUBLIC_ADDRESS: server.port}),
        pytest.raises(smtplib.SMTPNotSupportedError),
    ):
        SendEmailBlock.send_email(
            SMTPConfig(smtp_server=SERVER_NAME, smtp_port=server.port),
            "recipient@example.com",
            "Subject",
            "Body",
            _make_credentials(None),
            addresses=[refusing_address, PUBLIC_ADDRESS],
            deadline=SMTPDeadline(30),
        )

    assert server.connections == 1


class _CertificateAuthority:
    """Issues certificates for the tests' TLS servers."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._key = ec.generate_private_key(ec.SECP256R1())
        self._name = _name(f"Test CA {directory.name}")
        signs_certificates = x509.KeyUsage(
            digital_signature=False,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,
            crl_sign=True,
            encipher_only=False,
            decipher_only=False,
        )
        self.certificate = (
            _certificate(self._name, self._name, self._key)
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None), critical=True
            )
            .add_extension(signs_certificates, critical=True)
            .sign(self._key, hashes.SHA256())
        )

    def client_context(self) -> ssl.SSLContext:
        """The block's TLS settings, trusting this authority as well."""
        context = verified_tls_context()
        context.load_verify_locations(
            cadata=self.certificate.public_bytes(serialization.Encoding.PEM).decode()
        )
        return context

    def server_context(self, server_name: str) -> ssl.SSLContext:
        key = ec.generate_private_key(ec.SECP256R1())
        issuer_key_id = x509.AuthorityKeyIdentifier.from_issuer_public_key(
            self._key.public_key()
        )
        certificate = (
            _certificate(_name(server_name), self._name, key)
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(server_name)]),
                critical=False,
            )
            .add_extension(issuer_key_id, critical=False)
            .sign(self._key, hashes.SHA256())
        )
        certificate_file = self._directory / f"{server_name}.pem"
        key_file = self._directory / f"{server_name}.key"
        certificate_file.write_bytes(
            certificate.public_bytes(serialization.Encoding.PEM)
        )
        key_file.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate_file, key_file)
        return context


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _certificate(
    subject: x509.Name, issuer: x509.Name, key: ec.EllipticCurvePrivateKey
) -> x509.CertificateBuilder:
    now = datetime.datetime.now(datetime.UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
    )


class _MailHost:
    """Takes one email over STARTTLS, recording the commands and the message."""

    def __init__(self, tls: ssl.SSLContext):
        tls.sni_callback = self._note_server_name
        self._tls = tls
        self.server_names: list[str | None] = []
        self.commands: list[bytes] = []
        self.message = b""

    def _note_server_name(self, _socket: Any, name: str | None, _context: Any) -> None:
        self.server_names.append(name)

    def __call__(self, connection: socket.socket) -> None:
        connection.sendall(b"220 smtp.example.test ESMTP\r\n")
        connection.recv(1024)  # EHLO
        connection.sendall(b"250-smtp.example.test\r\n250 STARTTLS\r\n")
        connection.recv(1024)  # STARTTLS
        connection.sendall(b"220 Ready\r\n")
        with self._tls.wrap_socket(connection, server_side=True) as secure:
            lines = secure.makefile("rb")
            for reply in (
                b"250-smtp.example.test\r\n250 AUTH PLAIN",  # to EHLO
                b"235 Accepted",  # AUTH
                b"250 OK",  # MAIL
                b"250 OK",  # RCPT
                b"354 Go ahead",  # DATA
            ):
                self.commands.append(lines.readline()[:4].upper())
                secure.sendall(reply + b"\r\n")
            message = []
            while (line := lines.readline()) not in (b".\r\n", b""):
                message.append(line)
            self.message = b"".join(message)
            secure.sendall(b"250 Queued\r\n")
            lines.readline()  # QUIT
            secure.sendall(b"221 Bye\r\n")


@pytest.mark.asyncio
async def test_sends_over_tls_valid_for_the_server_name(tmp_path: Path):
    """The email goes to the checked address, over TLS that names the configured
    server and requires a certificate for that name."""
    authority = _CertificateAuthority(tmp_path)
    mail = _MailHost(authority.server_context(SERVER_NAME))

    with (
        _smtp_server(mail) as server,
        patch(
            "backend.blocks.email_block.verified_tls_context",
            authority.client_context,
        ),
    ):
        outputs = await _send(SERVER_NAME, server.port)

    assert outputs == {"status": "Email sent successfully"}
    assert mail.server_names == [SERVER_NAME]
    assert mail.commands == [b"EHLO", b"AUTH", b"MAIL", b"RCPT", b"DATA"]
    assert b"Subject: Subject" in mail.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issued_for,issuer_trusted",
    [("other.example.test", True), (SERVER_NAME, False)],
)
async def test_no_login_without_a_certificate_for_the_server_name(
    tmp_path: Path, issued_for: str, issuer_trusted: bool
):
    """No login is sent to a server whose certificate is for another name, or is
    not from a trusted authority."""
    authority = _CertificateAuthority(tmp_path / "trusted")
    issuer = authority if issuer_trusted else _CertificateAuthority(tmp_path / "other")
    mail = _MailHost(issuer.server_context(issued_for))

    with (
        _smtp_server(mail) as server,
        patch(
            "backend.blocks.email_block.verified_tls_context",
            authority.client_context,
        ),
    ):
        outputs = await _send(SERVER_NAME, server.port)

    assert outputs["error"].startswith(
        f"The TLS certificate of SMTP server '{SERVER_NAME}' could not be verified: "
    )
    assert mail.commands == []


@pytest.mark.asyncio
async def test_a_reply_from_the_server_is_not_passed_on():
    """When the server refuses a command, the error does not repeat its reply."""

    def refuse_starttls(connection: socket.socket) -> None:
        connection.sendall(b"220 mail.example.com ESMTP\r\n")
        connection.recv(1024)  # EHLO
        connection.sendall(b"250-mail.example.com\r\n250 STARTTLS\r\n")
        connection.recv(1024)  # STARTTLS
        connection.sendall(b"454 4.7.0 TLS not available on node 7\r\n")
        connection.recv(1024)  # QUIT
        connection.sendall(b"221 Bye\r\n")

    with _smtp_server(refuse_starttls) as server:
        outputs = await _send(SERVER_NAME, server.port)

    assert outputs == {
        "error": f"SMTP server '{SERVER_NAME}' rejected the request. "
        "Please verify the server address and port."
    }


@pytest.mark.asyncio
async def test_the_event_loop_runs_while_the_server_is_slow():
    """The conversation runs in a worker thread, so the event loop, and every
    other node on it, keeps running while the server takes its time."""
    released = threading.Event()
    released_in_time: list[bool] = []

    def greet_once_released(connection: socket.socket) -> None:
        released_in_time.append(released.wait(timeout=10))
        _no_starttls(connection)

    with _smtp_server(greet_once_released) as server:
        sending = asyncio.create_task(_send(SERVER_NAME, server.port))
        await _until(lambda: server.connections > 0)
        released.set()  # only the event loop can get here
        outputs = await sending

    assert released_in_time == [True]
    assert "STARTTLS not supported" in outputs["error"]


def _trickle_until_the_client_leaves(
    connection: socket.socket, line: bytes, client_left: threading.Event
) -> None:
    """Send ``line`` every 50 ms for up to 20 seconds; set ``client_left`` once
    the client closes the connection."""
    try:
        for _ in range(400):
            connection.sendall(line)
            readable, _, _ = select.select([connection], [], [], 0.05)
            if readable and not connection.recv(1024):
                client_left.set()
                return
    except (ConnectionError, ssl.SSLError):
        client_left.set()


@pytest.mark.asyncio
async def test_the_conversation_ends_at_the_deadline():
    """A server that keeps the conversation going a line at a time, never idle
    long enough for a socket timeout, is cut off when the deadline passes."""
    client_left = threading.Event()

    def never_finish_the_greeting(connection: socket.socket) -> None:
        _trickle_until_the_client_leaves(connection, b"220-still\r\n", client_left)

    with (
        _smtp_server(never_finish_the_greeting) as server,
        patch.object(SendEmailBlock, "SMTP_DEADLINE_SECONDS", 2),
    ):
        outputs = await asyncio.wait_for(_send(SERVER_NAME, server.port), 20)

    assert outputs == {
        "error": f"SMTP server '{SERVER_NAME}' on port {server.port} did not "
        "finish within 2 seconds."
    }
    assert client_left.is_set()


@pytest.mark.asyncio
async def test_the_deadline_also_ends_a_conversation_over_tls(tmp_path: Path):
    """The deadline still reaches the connection after STARTTLS has replaced the
    socket smtplib holds with a TLS one."""
    authority = _CertificateAuthority(tmp_path)
    tls = authority.server_context(SERVER_NAME)
    client_left = threading.Event()

    def stall_after_starttls(connection: socket.socket) -> None:
        connection.sendall(b"220 smtp.example.test ESMTP\r\n")
        connection.recv(1024)  # EHLO
        connection.sendall(b"250-smtp.example.test\r\n250 STARTTLS\r\n")
        connection.recv(1024)  # STARTTLS
        connection.sendall(b"220 Ready\r\n")
        with tls.wrap_socket(connection, server_side=True) as secure:
            secure.recv(1024)  # EHLO
            _trickle_until_the_client_leaves(secure, b"250-still\r\n", client_left)

    with (
        _smtp_server(stall_after_starttls) as server,
        patch(
            "backend.blocks.email_block.verified_tls_context",
            authority.client_context,
        ),
        patch.object(SendEmailBlock, "SMTP_DEADLINE_SECONDS", 2),
    ):
        outputs = await asyncio.wait_for(_send(SERVER_NAME, server.port), 20)

    assert outputs == {
        "error": f"SMTP server '{SERVER_NAME}' on port {server.port} did not "
        "finish within 2 seconds."
    }
    assert client_left.is_set()


@contextlib.contextmanager
def _send_email_ended() -> Iterator[threading.Event]:
    """An event that is set when SendEmailBlock.send_email returns or raises."""
    ended = threading.Event()
    send_email = SendEmailBlock.send_email

    def watched(*args: Any, **kwargs: Any) -> str:
        try:
            return send_email(*args, **kwargs)
        finally:
            ended.set()

    with patch.object(SendEmailBlock, "send_email", staticmethod(watched)):
        yield ended


@pytest.mark.asyncio
async def test_stopping_the_run_ends_the_conversation():
    """A stopped run takes its conversation with it, instead of leaving it to
    run on in the worker thread."""
    client_left = threading.Event()

    def say_nothing(connection: socket.socket) -> None:
        if not connection.recv(1024):
            client_left.set()

    with _smtp_server(say_nothing) as server, _send_email_ended() as send_ended:
        sending = asyncio.create_task(_send(SERVER_NAME, server.port))
        await _until(lambda: server.connections > 0)
        sending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sending

        assert await asyncio.to_thread(client_left.wait, 10)
        assert await asyncio.to_thread(send_ended.wait, 10)
