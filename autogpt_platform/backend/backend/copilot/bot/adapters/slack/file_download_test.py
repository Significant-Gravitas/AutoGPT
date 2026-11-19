"""Tests for downloading the files a Slack message carries.

Every download runs through the platform's real request client against local
servers that answer for Slack's file host and for every other host name used
here, over HTTPS (with a certificate the test issues) and plain HTTP. The
client's resolver is pointed at them and records each name it is asked for,
and the servers record each request, so a request that should never be made
shows up in the record instead of reaching the network.
"""

import datetime
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from backend.copilot.bot.adapters.base import MessageContext
from backend.util import request as request_module
from backend.util.request import ResponseTooLargeError

from .adapter import SlackAdapter
from .file_download import download

SLACK = "files.slack.com"
OUTSIDE = "outside.example"
LOOKALIKE = "files.slack.com.outside.example"
TOKEN = "xoxb-test"

NOTES = "/files-pri/T1-F1/download/notes.txt"
MOVED = "/files-pri/T1-F2/download/moved.txt"
LARGE = "/files-pri/T1-F3/download/large.bin"
ELSEWHERE = "/collect"

_BODIES = {NOTES: b"slack file", LARGE: b"x" * 4096}


class FakeHosts:
    """The local servers' ports, and every lookup and request they saw."""

    def __init__(self) -> None:
        self.https_port = 0
        self.http_port = 0
        self.lookups: list[str] = []
        self.requests: list[tuple[str, str, str | None]] = []

    def url(self, template: str) -> str:
        return template.format(https=self.https_port, http=self.http_port)


@pytest.fixture(scope="module")
def tls(tmp_path_factory: pytest.TempPathFactory) -> tuple[ssl.SSLContext, Path]:
    """A server TLS context for the test hosts, and the CA file that trusts it."""
    ca_file, cert_file, key_file = _issue_certificates(tmp_path_factory.mktemp("tls"))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    return server, ca_file


@pytest.fixture
async def hosts(
    monkeypatch: pytest.MonkeyPatch, tls: tuple[ssl.SSLContext, Path]
) -> AsyncIterator[FakeHosts]:
    server_context, ca_file = tls
    record = FakeHosts()

    async def answer(request: web.Request) -> web.StreamResponse:
        record.requests.append(
            (request.host, request.path, request.headers.get("Authorization"))
        )
        if request.path == MOVED:
            raise web.HTTPFound(record.url(f"https://{OUTSIDE}:{{https}}{ELSEWHERE}"))
        return web.Response(body=_BODIES.get(request.path, b"outside content"))

    async def resolve(hostname: str) -> list[str]:
        record.lookups.append(hostname)
        return ["127.0.0.1"]

    # The client checks and pins the addresses this returns, as it would DNS's.
    monkeypatch.setattr(request_module, "resolve_and_check_blocked", resolve)
    # The client's ``ssl.create_default_context()`` trusts the CAs in this file.
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))

    https_runner, record.https_port = await _serve(answer, server_context)
    http_runner, record.http_port = await _serve(answer, None)
    try:
        yield record
    finally:
        await https_runner.cleanup()
        await http_runner.cleanup()


async def test_a_slack_hosted_file_is_downloaded_with_the_bot_token(
    hosts: FakeHosts,
):
    content = await download(
        _hosted(hosts.url(f"https://{SLACK}:{{https}}{NOTES}")), TOKEN, max_bytes=1024
    )

    assert content == b"slack file"
    assert hosts.requests == [(SLACK, NOTES, f"Bearer {TOKEN}")]


@pytest.mark.parametrize(
    "marker",
    [{"is_external": True}, {"mode": "external"}, {"filetype": "remote"}],
    ids=["is_external", "mode", "filetype"],
)
@pytest.mark.parametrize("host", [OUTSIDE, SLACK])
async def test_a_remote_file_is_never_requested(
    hosts: FakeHosts, marker: dict[str, Any], host: str
):
    # A remote file's URL is the one the app that added it chose, so it is
    # not fetched with the token even when it names Slack's own file host.
    file_obj = {**_hosted(hosts.url(f"https://{host}:{{https}}{NOTES}")), **marker}

    with pytest.raises(ValueError):
        await download(file_obj, TOKEN, max_bytes=1024)

    assert hosts.lookups == []
    assert hosts.requests == []


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(f"https://{OUTSIDE}:{{https}}{ELSEWHERE}", id="other-host"),
        pytest.param(f"https://{LOOKALIKE}:{{https}}{ELSEWHERE}", id="lookalike"),
        pytest.param(f"https://{OUTSIDE}:{{https}}/{SLACK}{ELSEWHERE}", id="in-path"),
        pytest.param(f"https://{SLACK}@{OUTSIDE}:{{https}}{ELSEWHERE}", id="userinfo"),
        # Read as userinfo by urlparse; the client folds the backslash into a
        # slash and connects to the host before it.
        pytest.param(
            f"https://{OUTSIDE}:{{https}}\\@{SLACK}{ELSEWHERE}", id="backslash"
        ),
        pytest.param("https://169.254.169.254/latest/meta-data/", id="internal"),
        pytest.param("", id="no-url"),
    ],
)
async def test_a_url_off_slacks_file_host_is_never_requested(
    hosts: FakeHosts, template: str
):
    with pytest.raises(ValueError):
        await download(_hosted(hosts.url(template)), TOKEN, max_bytes=1024)

    assert hosts.lookups == []
    assert hosts.requests == []


async def test_plain_http_is_refused_even_on_slacks_file_host(hosts: FakeHosts):
    with pytest.raises(ValueError):
        await download(
            _hosted(hosts.url(f"http://{SLACK}:{{http}}{NOTES}")),
            TOKEN,
            max_bytes=1024,
        )

    assert hosts.lookups == []
    assert hosts.requests == []


async def test_a_redirect_off_slack_is_not_followed(hosts: FakeHosts):
    with pytest.raises(ValueError):
        await download(
            _hosted(hosts.url(f"https://{SLACK}:{{https}}{MOVED}")),
            TOKEN,
            max_bytes=1024,
        )

    # Slack's file host was asked once, with the token; the redirect's target
    # was never looked up, let alone asked.
    assert hosts.requests == [(SLACK, MOVED, f"Bearer {TOKEN}")]
    assert hosts.lookups == [SLACK]


async def test_the_size_cap_holds_whatever_size_the_event_declared(
    hosts: FakeHosts,
):
    file_obj = {**_hosted(hosts.url(f"https://{SLACK}:{{https}}{LARGE}")), "size": 10}

    with pytest.raises(ResponseTooLargeError):
        await download(file_obj, TOKEN, max_bytes=1024)


async def test_a_message_keeps_its_slack_file_and_skips_a_remote_one(
    hosts: FakeHosts,
):
    """A DM carrying an upload and a remote file, through the adapter: the
    upload reaches the turn, the remote file is skipped with a note, and only
    Slack's file host is ever contacted."""
    adapter = SlackAdapter(MagicMock())
    client = MagicMock()
    client.token = TOKEN
    client.users_info = AsyncMock(
        return_value={"user": {"profile": {"display_name": "Ada"}}}
    )
    client.auth_test = AsyncMock(return_value={"user_id": "UBOT"})
    adapter._clients["T1"] = client
    turns: list[MessageContext] = []

    async def on_message(ctx: MessageContext, _adapter: object) -> None:
        turns.append(ctx)

    adapter.on_message(on_message)
    remote_url = hosts.url(f"https://{OUTSIDE}:{{https}}{ELSEWHERE}")

    await adapter._dispatch_event(
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "1.0",
            "user": "U1",
            "team": "T1",
            "text": "the notes, and the plan",
            "files": [
                _hosted(hosts.url(f"https://{SLACK}:{{https}}{NOTES}")),
                {
                    "id": "F9",
                    "name": "plan",
                    "mode": "external",
                    "is_external": True,
                    "external_type": "app",
                    "filetype": "remote",
                    "url_private": remote_url,
                    "external_url": remote_url,
                },
            ],
        }
    )

    (ctx,) = turns
    assert [(a.filename, a.content) for a in ctx.attachments] == [
        ("notes.txt", b"slack file")
    ]
    assert ctx.skipped_attachments == (("plan", "couldn't be downloaded"),)
    assert hosts.requests == [(SLACK, NOTES, f"Bearer {TOKEN}")]


def _hosted(url: str) -> dict[str, Any]:
    """A file Slack hosts, shaped as the Events API sends it."""
    return {
        "id": "F1",
        "name": "notes.txt",
        "mimetype": "text/plain",
        "filetype": "text",
        "mode": "hosted",
        "is_external": False,
        "size": 10,
        "url_private": url,
        "url_private_download": url,
    }


async def _serve(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ssl_context: ssl.SSLContext | None,
) -> tuple[web.AppRunner, int]:
    app = web.Application()
    app.router.add_get("/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=ssl_context).start()
    return runner, runner.addresses[0][1]


def _issue_certificates(directory: Path) -> tuple[Path, Path, Path]:
    """Issue a throwaway CA and a server certificate it signs for every host
    name used here, as the CA, certificate and key files.

    Each extension is set as RFC 5280 asks: Python 3.13 verifies strictly by
    default.
    """
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])
    ca = (
        _certificate(ca_name, ca_name, ca_key, ca_key, now)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(_key_usage(signs_certificates=True), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, SLACK)])
    certificate = (
        _certificate(name, ca_name, key, ca_key, now)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(host) for host in (SLACK, OUTSIDE, LOOKALIKE)]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(signs_certificates=False), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    files = (directory / "ca.pem", directory / "cert.pem", directory / "key.pem")
    files[0].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    files[1].write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    files[2].write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return files


def _certificate(
    subject: x509.Name,
    issuer: x509.Name,
    key: ec.EllipticCurvePrivateKey,
    issuer_key: ec.EllipticCurvePrivateKey,
    now: datetime.datetime,
) -> x509.CertificateBuilder:
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()),
            critical=False,
        )
    )


def _key_usage(*, signs_certificates: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=not signs_certificates,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=signs_certificates,
        crl_sign=signs_certificates,
        encipher_only=False,
        decipher_only=False,
    )
