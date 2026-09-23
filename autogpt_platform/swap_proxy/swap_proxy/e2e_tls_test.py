"""The same stack over TLS, where the rules that matter in production live:
only bound hosts are opened, and a value goes only where the upstream
certificate proves the request is going."""

import asyncio
import datetime
import ipaddress
import logging
import ssl

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from swap_proxy.e2e_test import (
    BOX_A,
    BOX_D,
    TOKEN_A,
    FakeRedis,
    FakeSource,
    Proxy,
    Upstream,
    audit_lines,
    body_of,
    http_get,
    socks5_request,
)

BOUND, ALSO_BOUND, UNBOUND = "localhost", "alt.localhost", "unbound.localhost"


def _make_ca_and_cert(tmp_path, names):
    """A throwaway CA and a server certificate for *names* signed by it."""
    now = datetime.datetime.now(datetime.timezone.utc)

    def build(subject, issuer, public_key, signing_key, *, ca):
        ski = x509.SubjectKeyIdentifier.from_public_key
        aki = x509.AuthorityKeyIdentifier.from_issuer_public_key
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), True)
            .add_extension(ski(public_key), False)
            .add_extension(aki(signing_key.public_key()), False)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=ca,
                    crl_sign=ca,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                True,
            )
        )
        if not ca:
            sans: list[x509.GeneralName] = [x509.DNSName(n) for n in names]
            sans.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
            builder = builder.add_extension(x509.SubjectAlternativeName(sans), False)
        return builder.sign(signing_key, hashes.SHA256())

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test upstream CA")])
    ca_cert = build(ca_name, ca_name, ca_key.public_key(), ca_key, ca=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])])
    cert = build(name, ca_name, key.public_key(), ca_key, ca=False)

    pem = serialization.Encoding.PEM
    ca_path, cert_path, key_path = (
        tmp_path / "upstream-ca.pem",
        tmp_path / "upstream.pem",
        tmp_path / "upstream.key",
    )
    ca_path.write_bytes(ca_cert.public_bytes(pem))
    cert_path.write_bytes(cert.public_bytes(pem))
    key_path.write_bytes(
        key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_path, cert_path, key_path


class TLSUpstream(Upstream):
    def __init__(self, cert_path, key_path):
        super().__init__()
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(cert_path, key_path)

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0, ssl=self._context
        )
        self.port = self._server.sockets[0].getsockname()[1]


async def tls_request(proxy_port, box, dest_port, sni, request, trust) -> tuple:
    """SOCKS5 to the upstream, then TLS inside the tunnel trusting *trust*.
    Returns the response and the issuer of the certificate that was served."""
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    try:
        u, p = box[0].encode(), box[1].encode()
        writer.write(b"\x05\x01\x02")
        assert await reader.readexactly(2) == b"\x05\x02"
        writer.write(b"\x01" + bytes([len(u)]) + u + bytes([len(p)]) + p)
        assert await reader.readexactly(2) == b"\x01\x00"
        # By address, as a box that resolved the name itself would.
        writer.write(b"\x05\x01\x00\x01" + bytes([127, 0, 0, 1]))
        writer.write(dest_port.to_bytes(2, "big"))
        assert (await reader.readexactly(10))[1] == 0
        context = ssl.create_default_context(cafile=str(trust))
        context.set_alpn_protocols(["http/1.1"])
        await writer.start_tls(context, server_hostname=sni)
        issuer = dict(pair[0] for pair in writer.get_extra_info("peercert")["issuer"])[
            "commonName"
        ]
        writer.write(request)
        await writer.drain()
        return await asyncio.wait_for(reader.read(), timeout=10), issuer
    finally:
        writer.close()


@pytest.fixture
async def tls_stack(tmp_path):
    ca_path, cert_path, key_path = _make_ca_and_cert(
        tmp_path, [BOUND, ALSO_BOUND, UNBOUND]
    )
    redis, source = FakeRedis(), FakeSource()
    source.hosts = (BOUND, ALSO_BOUND)
    upstream = TLSUpstream(cert_path, key_path)
    await upstream.start()
    proxy = Proxy(redis, source, allow=["127.0.0.0/8"], allow_insecure_swap=False)
    await proxy.start(tmp_path / "conf")
    proxy.master.options.update(ssl_verify_upstream_trusted_ca=str(ca_path))
    redis.add_box(*BOX_A, "user-a", "session:s-a")
    redis.add_box(*BOX_D, "user-a", "block:user-a", swaps=False)
    mitm_ca = tmp_path / "conf" / "mitmproxy-ca-cert.pem"
    yield proxy, upstream, ca_path, mitm_ca
    await proxy.stop()
    await upstream.stop()


BEARER = {"Authorization": "Bearer hsurr:github"}


async def test_a_bound_host_is_opened_and_the_token_swapped(tls_stack):
    proxy, upstream, _, mitm_ca = tls_stack
    raw, issuer = await tls_request(
        proxy.port, BOX_A, upstream.port, BOUND, http_get(headers=BEARER), mitm_ca
    )
    assert issuer == "mitmproxy"
    assert upstream.seen[0]["headers"]["authorization"] == f"Bearer {TOKEN_A}"
    assert body_of(raw)["headers"]["authorization"] == "Bearer hsurr:github"


async def test_an_unbound_host_is_never_opened(tls_stack):
    """The box talks TLS to the real server; the proxy only moves bytes."""
    proxy, upstream, upstream_ca, _ = tls_stack
    request = http_get(headers=BEARER, host=UNBOUND)
    raw, issuer = await tls_request(
        proxy.port, BOX_A, upstream.port, UNBOUND, request, upstream_ca
    )
    assert issuer == "test upstream CA"
    assert body_of(raw)["headers"]["authorization"] == "Bearer hsurr:github"


async def test_a_host_header_the_certificate_was_not_verified_for_gets_nothing(
    tls_stack,
):
    """Both names are bound, but the connection proved only one of them."""
    proxy, upstream, _, mitm_ca = tls_stack
    request = http_get(headers=BEARER, host=ALSO_BOUND)
    await tls_request(proxy.port, BOX_A, upstream.port, BOUND, request, mitm_ca)
    assert upstream.seen[0]["headers"]["authorization"] == "Bearer hsurr:github"


async def test_an_upstream_whose_certificate_does_not_verify_gets_nothing(tls_stack):
    proxy, upstream, _, mitm_ca = tls_stack
    proxy.master.options.update(ssl_verify_upstream_trusted_ca=None)
    try:
        raw, _ = await tls_request(
            proxy.port, BOX_A, upstream.port, BOUND, http_get(headers=BEARER), mitm_ca
        )
    except (ssl.SSLError, ConnectionError, asyncio.IncompleteReadError):
        raw = b""
    assert upstream.seen == [] and TOKEN_A.encode() not in raw
    assert b"200 OK" not in raw


async def test_plain_http_proves_nothing_so_nothing_is_swapped(tmp_path, caplog):
    """No certificate, no proof of where the request is going: a token must
    not cross the network in the clear to whoever answers on port 80."""
    redis, source, upstream = FakeRedis(), FakeSource(), Upstream()
    await upstream.start()
    proxy = Proxy(redis, source, allow=["127.0.0.0/8"], allow_insecure_swap=False)
    await proxy.start(tmp_path)
    redis.add_box(*BOX_A, "user-a", "session:s-a")
    try:
        with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
            await socks5_request(
                proxy.port, *BOX_A, BOUND, upstream.port, http_get(headers=BEARER)
            )
    finally:
        await proxy.stop()
        await upstream.stop()
    assert upstream.seen[0]["headers"]["authorization"] == "Bearer hsurr:github"
    assert source.asked == []
    assert '"reason": "unverified-destination"' in caplog.text


async def test_a_cold_start_outage_still_opens_a_swapping_boxs_connection(
    tls_stack, caplog
):
    """No bindings table has ever arrived, so whether the host is bound is
    unknown.  Tunnelled, the hooks that refuse what cannot be scrubbed would
    never run, and a value the box stored there earlier would come back."""
    proxy, upstream, _, mitm_ca = tls_stack
    proxy.source.down = True
    upstream.stored = TOKEN_A
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        raw, issuer = await tls_request(
            proxy.port, BOX_A, upstream.port, BOUND, http_get(), mitm_ca
        )
    assert issuer == "mitmproxy"
    assert TOKEN_A.encode() not in raw and b"200 OK" not in raw
    events = [(line["event"], line.get("reason")) for line in audit_lines(caplog)]
    assert events == [("refused-response", "resolver-unavailable")]


async def test_a_cold_start_outage_leaves_a_box_that_does_not_swap_alone(tls_stack):
    """Nothing of its user's can be swapped in for it: no reason to read it."""
    proxy, upstream, upstream_ca, _ = tls_stack
    proxy.source.down = True
    raw, issuer = await tls_request(
        proxy.port, BOX_D, upstream.port, BOUND, http_get(), upstream_ca
    )
    assert issuer == "test upstream CA" and b"200 OK" in raw
