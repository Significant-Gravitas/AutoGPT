import asyncio
import datetime
import ipaddress
import socket
import ssl
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from backend.util.link_checkout import broker_client
from backend.util.link_checkout.broker_protocol import BrowserCommand, BrowserOutput
from backend.util.link_checkout.broker_service import create_app
from backend.util.link_checkout.refusals import FIELDS_NOT_READY, CheckoutRefused


def certificates(directory):
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Checkout test CA")])
    now = datetime.datetime.now(datetime.timezone.utc)

    def builder(subject, public_key):
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
        )

    ca = (
        builder(name, ca_key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    (directory / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for purpose, usage in [
        ("server", ExtendedKeyUsageOID.SERVER_AUTH),
        ("client", ExtendedKeyUsageOID.CLIENT_AUTH),
    ]:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        certificate = builder(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, purpose)]),
            key.public_key(),
        ).add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
        if purpose == "server":
            certificate = certificate.add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                ),
                critical=False,
            )
        signed = certificate.sign(ca_key, hashes.SHA256())
        (directory / f"{purpose}.crt").write_bytes(
            signed.public_bytes(serialization.Encoding.PEM)
        )
        (directory / f"{purpose}.key").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )


@pytest.mark.asyncio
async def test_broker_requires_client_certificate_controller_secret_and_tenant(
    tmp_path, monkeypatch
):
    certificates(tmp_path)
    monkeypatch.setattr(
        "backend.util.link_checkout.broker_service.execute_browser",
        AsyncMock(return_value=BrowserOutput(code=0, output="synthetic")),
    )
    monkeypatch.delenv("COPILOT_LINK_PRIVATE_CHECKOUT", raising=False)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    origin = f"https://127.0.0.1:{listener.getsockname()[1]}"
    secret = "synthetic-controller-" * 3
    config = uvicorn.Config(
        create_app("owner", secret.encode()),
        ssl_certfile=str(tmp_path / "server.crt"),
        ssl_keyfile=str(tmp_path / "server.key"),
        ssl_ca_certs=str(tmp_path / "ca.crt"),
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        access_log=False,
        log_config=None,
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        context = ssl.create_default_context(cafile=str(tmp_path / "ca.crt"))
        payload = {"user_id": "owner", "session_id": "chat", "args": ["snapshot", "-i"]}
        async with httpx.AsyncClient(verify=context, trust_env=False) as client:
            with pytest.raises(httpx.HTTPError):
                await client.post(origin + "/v1/browser", json=payload)
        context.load_cert_chain(
            str(tmp_path / "client.crt"), str(tmp_path / "client.key")
        )
        async with httpx.AsyncClient(verify=context, trust_env=False) as client:
            missing = await client.post(origin + "/v1/browser", json=payload)
            assert missing.status_code == 401
            headers = {"Authorization": f"Bearer {secret}"}
            wrong = await client.post(
                origin + "/v1/browser",
                json={**payload, "user_id": "other"},
                headers=headers,
            )
            assert wrong.status_code == 403
            valid = await client.post(
                origin + "/v1/browser", json=payload, headers=headers
            )
            assert valid.status_code == 200
            assert valid.json()["output"] == "synthetic"
            assert valid.headers["cache-control"] == "no-store"
        # The controller's own client, configured the way an operator would.
        (tmp_path / "controller-secret").write_text(secret)
        for name, value in {
            "CHECKOUT_BROKER_URL": origin,
            "CHECKOUT_BROKER_USER_ID": "owner",
            "CHECKOUT_BROKER_CA": str(tmp_path / "ca.crt"),
            "CHECKOUT_BROKER_CLIENT_CERT": str(tmp_path / "client.crt"),
            "CHECKOUT_BROKER_CLIENT_KEY": str(tmp_path / "client.key"),
            "CHECKOUT_BROKER_SECRET_FILE": str(tmp_path / "controller-secret"),
        }.items():
            monkeypatch.setenv(name, value)
        reply = await broker_client.request(
            "browser", BrowserCommand(**{**payload, "args": ["get", "url"]})
        )
        assert reply["output"] == "synthetic"
        # A refusal crosses as its fixed text; any other failure stays generic.
        for raised, expected in [
            (CheckoutRefused(FIELDS_NOT_READY), CheckoutRefused),
            (ValueError("canary broker detail"), RuntimeError),
        ]:
            monkeypatch.setattr(
                "backend.util.link_checkout.broker_service.execute_browser",
                AsyncMock(side_effect=raised),
            )
            with pytest.raises(expected) as failure:
                await broker_client.request("browser", BrowserCommand(**payload))
            assert "canary" not in str(failure.value)
            if expected is CheckoutRefused:
                assert str(failure.value) == FIELDS_NOT_READY
    finally:
        server.should_exit = True
        await serving
        listener.close()
