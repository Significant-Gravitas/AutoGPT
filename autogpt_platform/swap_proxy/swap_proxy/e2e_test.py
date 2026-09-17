"""The whole proxy, over real sockets: a SOCKS5 client on one side, an HTTP
server on the other, mitmproxy with the addon in between.

This is where the tenant boundary is tested adversarially: what a box can and
cannot get by presenting credentials that are wrong, stale, or another's.
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional

import pytest

from swap_proxy.__main__ import build_master
from swap_proxy.addon import SwapProxyAddon
from swap_proxy.egress import EgressGuard
from swap_proxy.owners import CREDENTIAL_KEY_PREFIX, OwnerDirectory
from swap_proxy.source import SourceUnavailable
from swap_proxy.swap import Credential, host_in_list

TOKEN_A = "ghp_userAsecretvalue0001"
TOKEN_B = "ghp_userBsecretvalue0002"
UPSTREAM_HOST = "localhost"


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, name):
        return self.store.get(name)

    def add_box(
        self,
        username: str,
        secret: str,
        user_id: Optional[str],
        owner: str,
        swaps: Optional[bool] = None,
    ):
        self.store[CREDENTIAL_KEY_PREFIX + username] = json.dumps(
            {
                "owner": owner,
                "user_id": user_id,
                "swaps": user_id is not None if swaps is None else swaps,
                "sandbox_id": "sb-" + username[-4:],
                "secret_sha256": hashlib.sha256(secret.encode()).hexdigest(),
            }
        )


class FakeSource:
    """github is bound to the upstream's name; each user has their own token."""

    def __init__(self):
        self.tokens = {"user-a": TOKEN_A, "user-b": TOKEN_B}
        self.hosts: tuple[str, ...] = (UPSTREAM_HOST,)
        self.down = False
        self.asked: list[tuple[str, str, str]] = []

    async def bound_names(self, host):
        if self.down:
            raise SourceUnavailable("down")
        return {"github"} if host_in_list(host, self.hosts) else set()

    async def resolve(self, user_id, name, host):
        self.asked.append((user_id, name, host))
        if self.down:
            raise SourceUnavailable("down")
        token = self.tokens.get(user_id)
        if name != "github" or token is None or not host_in_list(host, self.hosts):
            return None
        return Credential("github", {"access_token": token}, self.hosts)


class Upstream:
    """Answers every request with what it received, as JSON."""

    def __init__(self):
        self.seen: list[dict] = []
        self.port = 0

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            lines = head.split("\r\n")
            method, path, _ = lines[0].split(" ")
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            length = int(
                headers.get("content-length", headers.get("Content-Length", 0))
            )
            body = (await reader.readexactly(length)).decode() if length else ""
            received = {
                "method": method,
                "path": path,
                "headers": headers,
                "body": body,
            }
            self.seen.append(received)
            payload = json.dumps(received).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        finally:
            writer.close()


class Proxy:
    def __init__(self, redis, source, allow, allow_insecure_swap=True):
        self.redis, self.source = redis, source
        self._addon = SwapProxyAddon(
            OwnerDirectory(redis),
            source,
            EgressGuard(allow),
            allow_insecure_swap=allow_insecure_swap,
        )

    async def start(self, tmp_path):
        self.master = build_master(self._addon, "127.0.0.1", 0, str(tmp_path))
        self._task = asyncio.create_task(self.master.run())
        server = self.master.addons.get("proxyserver")
        assert server is not None
        for _ in range(200):
            addrs = server.listen_addrs()
            if addrs and addrs[0][1]:
                self.port = addrs[0][1]
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("proxy did not start")

    async def stop(self):
        self.master.shutdown()
        await self._task


async def socks5_request(
    proxy_port: int,
    username: str,
    secret: str,
    dest_host: str,
    dest_port: int,
    request: bytes,
) -> Optional[bytes]:
    """Speak SOCKS5 by hand; ``None`` means the proxy refused the connection."""
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    try:
        writer.write(b"\x05\x01\x02")  # one method offered: username/password
        if await reader.readexactly(2) != b"\x05\x02":
            return None
        u, p = username.encode(), secret.encode()
        writer.write(b"\x01" + bytes([len(u)]) + u + bytes([len(p)]) + p)
        if await reader.readexactly(2) != b"\x01\x00":
            return None
        h = dest_host.encode()
        writer.write(
            b"\x05\x01\x00\x03" + bytes([len(h)]) + h + dest_port.to_bytes(2, "big")
        )
        reply = await reader.readexactly(10)
        if reply[1] != 0:
            return None
        writer.write(request)
        await writer.drain()
        return await asyncio.wait_for(reader.read(), timeout=10)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    finally:
        writer.close()


def http_get(path="/user", headers=None, host=UPSTREAM_HOST) -> bytes:
    lines = [f"GET {path} HTTP/1.1", f"Host: {host}", "Connection: close"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def body_of(raw: Optional[bytes]) -> dict:
    assert raw, "no response through the proxy"
    return json.loads(raw.split(b"\r\n\r\n", 1)[1])


@pytest.fixture
async def stack(tmp_path):
    redis, source, upstream = FakeRedis(), FakeSource(), Upstream()
    await upstream.start()
    proxy = Proxy(redis, source, allow=["127.0.0.0/8", "::1"])
    await proxy.start(tmp_path)
    redis.add_box("box-" + "a" * 16, "secret-a", "user-a", "session:s-a")
    redis.add_box("box-" + "b" * 16, "secret-b", "user-b", "expert:e-b")
    redis.add_box("box-" + "c" * 16, "secret-c", None, "session:anonymous")
    redis.add_box("box-" + "d" * 16, "secret-d", "user-a", "block:user-a", swaps=False)
    yield proxy, upstream, source
    await proxy.stop()
    await upstream.stop()


BOX_A = ("box-" + "a" * 16, "secret-a")
BOX_B = ("box-" + "b" * 16, "secret-b")
BOX_C = ("box-" + "c" * 16, "secret-c")
BOX_D = ("box-" + "d" * 16, "secret-d")
BEARER = {"Authorization": "Bearer hsurr:github"}


async def test_a_placeholder_becomes_the_owners_token_and_the_echo_is_scrubbed(stack):
    proxy, upstream, _ = stack
    raw = await socks5_request(
        proxy.port, *BOX_A, UPSTREAM_HOST, upstream.port, http_get(headers=BEARER)
    )
    # The provider received the real token...
    assert upstream.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN_A}"
    # ...and the box, which got its own request echoed back, did not.
    assert TOKEN_A.encode() not in (raw or b"")
    assert body_of(raw)["headers"]["Authorization"] == "Bearer hsurr:github"


async def test_each_box_gets_its_own_users_token(stack):
    proxy, upstream, _ = stack
    for box in (BOX_A, BOX_B):
        await socks5_request(
            proxy.port, *box, UPSTREAM_HOST, upstream.port, http_get(headers=BEARER)
        )
    sent = [seen["headers"]["Authorization"] for seen in upstream.seen]
    assert sent == [f"Bearer {TOKEN_A}", f"Bearer {TOKEN_B}"]


@pytest.mark.parametrize(
    "username, secret",
    [
        (BOX_A[0], "wrong-secret"),
        (BOX_A[0], BOX_B[1]),  # another box's secret
        ("box-" + "f" * 16, "secret-a"),  # a username nobody minted
        ("e2b:egress:cred:" + BOX_A[0], "secret-a"),  # not a username at all
        (BOX_A[0], ""),
    ],
)
async def test_a_connection_nobody_vouches_for_is_refused_at_the_door(
    stack, username, secret
):
    proxy, upstream, _ = stack
    raw = await socks5_request(
        proxy.port, username, secret, UPSTREAM_HOST, upstream.port, http_get()
    )
    assert raw is None and upstream.seen == []


async def test_no_authentication_is_not_an_offer_the_proxy_accepts(stack):
    proxy, _, _ = stack
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    writer.write(b"\x05\x01\x00")  # "no authentication" only
    assert (await reader.readexactly(2))[1] == 0xFF
    writer.close()


async def test_a_rotated_credential_stops_opening_the_door(stack):
    proxy, upstream, _ = stack
    del proxy.redis.store[CREDENTIAL_KEY_PREFIX + BOX_A[0]]
    raw = await socks5_request(
        proxy.port, *BOX_A, UPSTREAM_HOST, upstream.port, http_get()
    )
    assert raw is None


@pytest.mark.parametrize("box", [BOX_C, BOX_D], ids=["no user", "a block's box"])
async def test_a_box_that_does_not_swap_egresses_with_its_placeholder_intact(
    stack, box
):
    """A block runs a graph someone else may have written: it has a user, and
    still must not get to act with that user's accounts."""
    proxy, upstream, source = stack
    username, secret = box
    request = http_get(headers=BEARER)
    await socks5_request(
        proxy.port, username, secret, UPSTREAM_HOST, upstream.port, request
    )
    assert upstream.seen[0]["headers"]["Authorization"] == "Bearer hsurr:github"
    assert source.asked == []


async def test_a_placeholder_for_an_unbound_host_goes_out_literally(stack, caplog):
    proxy, upstream, source = stack
    source.hosts = ("api.github.com",)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await socks5_request(
            proxy.port, *BOX_A, UPSTREAM_HOST, upstream.port, http_get(headers=BEARER)
        )
    assert upstream.seen[0]["headers"]["Authorization"] == "Bearer hsurr:github"
    assert source.asked == []
    (line,) = [json.loads(r.message) for r in caplog.records]
    assert (line["event"], line["reason"]) == ("refused", "unbound-host")
    assert line["owner"] == "session:s-a" and line["placeholder"] == "hsurr:github"


async def test_a_backend_that_cannot_be_asked_means_no_swap(stack):
    proxy, upstream, source = stack
    source.down = True
    await socks5_request(
        proxy.port, *BOX_A, UPSTREAM_HOST, upstream.port, http_get(headers=BEARER)
    )
    assert upstream.seen[0]["headers"]["Authorization"] == "Bearer hsurr:github"


async def test_the_audit_names_the_swap_and_never_the_value(stack, caplog):
    proxy, upstream, _ = stack
    with caplog.at_level(logging.INFO):
        await socks5_request(
            proxy.port, *BOX_A, UPSTREAM_HOST, upstream.port, http_get(headers=BEARER)
        )
    audit = [
        json.loads(r.message) for r in caplog.records if r.name == "swap_proxy.audit"
    ]
    assert audit == [
        {
            "ts": audit[0]["ts"],
            "event": "swapped",
            "owner": "session:s-a",
            "user_id": "user-a",
            "sandbox_id": "sb-aaaa",
            "host": UPSTREAM_HOST,
            "placeholder": "hsurr:github",
        }
    ]
    assert TOKEN_A not in caplog.text


async def test_private_address_space_is_out_of_reach(tmp_path):
    """The same stack with nothing allowed: the upstream on loopback stands in
    for a metadata server or a neighbouring service."""
    redis, upstream = FakeRedis(), Upstream()
    await upstream.start()
    proxy = Proxy(redis, FakeSource(), allow=[])
    await proxy.start(tmp_path)
    redis.add_box(*BOX_A, "user-a", "session:s-a")
    try:
        for dest in ("localhost", "127.0.0.1", "::ffff:127.0.0.1"):
            raw = await socks5_request(
                proxy.port, *BOX_A, dest, upstream.port, http_get()
            )
            assert not raw or b"200 OK" not in raw
        assert upstream.seen == []
    finally:
        await proxy.stop()
        await upstream.stop()
