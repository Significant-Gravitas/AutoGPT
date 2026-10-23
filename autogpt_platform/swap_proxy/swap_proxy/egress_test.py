import asyncio
import socket

import pytest

from swap_proxy.egress import EgressGuard, is_private, normalize_ip


def _resolving(guard: EgressGuard, answers: dict[str, list[str]]):
    async def resolve(host):
        return answers.get(host)

    guard._resolve = resolve  # type: ignore[method-assign]
    return guard


@pytest.mark.parametrize(
    "ip",
    [
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fd00::1",
        "fe80::1%eth0",
        "::ffff:127.0.0.1",  # IPv4-mapped: judged as the v4 address it reaches
        "::ffff:169.254.169.254",
        # A v4 address inside a public-looking v6 one: NAT64 and 6to4 reach it
        # wherever a translator or a relay exists.
        "64:ff9b::a9fe:a9fe",  # 169.254.169.254
        "64:ff9b::10.0.0.5",
        "64:ff9b:1::a9fe:a9fe",  # NAT64's local-use prefix
        "2002:a9fe:a9fe::",  # 6to4 of 169.254.169.254
        "2002:7f00:1::1",  # 6to4 of 127.0.0.1
        "fec0::1",  # site-local
        "::127.0.0.1",  # IPv4-compatible
        "::a9fe:a9fe",
        "::ffff:0:a9fe:a9fe",  # SIIT IPv4-translated
    ],
)
def test_private_space(ip):
    addr = normalize_ip(ip)
    assert addr is not None and is_private(addr)


@pytest.mark.parametrize(
    "ip",
    [
        "140.82.112.3",
        "8.8.8.8",
        "2606:50c0:8000::154",
        "64:ff9b::8c52:7003",  # NAT64 of 140.82.112.3: public stays reachable
        "2002:8c52:7003::1",  # 6to4 of the same
    ],
)
def test_public_space(ip):
    addr = normalize_ip(ip)
    assert addr is not None and not is_private(addr)


async def test_a_public_answer_is_pinned():
    guard = _resolving(EgressGuard(), {"api.github.com": ["140.82.112.5"]})
    verdict = await guard.check("api.github.com")
    assert verdict.ip == "140.82.112.5" and verdict.refused is None


async def test_a_private_answer_is_refused_with_the_address_for_the_audit():
    guard = _resolving(EgressGuard(), {"metadata.internal": ["169.254.169.254"]})
    verdict = await guard.check("metadata.internal")
    assert (verdict.ip, verdict.refused, verdict.refused_ip) == (
        None,
        "private-range",
        "169.254.169.254",
    )


async def test_a_mixed_answer_connects_only_to_the_public_address():
    guard = _resolving(EgressGuard(), {"rebind.test": ["10.0.0.5", "93.184.216.34"]})
    assert (await guard.check("rebind.test")).ip == "93.184.216.34"


async def test_a_name_that_does_not_resolve_is_refused():
    """Not waved through: mitmproxy would resolve it again by itself."""
    guard = _resolving(EgressGuard(), {})
    assert (await guard.check("nowhere.test")).refused == "unresolvable"


async def test_allow_by_host_and_by_network():
    answers = {"db.internal": ["10.0.0.9"], "other.internal": ["10.0.0.10"]}
    by_host = _resolving(EgressGuard(["db.internal"]), answers)
    assert (await by_host.check("db.internal")).ip == "10.0.0.9"
    assert (await by_host.check("other.internal")).refused == "private-range"
    by_net = _resolving(EgressGuard(["10.0.0.8/30"]), answers)
    assert (await by_net.check("db.internal")).ip == "10.0.0.9"
    assert (await by_net.check("other.internal")).ip == "10.0.0.10"


async def test_literals_resolve_to_themselves_and_ipv4_comes_first(monkeypatch):
    guard = EgressGuard(["127.0.0.0/8", "::1"])
    assert (await guard.check("127.0.0.1")).ip == "127.0.0.1"

    async def both(host, port, **kwargs):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", both)
    assert (await EgressGuard(["127.0.0.0/8", "::1"]).check("dual.test")).ip == (
        "127.0.0.1"
    )


async def test_a_nat64_answer_for_the_metadata_server_is_refused():
    """What a DNS64 resolver would synthesise for a v4-only name."""
    guard = _resolving(EgressGuard(), {"metadata.internal": ["64:ff9b::a9fe:a9fe"]})
    verdict = await guard.check("metadata.internal")
    assert (verdict.ip, verdict.refused) == (None, "private-range")
    # A public v4 address behind the same prefix is dialled in its v6 form.
    guard = _resolving(EgressGuard(), {"api.github.com": ["64:ff9b::8c52:7003"]})
    assert (await guard.check("api.github.com")).ip == "64:ff9b::8c52:7003"


async def test_a_failed_lookup_is_not_remembered(monkeypatch):
    calls = 0

    async def flaky(host, port, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise socket.gaierror("temporary failure in name resolution")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("140.82.112.5", 0))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", flaky)
    guard = EgressGuard()
    assert (await guard.check("api.github.com")).refused == "unresolvable"
    # Recovered a moment later: asked again, not answered from the failure.
    assert (await guard.check("api.github.com")).ip == "140.82.112.5"
    # A good answer is what gets remembered.
    assert (await guard.check("api.github.com")).ip == "140.82.112.5"
    assert calls == 2


@pytest.mark.parametrize(
    "host",
    ["db.internal:.169.254.169.254.nip.io", "db.internal:443", ":db.internal"],
)
async def test_a_name_with_a_colon_is_refused_before_the_allow_list_sees_it(host):
    """``host_in_list`` cuts at the first ':', so this would have matched the
    allowed name and gone wherever the rest of it resolves."""
    guard = _resolving(EgressGuard(["db.internal"]), {host: ["169.254.169.254"]})
    assert (await guard.check(host)).refused == "malformed-host"


async def test_an_ipv6_literal_is_not_a_name_with_a_colon():
    guard = EgressGuard(["::1"])
    assert (await guard.check("::1")).ip == "::1"
