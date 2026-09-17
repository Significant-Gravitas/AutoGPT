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
    ],
)
def test_private_space(ip):
    addr = normalize_ip(ip)
    assert addr is not None and is_private(addr)


@pytest.mark.parametrize("ip", ["140.82.112.3", "8.8.8.8", "2606:50c0:8000::154"])
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
