#!/usr/bin/env python3
"""TEMPORARY evidence for #17298; removed before merge.

Watches every network interface of the namespace it runs in (run it as root
inside the frontend container's, `nsenter -t <pid> -n`: the container sits on
two networks) until a stop file appears, then writes a JSON summary of the
traffic that left for addresses outside the private ranges: each TCP
connection, named by its TLS ClientHello's SNI or its HTTP Host header, the
bytes received back, and the DNS questions sent out.

    frontend_egress_sniffer.py <summary.json> <stop-file>
"""

import collections
import ipaddress
import json
import os
import socket
import struct
import sys

ETH_P_ALL = 0x0003
HTTP_METHODS = (b"GET", b"POST", b"HEAD", b"PUT", b"PATCH", b"DELETE", b"OPTIONS")


def is_external(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    )


def client_hello_sni(payload: bytes) -> str | None:
    """The server name in a TLS ClientHello, even one cut off after it."""
    if len(payload) < 44 or payload[0] != 0x16 or payload[5] != 0x01:
        return None
    # Record header (5), handshake header (4), version (2), random (32).
    pos = 43
    try:
        pos += 1 + payload[pos]  # session id
        pos += 2 + struct.unpack_from(">H", payload, pos)[0]  # cipher suites
        pos += 1 + payload[pos]  # compression methods
        end = min(len(payload), pos + 2 + struct.unpack_from(">H", payload, pos)[0])
        pos += 2
        while pos + 4 <= end:
            kind, length = struct.unpack_from(">HH", payload, pos)
            pos += 4
            if kind == 0:  # server_name: list length, name type, name length
                size = struct.unpack_from(">H", payload, pos + 3)[0]
                return payload[pos + 5 : pos + 5 + size].decode("ascii", "replace")
            pos += length
    except (IndexError, struct.error):
        return None
    return None


def http_host(payload: bytes) -> str | None:
    if payload.split(b" ", 1)[0] not in HTTP_METHODS:
        return None
    for line in payload.split(b"\r\n")[1:]:
        if line.lower().startswith(b"host:"):
            return line[5:].strip().decode("ascii", "replace")
    return None


def dns_question(payload: bytes) -> str | None:
    labels, pos = [], 12
    while pos < len(payload) and payload[pos]:
        size = payload[pos]
        if size & 0xC0:
            return None
        labels.append(payload[pos + 1 : pos + 1 + size].decode("ascii", "replace"))
        pos += 1 + size
    return ".".join(labels) or None


def main() -> None:
    summary_path, stop_path = sys.argv[1:3]
    # Not bound to an interface: it receives from all of them.
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(ETH_P_ALL))
    sock.settimeout(1.0)

    connections: dict[tuple[str, int], set[int]] = collections.defaultdict(set)
    names: dict[tuple[str, int], str] = {}
    http_requests: collections.Counter[str] = collections.Counter()
    bytes_in: collections.Counter[str] = collections.Counter()
    dns: collections.Counter[str] = collections.Counter()
    frames: collections.Counter[str] = collections.Counter()
    with open(os.path.join(os.path.dirname(summary_path), "sniffer.ready"), "w"):
        pass
    while not os.path.exists(stop_path):
        try:
            frame, address = sock.recvfrom(65535)
        except socket.timeout:
            continue
        frames[address[0]] += 1
        if len(frame) < 34 or frame[12:14] != b"\x08\x00":
            continue
        l4 = 14 + (frame[14] & 0x0F) * 4
        protocol = frame[23]
        src = socket.inet_ntoa(frame[26:30])
        dst = socket.inet_ntoa(frame[30:34])
        if protocol == 6 and len(frame) >= l4 + 20:
            sport, dport = struct.unpack_from(">HH", frame, l4)
            flags = frame[l4 + 13]
            payload = frame[l4 + (frame[l4 + 12] >> 4) * 4 :]
            if is_external(dst):
                if flags & 0x12 == 0x02:
                    connections[(dst, dport)].add(sport)
                if host := client_hello_sni(payload):
                    names[(dst, dport)] = host
                elif host := http_host(payload):
                    names[(dst, dport)] = host
                    http_requests[host] += 1
            elif is_external(src):
                bytes_in[src] += len(frame)
        elif protocol == 17 and len(frame) >= l4 + 8:
            dport = struct.unpack_from(">H", frame, l4 + 2)[0]
            if dport == 53 and (question := dns_question(frame[l4 + 8 :])):
                dns[question] += 1

    hosts: dict[str, dict[str, int]] = {}
    named_ips: dict[str, str] = {}
    for (ip, port), sources in sorted(connections.items()):
        host = names.get((ip, port), f"{ip}:{port} (unnamed)")
        named_ips.setdefault(ip, host)
        entry = hosts.setdefault(host, {"connections": 0, "http_requests": 0})
        entry["connections"] += len(sources)
    for host, count in http_requests.items():
        hosts.setdefault(host, {"connections": 0, "http_requests": 0})
        hosts[host]["http_requests"] = count
    for ip, count in bytes_in.items():
        host = named_ips.get(ip, f"{ip} (no connection seen)")
        hosts.setdefault(host, {"connections": 0, "http_requests": 0})
        hosts[host]["bytes_in"] = hosts[host].get("bytes_in", 0) + count
    with open(summary_path, "w", encoding="utf-8") as out:
        json.dump(
            {"frames_seen": dict(frames), "hosts": hosts, "dns_questions": dict(dns)},
            out,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
