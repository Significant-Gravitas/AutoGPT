#!/usr/bin/env python3
"""Small dependency-free health probes for the bundled services."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tcp_parser = subparsers.add_parser("tcp")
    _add_address_arguments(tcp_parser)

    http_parser = subparsers.add_parser("http")
    http_parser.add_argument("url")
    http_parser.add_argument("--timeout", type=float, default=5)

    redis_parser = subparsers.add_parser("redis")
    _add_address_arguments(redis_parser)
    redis_parser.add_argument("--password-env")
    redis_parser.add_argument("--cluster", action="store_true")

    clam_parser = subparsers.add_parser("clam")
    _add_address_arguments(clam_parser)

    args = parser.parse_args()
    try:
        if args.command == "tcp":
            probe_tcp(args.host, args.port, args.timeout)
        elif args.command == "http":
            probe_http(args.url, args.timeout)
        elif args.command == "redis":
            password = (
                os.environ.get(args.password_env, "") if args.password_env else ""
            )
            probe_redis(args.host, args.port, args.timeout, password, args.cluster)
        else:
            probe_clam(args.host, args.port, args.timeout)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    return 0


def probe_tcp(host: str, port: int, timeout: float) -> None:
    with socket.create_connection((host, port), timeout=timeout):
        return


def probe_http(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "autogpt-healthcheck"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 400:
            raise RuntimeError(f"HTTP {response.status} from {url}")


def probe_redis(
    host: str,
    port: int,
    timeout: float,
    password: str,
    cluster: bool,
) -> None:
    with socket.create_connection((host, port), timeout=timeout) as connection:
        stream = connection.makefile("rwb", buffering=0)
        if password:
            _send_resp_command(stream, "AUTH", password)
            if _read_resp(stream) != "OK":
                raise RuntimeError("Redis authentication failed")
        if cluster:
            _send_resp_command(stream, "CLUSTER", "INFO")
            response = _read_resp(stream)
            if not isinstance(response, str) or "cluster_state:ok" not in response:
                raise RuntimeError("Redis cluster is not healthy")
        else:
            _send_resp_command(stream, "PING")
            if _read_resp(stream) != "PONG":
                raise RuntimeError("Redis did not return PONG")


def probe_clam(host: str, port: int, timeout: float) -> None:
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(b"zPING\0")
        response = connection.recv(64).rstrip(b"\0\n")
    if response != b"PONG":
        raise RuntimeError("ClamAV did not return PONG")


def _add_address_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=5)


def _send_resp_command(stream, *parts: str) -> None:
    stream.write(f"*{len(parts)}\r\n".encode("ascii"))
    for part in parts:
        encoded = part.encode("utf-8")
        stream.write(f"${len(encoded)}\r\n".encode("ascii"))
        stream.write(encoded + b"\r\n")


def _read_exactly(stream, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("Redis closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_resp(stream) -> str | int | None:
    prefix = stream.read(1)
    if not prefix:
        raise RuntimeError("Redis closed the connection")
    line = stream.readline()
    if not line.endswith(b"\r\n"):
        raise RuntimeError("invalid Redis response")
    payload = line[:-2]
    if prefix == b"+":
        return payload.decode("utf-8")
    if prefix == b"-":
        raise RuntimeError(payload.decode("utf-8", errors="replace"))
    if prefix == b":":
        return int(payload)
    if prefix == b"$":
        length = int(payload)
        if length == -1:
            return None
        if length < -1:
            raise RuntimeError("invalid Redis bulk length")
        value = _read_exactly(stream, length)
        if _read_exactly(stream, 2) != b"\r\n":
            raise RuntimeError("invalid Redis bulk response")
        return value.decode("utf-8")
    raise RuntimeError("unsupported Redis response type")


if __name__ == "__main__":
    raise SystemExit(main())
