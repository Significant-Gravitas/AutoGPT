#!/usr/bin/env python3
"""Small health probes for the bundled services."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tcp_parser = subparsers.add_parser("tcp")
    _add_address_arguments(tcp_parser)

    http_parser = subparsers.add_parser("http")
    http_parser.add_argument("urls", nargs="+")
    http_parser.add_argument("--timeout", type=float, default=5)

    amqp_parser = subparsers.add_parser("amqp")
    _add_address_arguments(amqp_parser)
    amqp_parser.add_argument("--username-env", required=True)
    amqp_parser.add_argument("--password-env", required=True)

    redis_parser = subparsers.add_parser("redis")
    _add_address_arguments(redis_parser)
    redis_parser.add_argument("--password-env")
    redis_parser.add_argument("--cluster", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "tcp":
            probe_tcp(args.host, args.port, args.timeout)
        elif args.command == "http":
            probe_http_many(args.urls, args.timeout)
        elif args.command == "amqp":
            probe_amqp(
                args.host,
                args.port,
                args.timeout,
                os.environ.get(args.username_env, ""),
                os.environ.get(args.password_env, ""),
            )
        else:
            password = (
                os.environ.get(args.password_env, "") if args.password_env else ""
            )
            probe_redis(args.host, args.port, args.timeout, password, args.cluster)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    return 0


def probe_tcp(host: str, port: int, timeout: float) -> None:
    with socket.create_connection((host, port), timeout=timeout):
        pass


def probe_http(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "autogpt-healthcheck"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status} from {url}")


def probe_http_many(urls: list[str], timeout: float) -> None:
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = [executor.submit(probe_http, url, timeout) for url in urls]
        for future in futures:
            future.result()


def probe_amqp(
    host: str,
    port: int,
    timeout: float,
    username: str,
    password: str,
) -> None:
    if not username or not password:
        raise RuntimeError("AMQP credentials are missing")
    try:
        import pika

        parameters = pika.ConnectionParameters(
            host=host,
            port=port,
            virtual_host="/",
            credentials=pika.PlainCredentials(username, password),
            connection_attempts=1,
            retry_delay=0,
            socket_timeout=timeout,
            stack_timeout=timeout,
            blocked_connection_timeout=timeout,
            heartbeat=0,
        )
        connection = pika.BlockingConnection(parameters)
        connection.close()
    except Exception as exc:
        raise RuntimeError("AMQP connection failed") from exc


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


def _add_address_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=5)


def _send_resp_command(stream: BinaryIO, *parts: str) -> None:
    stream.write(f"*{len(parts)}\r\n".encode("ascii"))
    for part in parts:
        encoded = part.encode("utf-8")
        stream.write(f"${len(encoded)}\r\n".encode("ascii"))
        stream.write(encoded + b"\r\n")


def _read_exactly(stream: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("Redis closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_resp(stream: BinaryIO) -> str | int | None:
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
