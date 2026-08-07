#!/usr/bin/env python3
"""Create and validate the persistent secrets used by the all-in-one image."""

from __future__ import annotations

import argparse
import base64
import binascii
import ipaddress
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

CONFIG_VERSION = "1"
SAFE_SECRET = re.compile(r"^[A-Za-z0-9._~-]+={0,2}$")
SAFE_USERNAME = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--path", type=Path, required=True)

    url_parser = subparsers.add_parser("validate-public-url")
    url_parser.add_argument("url")

    args = parser.parse_args()
    try:
        if args.command == "ensure":
            ensure_runtime_config(args.path, os.environ)
        else:
            print(validate_public_url(args.url))
    except (OSError, ValueError) as exc:
        print(f"runtime configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


def ensure_runtime_config(path: Path, environment: os._Environ[str]) -> dict[str, str]:
    """Return the existing secrets or atomically create them on first boot."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink at {path}")
    if path.exists():
        values = _read_config(path)
        _verify_environment_matches(values, environment)
        path.chmod(0o600)
        return values

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    values = _new_values(environment)
    _write_config(path, values)
    return values


def validate_public_url(value: str) -> str:
    """Validate and normalize the externally reachable origin."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("AUTOGPT_PUBLIC_URL must use http or https")
    if not parsed.hostname:
        raise ValueError("AUTOGPT_PUBLIC_URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("AUTOGPT_PUBLIC_URL must not include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "AUTOGPT_PUBLIC_URL must be an origin without path, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("AUTOGPT_PUBLIC_URL has an invalid port") from exc

    host = _normalize_public_host(parsed.hostname)
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _normalize_public_host(host: str) -> str:
    """Return a config-safe IP literal or IDNA DNS name."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            ascii_host = host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("AUTOGPT_PUBLIC_URL has an invalid host") from exc
        labels = ascii_host.split(".")
        if (
            not ascii_host
            or len(ascii_host) > 253
            or any(not SAFE_DNS_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError("AUTOGPT_PUBLIC_URL has an invalid host")
        return ascii_host
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _new_values(environment: os._Environ[str]) -> dict[str, str]:
    vapid_private, vapid_public = _configured_or_generated_vapid(environment)
    values = {
        "AUTOGPT_RUNTIME_CONFIG_VERSION": CONFIG_VERSION,
        "POSTGRES_PASSWORD": _configured_or_generated(
            environment, "POSTGRES_PASSWORD", lambda: secrets.token_urlsafe(36)
        ),
        "RABBITMQ_DEFAULT_USER": environment.get("RABBITMQ_DEFAULT_USER") or "autogpt",
        "RABBITMQ_DEFAULT_PASS": _configured_or_generated(
            environment, "RABBITMQ_DEFAULT_PASS", lambda: secrets.token_urlsafe(36)
        ),
        "BETTER_AUTH_SECRET": _configured_or_generated(
            environment, "BETTER_AUTH_SECRET", lambda: secrets.token_urlsafe(48)
        ),
        "ENCRYPTION_KEY": _configured_or_generated(
            environment, "ENCRYPTION_KEY", _fernet_key
        ),
        "UNSUBSCRIBE_SECRET_KEY": _configured_or_generated(
            environment, "UNSUBSCRIBE_SECRET_KEY", lambda: secrets.token_urlsafe(36)
        ),
        "GRAPHITI_FALKORDB_PASSWORD": _configured_or_generated(
            environment,
            "GRAPHITI_FALKORDB_PASSWORD",
            lambda: secrets.token_urlsafe(36),
        ),
        "VAPID_PRIVATE_KEY": vapid_private,
        "VAPID_PUBLIC_KEY": vapid_public,
    }
    _validate_values(values)
    return values


def _configured_or_generated(
    environment: os._Environ[str], name: str, generator
) -> str:
    return environment.get(name) or generator()


def _configured_or_generated_vapid(
    environment: os._Environ[str],
) -> tuple[str, str]:
    private = environment.get("VAPID_PRIVATE_KEY")
    public = environment.get("VAPID_PUBLIC_KEY")
    if bool(private) != bool(public):
        raise ValueError("VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY must be set together")
    return (private, public) if private and public else _generate_vapid_keypair()


def _generate_vapid_keypair() -> tuple[str, str]:
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as exc:
        raise ValueError("cryptography is required to generate VAPID keys") from exc

    key = ec.generate_private_key(ec.SECP256R1())
    numbers = key.private_numbers()
    private = numbers.private_value.to_bytes(32, "big")
    public_numbers = numbers.public_numbers
    public = (
        b"\x04"
        + public_numbers.x.to_bytes(32, "big")
        + public_numbers.y.to_bytes(32, "big")
    )
    return _base64url(private), _base64url(public)


def _fernet_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _validate_values(values: dict[str, str]) -> None:
    expected = {
        "AUTOGPT_RUNTIME_CONFIG_VERSION",
        "POSTGRES_PASSWORD",
        "RABBITMQ_DEFAULT_USER",
        "RABBITMQ_DEFAULT_PASS",
        "BETTER_AUTH_SECRET",
        "ENCRYPTION_KEY",
        "UNSUBSCRIBE_SECRET_KEY",
        "GRAPHITI_FALKORDB_PASSWORD",
        "VAPID_PRIVATE_KEY",
        "VAPID_PUBLIC_KEY",
    }
    if set(values) != expected:
        raise ValueError("runtime configuration has missing or unknown keys")
    if values["AUTOGPT_RUNTIME_CONFIG_VERSION"] != CONFIG_VERSION:
        raise ValueError("unsupported runtime configuration version")
    if not SAFE_USERNAME.fullmatch(values["RABBITMQ_DEFAULT_USER"]):
        raise ValueError("RABBITMQ_DEFAULT_USER contains unsupported characters")
    if values["RABBITMQ_DEFAULT_USER"] == "guest":
        raise ValueError("RABBITMQ_DEFAULT_USER may not use the built-in guest account")

    for name in expected - {"AUTOGPT_RUNTIME_CONFIG_VERSION", "RABBITMQ_DEFAULT_USER"}:
        value = values[name]
        if len(value) < 32 or not SAFE_SECRET.fullmatch(value):
            raise ValueError(f"{name} must be at least 32 URL-safe characters")

    try:
        decoded_key = base64.urlsafe_b64decode(values["ENCRYPTION_KEY"])
    except (ValueError, binascii.Error) as exc:
        raise ValueError("ENCRYPTION_KEY must be a Fernet-compatible key") from exc
    if len(decoded_key) != 32:
        raise ValueError("ENCRYPTION_KEY must decode to exactly 32 bytes")

    vapid_private = _decode_base64url(values["VAPID_PRIVATE_KEY"])
    vapid_public = _decode_base64url(values["VAPID_PUBLIC_KEY"])
    if len(vapid_private) != 32:
        raise ValueError("VAPID_PRIVATE_KEY must decode to exactly 32 bytes")
    if len(vapid_public) != 65 or vapid_public[0] != 4:
        raise ValueError("VAPID_PUBLIC_KEY must be an uncompressed P-256 public key")


def _decode_base64url(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid URL-safe base64 value") from exc


def _read_config(path: Path) -> dict[str, str]:
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"runtime configuration is not a regular file: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="ascii").splitlines(), 1
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        name, separator, value = raw_line.partition("=")
        if not separator or not name or name in values:
            raise ValueError(f"invalid runtime configuration line {line_number}")
        values[name] = value
    _validate_values(values)
    return values


def _verify_environment_matches(
    values: dict[str, str], environment: os._Environ[str]
) -> None:
    for name, persisted in values.items():
        configured = environment.get(name)
        if configured and configured != persisted:
            raise ValueError(
                f"{name} differs from the value persisted on first boot; "
                "restore the original value or start with a new data volume"
            )


def _write_config(path: Path, values: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="ascii")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("# Generated once by the AutoGPT all-in-one image.\n")
            for name, value in values.items():
                stream.write(f"{name}={value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
