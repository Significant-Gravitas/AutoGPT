#!/usr/bin/env python3
"""Fail when the frontend's Next.js server logged an error during the e2e run.

Next.js prints every error it catches on the server with a leading "⨯": a
render or route handler that threw, a module that failed to load, an image the
optimizer couldn't read. The browser tests can still pass while it happens (a
retried request, a broken image, a fallback page), so the full-stack legs read
the frontend container's log after the tests and fail on any such line.
"""

import argparse
import ipaddress
import re
import sys
from collections import Counter
from pathlib import Path

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# `docker compose logs` without --no-log-prefix: "frontend-1  | <line>".
_COMPOSE_PREFIX = re.compile(r"^[\w.-]+\s+\|\s?")
_ERROR_MARK = "⨯"
# What Node prints under an error: its frames, Next's digest, and the
# `[cause]:` of a wrapped error.
_STACK_LINE = re.compile(r"^\s+(at |digest:|\[cause\]:)")
_MAX_STACK_LINES = 8


def _clean(line: str) -> str:
    return _COMPOSE_PREFIX.sub("", _ANSI_ESCAPE.sub("", line.rstrip("\n")), count=1)


def server_errors(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Each error Next.js logged: its message and the stack lines under it."""
    errors: list[tuple[str, list[str]]] = []
    # Only the stack lines right under an error belong to it: any other line
    # (another component's log entry) ends that error's stack.
    collecting = False
    for raw in lines:
        line = _clean(raw)
        stripped = line.strip()
        if stripped.startswith(_ERROR_MARK):
            message = stripped.removeprefix(_ERROR_MARK).strip()
            errors.append((message or "(no message)", []))
            collecting = True
        elif collecting and _STACK_LINE.match(line):
            stack = errors[-1][1]
            if len(stack) < _MAX_STACK_LINES:
                stack.append(line.rstrip())
        else:
            collecting = False
    return errors


# The image optimizer timing out on a host outside the stack: the e2e seed's
# Marketplace listings point at production media on GCS, so a slow network
# between the runner and that host, not the platform, makes this one. It is
# reported, but doesn't fail the leg. A failed (rather than timed-out) fetch,
# or any local URL, still fails it.
_EXTERNAL_IMAGE_TIMEOUT = re.compile(
    r"^upstream image response timed out for https?://(?P<host>[^/:\s]+)"
)


def is_external_image_timeout(message: str) -> bool:
    match = _EXTERNAL_IMAGE_TIMEOUT.match(message)
    return bool(match) and not _is_local_host(match["host"].lower())


def _is_local_host(host: str) -> bool:
    """A host inside the stack or on the runner. Docker service names have no
    dot (nor does a bracketed IPv6 host as matched here); an IP literal is
    local if it's loopback, unspecified, link-local or private. A DNS name is
    local only if it's localhost or Docker's host alias, whatever its labels."""
    if "." not in host or host in ("localhost", "host.docker.internal"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_loopback
        or address.is_unspecified
        or address.is_link_local
        or address.is_private
    )


def report(errors: list[tuple[str, list[str]]], source: str) -> str:
    counts = Counter(message for message, _ in errors)
    summary = (
        f"{source}: the Next.js server logged {len(errors)} error(s), "
        f"{len(counts)} distinct. Fix the cause; the tests passing doesn't make "
        "these harmless."
    )
    out = [summary]
    first_stack: dict[str, list[str]] = {}
    for message, stack in errors:
        first_stack.setdefault(message, stack)
    for message, count in counts.most_common():
        title = f"Next.js server error ({count}x)"
        out.append(f"::error title={title}::{message}")
        out.extend(f"    {line.strip()}" for line in first_stack[message])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="the frontend container's log")
    args = parser.parse_args(argv)
    # Messages name the log by its file name only.
    name = args.log.name
    try:
        text = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"{name}: could not read the log: {exc.strerror or type(exc).__name__}")
        return 1
    lines = text.splitlines()
    # The server always logs its startup, so an empty log means the capture
    # broke (a renamed service, a recreated container), not a clean run.
    if not any(line.strip() for line in lines):
        print(f"{name}: the log is empty, so the server's errors can't be checked")
        return 1
    errors = server_errors(lines)
    external = Counter(
        message for message, _ in errors if is_external_image_timeout(message)
    )
    for message, count in external.most_common():
        print(f"::warning title=External image timed out ({count}x)::{message}")
    errors = [error for error in errors if not is_external_image_timeout(error[0])]
    if errors:
        print(report(errors, name))
        return 1
    print(f"{name}: no Next.js server errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
