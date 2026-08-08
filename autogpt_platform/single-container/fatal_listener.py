#!/usr/bin/env python3
"""Stop Supervisor for fatal services and unexpected bootstrap exits."""

from __future__ import annotations

import os
import re
import signal
import sys
from collections.abc import Callable
from typing import TextIO

SAFE_PROCESS_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")
MAX_PAYLOAD_LENGTH = 64 * 1024


def main() -> int:
    while True:
        sys.stdout.write("READY\n")
        sys.stdout.flush()
        header = sys.stdin.readline()
        if not header:
            return 0
        handle_event(header, sys.stdin, sys.stdout, _terminate_supervisor)


def handle_event(
    header: str,
    input_stream: TextIO,
    output_stream: TextIO,
    terminate: Callable[[], None],
) -> None:
    fields = _parse_fields(header)
    try:
        length = int(fields["len"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Supervisor event has an invalid payload length") from exc
    if not 0 <= length <= MAX_PAYLOAD_LENGTH:
        raise RuntimeError("Supervisor event payload is too large")

    payload = input_stream.read(length)
    if len(payload) != length:
        raise RuntimeError("Supervisor event payload ended unexpectedly")
    event_name = fields.get("eventname")
    if event_name not in {"PROCESS_STATE_FATAL", "PROCESS_STATE_EXITED"}:
        raise RuntimeError("Supervisor event has an unsupported type")

    process_name = _safe_process_name(payload)
    payload_fields = _parse_fields(payload)
    if event_name == "PROCESS_STATE_EXITED" and not (
        process_name == "bootstrap" and payload_fields.get("expected") == "0"
    ):
        _acknowledge(output_stream)
        return

    reason = (
        "exited unexpectedly"
        if event_name == "PROCESS_STATE_EXITED"
        else "entered FATAL state"
    )
    print(
        f"[single-container] required process {reason}: {process_name}; "
        "terminating Supervisor for container restart",
        file=sys.stderr,
        flush=True,
    )
    _acknowledge(output_stream)
    terminate()


def _acknowledge(output_stream: TextIO) -> None:
    output_stream.write("RESULT 2\nOK")
    output_stream.flush()


def _parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in line.strip().split():
        name, separator, value = item.partition(":")
        if separator:
            fields[name] = value
    return fields


def _safe_process_name(payload: str) -> str:
    if any(character in payload for character in "\r\n\0"):
        return "unknown"
    fields = _parse_fields(payload)
    process_name = fields.get("processname", "unknown")
    return process_name if SAFE_PROCESS_NAME.fullmatch(process_name) else "unknown"


def _terminate_supervisor() -> None:
    os.kill(os.getppid(), signal.SIGTERM)
    while True:
        signal.pause()


if __name__ == "__main__":
    raise SystemExit(main())
