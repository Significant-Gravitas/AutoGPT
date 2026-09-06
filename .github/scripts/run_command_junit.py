#!/usr/bin/env python3

import argparse
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--classname", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def write_junit(
    path: Path,
    name: str,
    classname: str,
    command: list[str],
    returncode: int,
    duration: float,
) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": classname,
            "tests": "1",
            "failures": "1" if returncode else "0",
            "errors": "0",
            "skipped": "0",
            "time": f"{duration:.6f}",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": classname, "name": name, "time": f"{duration:.6f}"},
    )
    if returncode:
        failure = ET.SubElement(
            case,
            "failure",
            {"message": f"command exited with status {returncode}"},
        )
        failure.text = " ".join(command)

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    completed = subprocess.run(args.command, check=False)
    duration = time.monotonic() - started_at
    write_junit(
        args.output,
        args.name,
        args.classname,
        args.command,
        completed.returncode,
        duration,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
