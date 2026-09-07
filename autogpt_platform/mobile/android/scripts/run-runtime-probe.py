#!/usr/bin/env python3
"""Run the disposable-device Android probe and fail closed on instrumentation errors."""

import argparse
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--adb", default="adb")
parser.add_argument("--serial", required=True)
parser.add_argument("--fixture-origin")
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--disposable",
    action="store_true",
    required=True,
    help="Confirm this is a disposable test device: app browsing data is cleared",
)
args = parser.parse_args()
command = [
    args.adb,
    "-s",
    args.serial,
    "shell",
    "am",
    "instrument",
    "-w",
    "-r",
    "-e",
    "disposable",
    "true",
]
if args.fixture_origin:
    command += ["-e", "fixtureOrigin", args.fixture_origin]
command += ["com.agpt.mobile.test/com.agpt.mobile.RuntimeProbe"]
try:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
except (OSError, subprocess.TimeoutExpired) as error:
    print(f"Runtime probe could not complete: {error}", file=sys.stderr)
    sys.exit(1)
output = completed.stdout
print(output, end="")
if args.output:
    args.output.write_text(output)
expected_suite = "fixture" if args.fixture_origin else "platform"
expected_count = 4 if args.fixture_origin else 3
required = [
    r"^INSTRUMENTATION_CODE: -1$",
    r"^INSTRUMENTATION_RESULT: probe_status=PASS$",
    rf"^INSTRUMENTATION_RESULT: suite={expected_suite}$",
    rf"^INSTRUMENTATION_RESULT: checks_passed={expected_count}$",
]
if completed.returncode != 0 or not all(
    re.search(pattern, output, re.MULTILINE) for pattern in required
):
    print(
        "Runtime probe failed: missing successful instrumentation completion.",
        file=sys.stderr,
    )
    sys.exit(1)
