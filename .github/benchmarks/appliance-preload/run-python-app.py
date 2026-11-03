import importlib.metadata
import shutil
import subprocess
import sys
import time

command, *arguments = sys.argv[1:]
executable = shutil.which(command)
if executable is None:
    raise RuntimeError(f"Console command is missing: {command}")
entries = [
    entry
    for entry in importlib.metadata.distribution(
        "autogpt-platform-backend"
    ).entry_points
    if entry.group == "console_scripts" and entry.name == command
]
if len(entries) != 1:
    raise RuntimeError(f"Expected exactly one project entry point for {command}")
started = time.monotonic()
main = entries[0].load()
print(
    f"[preload diagnostic] {command} imported in {time.monotonic() - started:.3f}s",
    flush=True,
)
subprocess.run(
    [
        "bash",
        "-euo",
        "pipefail",
        "-c",
        'source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"; wait_for_ready_file',
    ],
    check=True,
)
print(f"[preload diagnostic] {command} starting after bootstrap readiness", flush=True)
sys.argv = [executable, *arguments]
raise SystemExit(main())
