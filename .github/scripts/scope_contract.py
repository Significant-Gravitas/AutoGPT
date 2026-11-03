import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def context():
    return {
        "environment": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in os.environ.items()
            if not key.startswith("SUDO_") and key not in {"_", "SHLVL", "INVOCATION_ID"}
        },
        "uid": os.getuid(),
        "gid": os.getgid(),
        "groups": sorted(os.getgroups()),
        "cwd": os.getcwd(),
    }


if sys.argv[1] == "save":
    Path(sys.argv[2]).write_text(json.dumps(context()))
elif sys.argv[1] == "verify":
    expected = json.loads(Path(sys.argv[2]).read_text())
    actual = context()
    mismatched = [key for key, value in expected["environment"].items()
                  if actual["environment"].get(key) != value]
    if mismatched:
        raise SystemExit("Environment mismatch in variable names: " + ", ".join(mismatched))
    for key in ("uid", "gid", "groups", "cwd"):
        assert actual[key] == expected[key], key
    print("PASS: complete caller environment, uid, gid, groups and cwd preserved")
elif sys.argv[1] == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([
        sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)"
    ], start_new_session=True)
    Path(sys.argv[2]).write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(300)
elif sys.argv[1] == "gone":
    for pid in json.loads(Path(sys.argv[2]).read_text()):
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z "), f"PID {pid} still running"
    print("PASS: parent and detached SIGTERM-ignoring child stopped")
else:
    raise SystemExit("Unknown mode")
