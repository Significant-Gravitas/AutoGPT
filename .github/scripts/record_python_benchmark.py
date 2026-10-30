import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
from pathlib import Path

assert platform.python_version() == os.environ["EXPECTED_PYTHON"]
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "cpuinfo": Path("/proc/cpuinfo").read_text(),
    "runner_image": os.environ.get("ImageVersion"),
    "runner_os": os.environ.get("ImageOS"),
    "head": os.environ.get("GITHUB_SHA"),
    "run_id": os.environ.get("GITHUB_RUN_ID"),
    "shard": os.environ.get("TEST_SHARD"),
    "config_args": sysconfig.get_config_var("CONFIG_ARGS"),
    "lock_sha256": hashlib.sha256(Path("poetry.lock").read_bytes()).hexdigest(),
    "packages": {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    },
}
Path("benchmark-environment.json").write_text(json.dumps(environment, indent=2))
print(
    json.dumps(
        {
            key: value
            for key, value in environment.items()
            if key not in {"packages", "cpuinfo"}
        },
        indent=2,
    )
)
