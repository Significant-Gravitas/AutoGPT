import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

assert os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "linux"
root = Path.cwd()
backend = root / "autogpt_platform/backend"
output = Path(os.environ["RUNNER_TEMP"]) / "installer-comparison"
output.mkdir()
records = []
base = dict(os.environ)
base.pop("VIRTUAL_ENV", None)
version = subprocess.check_output(
    [sys.executable, str(root / ".github/workflows/scripts/get_package_version_from_lockfile.py"), "poetry"],
    cwd=backend,
    text=True,
).strip()
base["POETRY_VERSION"] = version
candidate = {
    **base,
    "UV_CACHE_DIR": str(output / "uv-cache"),
    "UV_TOOL_DIR": str(output / "uv-tools"),
    "UV_TOOL_BIN_DIR": str(output / "uv-bin"),
}
baseline = {
    **base,
    "POETRY_HOME": str(output / "poetry-tool"),
    "POETRY_CACHE_DIR": str(output / "poetry-cache"),
    "VIRTUAL_ENV": str(output / "baseline-venv"),
}


def run(name, command, env):
    start = time.perf_counter()
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=backend, env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append({"name": name, "seconds": round(time.perf_counter() - start, 3), "exit_code": result.returncode})
    print(json.dumps(records[-1]), flush=True)
    if result.returncode:
        print((output / f"{name}.log").read_text()[-8000:])
        raise RuntimeError(f"{name} failed")


try:
    for name in os.environ["INSTALL_ORDER"].split(","):
        if name == "baseline":
            start = time.perf_counter()
            installer = output / "install-poetry.py"
            with urllib.request.urlopen("https://install.python-poetry.org", timeout=30) as response:
                installer.write_bytes(response.read())
            run("baseline-poetry-tool", [sys.executable, str(installer)], baseline)
            run("baseline-venv", [sys.executable, "-m", "venv", baseline["VIRTUAL_ENV"]], baseline)
            run("baseline-install", [str(Path(baseline["POETRY_HOME"]) / "bin/poetry"), "install", "--no-interaction"], baseline)
            records.append({"name": "baseline-total", "seconds": round(time.perf_counter() - start, 3), "exit_code": 0})
        else:
            assert name == "candidate"
            run("candidate-total", ["bash", str(root / ".github/scripts/install-backend-uv-candidate.sh")], candidate)
    for name, python in [("baseline", Path(baseline["VIRTUAL_ENV"]) / "bin/python"), ("candidate", backend / ".venv/bin/python")]:
        inventory = subprocess.check_output([str(python), str(root / ".github/scripts/benchmark_backend_tools.py"), "inventory"], cwd=backend)
        (output / f"{name}-inventory.json").write_bytes(inventory)
        run(f"{name}-dependency-check", ["uv", "pip", "check", "--python", str(python)], base)
    run("inventory-parity", [sys.executable, str(root / ".github/scripts/benchmark_backend_tools.py"), "compare", "--baseline", str(output / "baseline-inventory.json"), "--candidate", str(output / "candidate-inventory.json")], base)
    print(json.dumps(records, indent=2))
finally:
    (output / "measurements.json").write_text(json.dumps({"sha": os.environ["GITHUB_SHA"], "python": sys.version, "order": os.environ["INSTALL_ORDER"], "measurements": records}, indent=2))
