import os
import subprocess
import sys
from pathlib import Path

from graphiti_core.telemetry.telemetry import TELEMETRY_ENV_VAR

import backend
from backend.util.vendor_telemetry import VENDOR_TELEMETRY_OFF, disable_vendor_telemetry

BACKEND_ROOT = Path(backend.__file__).resolve().parents[1]


def run_in_fresh_process(code: str, tmp_path: Path, **env: str) -> None:
    """Runs `code` in a new interpreter whose environment has neither switch
    unless passed in `env`, so the result doesn't depend on this process."""
    environ = {k: v for k, v in os.environ.items() if k not in VENDOR_TELEMETRY_OFF}
    # mem0 writes its anonymous id under MEM0_DIR on import.
    environ["MEM0_DIR"] = str(tmp_path)
    environ.update(env)
    subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_ROOT, env=environ, check=True
    )


def test_turns_vendor_telemetry_off_when_unset():
    environ: dict[str, str] = {}

    disable_vendor_telemetry(environ)

    assert environ == {
        "GRAPHITI_TELEMETRY_ENABLED": "false",
        "MEM0_TELEMETRY": "false",
    }


def test_keeps_a_value_the_deployment_set():
    environ = {"MEM0_TELEMETRY": "true"}

    disable_vendor_telemetry(environ)

    assert environ["MEM0_TELEMETRY"] == "true"
    assert environ["GRAPHITI_TELEMETRY_ENABLED"] == "false"


def test_graphiti_reads_the_variable_we_set():
    assert TELEMETRY_ENV_VAR in VENDOR_TELEMETRY_OFF


def test_importing_backend_turns_mem0_telemetry_off(tmp_path: Path):
    run_in_fresh_process(
        "import os\n"
        "import backend\n"
        "from mem0.memory import telemetry\n"
        "assert os.environ['GRAPHITI_TELEMETRY_ENABLED'] == 'false'\n"
        "assert os.environ['MEM0_TELEMETRY'] == 'false'\n"
        "assert telemetry.MEM0_TELEMETRY is False\n"
        "assert telemetry.client_telemetry.posthog is None\n",
        tmp_path,
    )


def test_mem0_telemetry_stays_on_when_the_environment_opts_in(tmp_path: Path):
    run_in_fresh_process(
        "import backend\n"
        "from mem0.memory import telemetry\n"
        "assert telemetry.MEM0_TELEMETRY is True\n",
        tmp_path,
        MEM0_TELEMETRY="true",
    )
