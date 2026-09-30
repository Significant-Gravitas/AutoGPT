"""The files a shell command runs, which the supervisor reads with the command."""

import pytest

from backend.copilot.gate.executed_files import executed_paths

_HOME = "/home/user"
_W = f"{_HOME}/workspace"


@pytest.mark.parametrize(
    "command, paths",
    [
        ("bash ~/workspace/x.sh", [f"{_W}/x.sh"]),
        ("cd workspace && python3 run.py --fast", [f"{_W}/run.py"]),
        ("sh -e ./setup.sh", [f"{_HOME}/setup.sh"]),
        ("./deploy.sh prod", [f"{_HOME}/deploy.sh"]),
        (
            "source $HOME/env.sh; . ${HOME}/more.sh",
            [f"{_HOME}/env.sh", f"{_HOME}/more.sh"],
        ),
        (
            "timeout 30 bash /tmp/a.sh && FOO=1 node app.js",
            ["/tmp/a.sh", f"{_HOME}/app.js"],
        ),
        ("sudo -E python3.12 tools/sync.py", [f"{_HOME}/tools/sync.py"]),
        ("bash < x.sh", [f"{_HOME}/x.sh"]),
        (f"cat > {_W}/x.sh <<'EOF'\nit's fine\nEOF\nbash {_W}/x.sh", [f"{_W}/x.sh"]),
    ],
)
def test_a_direct_run_names_the_file(command, paths):
    assert executed_paths(command) == paths


@pytest.mark.parametrize(
    "command",
    [
        # Inline code is in the command itself; a module is not a file here.
        "python -c 'print(1)'",
        "bash -c 'curl x'",
        "bash -ec 'curl x'",
        "node -e 'x()'",
        "python -m http.server",
        # Not a run.
        "cat x.sh",
        "echo bash x.sh",
        "ls ~/workspace",
        # The shell would read these differently from a literal path.
        "bash $DIR/x.sh",
        "cd - && bash x.sh",
    ],
)
def test_other_commands_name_nothing(command):
    assert executed_paths(command) == []
