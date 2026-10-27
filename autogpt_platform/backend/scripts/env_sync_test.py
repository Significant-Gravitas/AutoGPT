import base64
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts import env_sync as sync

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    return state


def test_dry_run_reports_each_key_without_writing(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.env"
    source.write_text("MICROSOFT_CLIENT_ID='id'\nMICROSOFT_CLIENT_SECRET='mine'\n")
    target = tmp_path / ".env"
    target.write_text("MICROSOFT_CLIENT_SECRET='theirs'\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["env_sync", "--source", f"local:{source}", "--target", str(target)],
    )
    sync.main()
    assert capsys.readouterr().out.splitlines() == [
        "MICROSOFT_CLIENT_ID: missing at target",
        "MICROSOFT_CLIENT_SECRET: different",
        "MICROSOFT_TENANT_ID: missing at source",
        "Dry run; no files changed.",
    ]
    assert target.read_text() == "MICROSOFT_CLIENT_SECRET='theirs'\n"


def test_dry_run_refuses_a_target_apply_would_refuse(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.env"
    source.write_text("MICROSOFT_CLIENT_ID='id'\n")
    target = tmp_path / ".env.default"
    target.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        ["env_sync", "--source", f"local:{source}", "--target", str(target)],
    )
    with pytest.raises(SystemExit) as stopped:
        sync.main()
    assert stopped.value.code == 2
    assert "private, regular env file" in capsys.readouterr().err


def test_dry_run_accepts_a_target_in_a_new_directory(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.env"
    source.write_text("MICROSOFT_CLIENT_ID='id'\n")
    target = tmp_path / "new" / ".env"
    monkeypatch.setattr(
        sys,
        "argv",
        ["env_sync", "--source", f"local:{source}", "--target", str(target)],
    )
    sync.main()
    assert "MICROSOFT_CLIENT_ID: missing at target" in capsys.readouterr().out
    assert not target.parent.exists()


def test_value_written_after_the_report_still_needs_replace(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.env"
    source.write_text("MICROSOFT_CLIENT_SECRET='mine'\n")
    target = tmp_path / ".env"
    target.write_text("OTHER='stay'\n")
    apply_updates = sync.apply_updates

    def another_run_writes_first(*args, **kwargs):
        target.write_text("OTHER='stay'\nMICROSOFT_CLIENT_SECRET='theirs'\n")
        return apply_updates(*args, **kwargs)

    monkeypatch.setattr(sync, "apply_updates", another_run_writes_first)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_sync",
            *("--source", f"local:{source}", "--target", str(target)),
            *("--key", "MICROSOFT_CLIENT_SECRET", "--apply"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        sync.main()
    assert stopped.value.code == 2
    assert "--replace" in capsys.readouterr().err
    secret = dotenv_values(target, interpolate=False)["MICROSOFT_CLIENT_SECRET"]
    assert secret == "theirs"


@pytest.mark.parametrize(
    ("flags", "hosts"),
    [
        ([], ["desktop"]),
        (["--windows-host", "laptop"], ["laptop"]),
    ],
)
def test_windows_hosts_default_to_desktop_until_named(
    tmp_path, monkeypatch, flags, hosts
):
    seen = []

    def read_source(source, windows_hosts):
        seen.append(windows_hosts)
        return {}

    monkeypatch.setattr(sync, "read_source", read_source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_sync",
            *("--source", "desktop:/backend/.env", "--target", str(tmp_path / ".env")),
            *flags,
        ],
    )
    sync.main()
    assert seen == [hosts]


def test_a_failing_read_stops_without_printing_its_output(
    tmp_path, monkeypatch, capsys
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/bin/sh\necho \"MICROSOFT_CLIENT_SECRET='leaked-value'\"\n"
        "echo leaked-value >&2\nexit 1\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_sync",
            *("--source", "laptop:/backend/.env", "--target", str(tmp_path / ".env")),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        sync.main()
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.err == "Env synchronization stopped: CalledProcessError\n"
    assert "leaked-value" not in output.out + output.err


def test_non_ascii_values_survive_a_non_utf8_locale(tmp_path):
    source = tmp_path / "source.env"
    source.write_bytes("MICROSOFT_TENANT_ID='café'\n".encode())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!/bin/sh\ncat {shlex.quote(str(source))}\n")
    ssh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
    }
    read = (
        "import sys; from scripts.env_sync import read_source; "
        "print(ascii([read_source(source, []) for source in sys.argv[1:]]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", read, f"local:{source}", "laptop:/backend/.env"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ascii([{"MICROSOFT_TENANT_ID": "café"}] * 2)


@pytest.mark.parametrize("quote", ["'", "\u2018", "\u2019", "\u201a", "\u201b"])
def test_windows_paths_stay_one_powershell_string(quote):
    command = sync.windows_read_command(f"C:\\Users\\O{quote}Brien\\.env")
    script = base64.b64decode(command.split()[-1]).decode("utf-16le")
    literal = f"'C:\\Users\\O{quote * 2}Brien\\.env'"
    assert f"ReadAllText({literal})" in script


def test_conflicting_sources_do_not_expose_values():
    with pytest.raises(ValueError) as error:
        sync.select_values(
            [
                {"MICROSOFT_CLIENT_SECRET": "secret-one"},
                {"MICROSOFT_CLIENT_SECRET": "secret-two"},
            ],
            ["MICROSOFT_CLIENT_SECRET"],
        )
    assert "MICROSOFT_CLIENT_SECRET" in str(error.value)
    assert "secret-one" not in str(error.value)
    assert "secret-two" not in str(error.value)


def test_empty_source_values_are_not_updates():
    assert (
        sync.select_values(
            [{"MICROSOFT_CLIENT_SECRET": ""}], ["MICROSOFT_CLIENT_SECRET"]
        )
        == {}
    )
