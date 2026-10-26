"""The smoke test's own checks on the bundle and on what it leaves behind.

They only bite on a real stack; these make sure they would: that a new
directory, a leftover of an earlier run and a refused write are each seen,
and that the temporary data directory really goes.
"""

import importlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP / "build"))
smoke_test = importlib.import_module("smoke_test")


def test_lifecycle_smoke_configures_local_catalog_before_any_child(
    tmp_path, monkeypatch
):
    from autogpt_desktop import settings
    from autogpt_desktop.layout import DataDir

    data = tmp_path / "smoke data"
    called = []

    def contract(runtime):
        user = settings.read_user_settings(DataDir(data))
        assert user.items() >= smoke_test.NO_WAY_OUT.items()
        assert user["AUTOGPT_DESKTOP_PROFILE"] == "compact"
        catalog = Path(user["SKILLS_CATALOG_PATH"])
        assert catalog.is_absolute() and catalog.is_relative_to(data)
        release = json.loads((catalog / "release.json").read_text())
        assert [p["slug"] for p in release["packages"]] == ["smoke-skill"]
        assert release["experts"] == []
        assert (
            release["catalog_sha256"]
            == "09626479caa3d7fc5ae84bb5eb9fe153b7f7c079f532841dd0c9cf4bad173cf6"
        )
        assert release["packages"][0]["files"] == [
            {
                "path": "SKILL.md",
                "executable": False,
                "sha256": "aaa16bbeaf15f80f2ac2539752df87432010fd75e42b20bda7550755f0aee498",
            }
        ]
        assert release["packages"][0]["tree_sha256"] == (
            "5dcb29319a372ab8b0db67fc4f88c86bb9679e0e47ffac63ff367d398fbf0e78"
        )
        called.append("contract")
        return []

    monkeypatch.setattr(smoke_test, "backend_contract", contract)
    monkeypatch.setattr(smoke_test, "snapshot", lambda _: {})
    monkeypatch.setattr(smoke_test, "left_by_earlier_runs", lambda _: [])
    monkeypatch.setattr(
        smoke_test, "run", lambda *args: called.append(args[3].__name__) or []
    )
    args = SimpleNamespace(
        profile="compact", quick=False, timeout=600, restart_budget=120, read_only=False
    )
    assert smoke_test.smoke(tmp_path / "bundle", data, args) == 0
    assert called == [
        "contract",
        "fresh_install",
        "restart",
        "restart_with_a_password_reset",
        "restart_after_runs_were_stopped",
    ]


def test_lifecycle_smoke_requires_catalog_publication(tmp_path, monkeypatch):
    install = smoke_test.Install(False)
    install.runtime = SimpleNamespace(children=list)
    monkeypatch.setattr(smoke_test, "wait_for", lambda *args: False)
    report = smoke_test.Report()
    smoke_test.skills_catalog(tmp_path, install, report)
    assert any("was published" in failure for failure in report.failures)


def test_lifecycle_smoke_overlay_survives_service_environment_filtering(
    tmp_path, monkeypatch
):
    from autogpt_desktop import apps, ports, settings
    from autogpt_desktop.layout import Bundle, DataDir
    from autogpt_desktop.process import base_env

    data, bundle = DataDir(tmp_path / "smoke data"), Bundle(tmp_path / "bundle")
    smoke_test.configure_smoke(data.root)
    user = settings.read_user_settings(data)
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    secret = dict.fromkeys(
        (
            "POSTGRES_PASSWORD",
            "BETTER_AUTH_SECRET",
            "ENCRYPTION_KEY",
            "INTERNAL_API_KEY",
            "UNSUBSCRIBE_SECRET_KEY",
            "VAPID_PRIVATE_KEY",
            "VAPID_PUBLIC_KEY",
            "REDIS_PASSWORD",
            "RABBITMQ_DEFAULT_USER",
            "RABBITMQ_DEFAULT_PASS",
            "AUTOGPT_FRONTEND_DB_PASSWORD",
        ),
        "synthetic-smoke-only",
    )
    backend = settings.backend_environment(bundle, data, port, secret, user)
    monkeypatch.setattr(apps, "StopEvent", SimpleNamespace(create=lambda _: None))
    host = apps.host_process(bundle, data, backend, apps.MERGED[0], None)
    for env in (backend, host.env, {**base_env(), **backend}):
        assert env.items() >= smoke_test.NO_WAY_OUT.items()
        assert env["SKILLS_CATALOG_PATH"] == str(data.root / "smoke-catalog")
    frontend = settings.frontend_environment(backend, port, secret, data)
    assert not set(smoke_test.NO_WAY_OUT) & frontend.keys()


def test_lifecycle_smoke_contract_child_has_disposable_clean_environment(
    tmp_path, monkeypatch
):
    contamination = (
        "CHAT_USE_LOCAL",
        "OPENAI_API_KEY",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "AUTOGPT_DESKTOP_NODE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_PROFILE",
        "GITHUB_TOKEN",
    )
    for name in contamination:
        monkeypatch.setenv(name, "ambient-sentinel")
    seen = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        assert not set(contamination) & env.keys()
        assert argv[:2] == ["bundle-python", "-B"]
        assert kwargs["timeout"] == 180 and kwargs["stdin"] == subprocess.DEVNULL
        assert env.items() >= smoke_test.NO_WAY_OUT.items()
        home = Path(env["HOME"])
        assert home.is_dir() and str(home) != os.environ.get("USERPROFILE")
        assert Path(env["CODEX_HOME"]).is_relative_to(home)
        assert Path(env["CLAUDE_CONFIG_DIR"]).is_relative_to(home)
        seen.append(home)
        return subprocess.CompletedProcess(argv, 0, "1 of 1 dependencies hold\n", "")

    monkeypatch.setattr(
        smoke_test, "subprocess", SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL)
    )
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["bundle-python"])
    assert smoke_test.backend_contract(tmp_path) == []
    assert seen and not seen[0].exists()


class SmokeClock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


@pytest.fixture
def lifecycle_stop(tmp_path, monkeypatch):
    clock = SmokeClock()
    monkeypatch.setattr(smoke_test, "time", clock)
    install = smoke_test.Install(False)
    install.profile = "compact"
    install.stopping = smoke_test.Stopping(
        "graph/run", "workers", None, "workers", None, cancels=False
    )
    install.said = SimpleNamespace(events=[], finished=threading.Event())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/workers.log").write_text("Received cancel for run\n")
    (tmp_path / "logs/runtime.log").write_text("")
    monkeypatch.setattr(
        smoke_test,
        "RunWatcher",
        lambda *args: SimpleNamespace(last=lambda: ("TERMINATED", 0)),
    )
    monkeypatch.setattr(smoke_test, "persisted_run", lambda *args: ("TERMINATED", 0))
    monkeypatch.setattr(smoke_test, "check_stop_records", lambda *args: None)
    monkeypatch.setattr(smoke_test, "turn_in_progress", lambda *args: None)

    class Process:
        def __init__(self, phase, total, grace=120):
            self.stdin = io.StringIO()
            self.phase, self.total, self.grace = phase, total, grace
            self.killed = False
            self.waits = []
            self.phase_events = [
                {
                    "event": "shutdown",
                    "phase": "cancellation",
                    "seconds": phase,
                    "outcome": (
                        "cancelled"
                        if install.stopping.cancels
                        else "hurried" if install.stopping.hurried else "deadline"
                    ),
                }
            ]

        def wait(self, timeout):
            self.waits.append(timeout)
            if self.total - clock.seconds > timeout:
                raise subprocess.TimeoutExpired("fake-runtime", timeout)
            clock.seconds = self.total
            message = "Stopping 1 agent run"
            install.said.events.append(
                {"step": "stopping", "message": message, "grace_seconds": self.grace}
            )
            install.said.events.extend(self.phase_events)
            install.said.finished.set()
            return 0

        def kill(self):
            self.killed = True

    return install, clock, Process


@pytest.mark.parametrize(
    "hurried,phase,total",
    [(False, 20, 57), (True, 3, 40)],
)
def test_lifecycle_smoke_separates_cancel_wait_from_frozen_host_cleanup(
    tmp_path, lifecycle_stop, hurried, phase, total
):
    install, _, Process = lifecycle_stop
    install.stopping.hurried = hurried
    process = Process(phase, total)
    assert smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install) == []
    assert not process.killed
    assert install.ended["graph/run"][0] == "TERMINATED"
    assert process.waits[0] <= 120 - (3 if hurried else 0)


def test_lifecycle_smoke_rejects_missing_full_cancellation_wait(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    failures = smoke_test.stop_the_runs(Process(1, 38), DESKTOP, tmp_path, install)
    assert any("waited for" in failure for failure in failures)


def test_lifecycle_smoke_fourth_restart_checks_the_public_address(
    tmp_path, monkeypatch
):
    install = smoke_test.Install(False)
    install.address = "http://127.0.0.1:10000"
    monkeypatch.setattr(
        smoke_test,
        "Browser",
        lambda _: SimpleNamespace(sign_in=lambda *args: (200, {})),
    )
    for name in (
        "ended_runs_stay_ended",
        "stopped_turns_are_not_run_again",
        "run_agent",
        "counted",
        "stop_with_a_run_in_a_long_step",
    ):
        monkeypatch.setattr(smoke_test, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke_test, "executor_host", lambda _: "workers")
    monkeypatch.setattr(smoke_test, "wait_for", lambda *args: True)
    failures = smoke_test.restart_after_runs_were_stopped(
        "http://127.0.0.1:10001", tmp_path, install
    )
    assert any("address" in failure for failure in failures)


@pytest.mark.parametrize("profile", ["compact", "balanced", "isolated"])
def test_lifecycle_smoke_budget_follows_host_grace_and_parallel_tiers(
    profile, monkeypatch
):
    from autogpt_desktop import apps, supervisor

    install = smoke_test.Install(False)
    install.profile = profile
    assert smoke_test.CANCEL_WAIT_SECONDS == supervisor.CANCEL_WAIT_SECONDS
    assert smoke_test.stop_runs_budget(install) == 101
    assert smoke_test.stop_runs_budget(install) < supervisor.STOPPING_RUNS_GRACE_SECONDS
    original = apps.host_stop_timeout
    monkeypatch.setattr(
        apps,
        "host_stop_timeout",
        lambda group: 38 if "executor" in group.services else original(group),
    )
    assert smoke_test.stop_runs_budget(install) == 109


@pytest.mark.parametrize(
    "phase,cancels,hurried", [(20, False, True), (20, True, False)]
)
def test_lifecycle_smoke_rejects_slow_cancel_or_ignored_hurry(
    tmp_path, lifecycle_stop, phase, cancels, hurried
):
    install, _, Process = lifecycle_stop
    install.stopping.cancels, install.stopping.hurried = cancels, hurried
    failures = smoke_test.stop_the_runs(Process(phase, 57), DESKTOP, tmp_path, install)
    assert any("wait" in failure for failure in failures)


def test_lifecycle_smoke_cancellation_can_finish_before_slow_cleanup(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    install.stopping.cancels = True
    assert smoke_test.stop_the_runs(Process(2, 35), DESKTOP, tmp_path, install) == []


@pytest.mark.parametrize(
    "seconds", [True, None, "20", -1, float("nan"), float("inf"), 58]
)
def test_lifecycle_smoke_rejects_invalid_cancellation_measurements(
    tmp_path, lifecycle_stop, seconds
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events[0]["seconds"] = seconds
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("cancellation phase" in failure for failure in failures)


@pytest.mark.parametrize(
    "outcome", [None, "cancelled", "hurried", "inactive", "unknown"]
)
def test_lifecycle_smoke_requires_deadline_outcome_for_frozen_run(
    tmp_path, lifecycle_stop, outcome
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events[0]["outcome"] = outcome
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("waited for" in failure for failure in failures)


@pytest.mark.parametrize("count", [0, 2])
def test_lifecycle_smoke_requires_exactly_one_cancellation_event(
    tmp_path, lifecycle_stop, count
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events *= count
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("cancellation phase" in failure for failure in failures)


def test_lifecycle_smoke_drains_final_events_after_process_exit(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    deferred = process.phase_events
    process.phase_events = []
    waits = []

    def drained(timeout):
        waits.append(timeout)
        install.said.events.extend(deferred)
        return True

    install.said.finished = SimpleNamespace(set=lambda: None, wait=drained)
    assert smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install) == []
    assert waits and 0 < waits[0] <= 3


def test_lifecycle_smoke_refuses_incomplete_event_drain(tmp_path, lifecycle_stop):
    install, _, Process = lifecycle_stop
    install.said.finished = SimpleNamespace(set=lambda: None, wait=lambda _: False)
    failures = smoke_test.stop_the_runs(Process(20, 57), DESKTOP, tmp_path, install)
    assert any("shutdown events" in failure for failure in failures)


def test_lifecycle_smoke_event_reader_reports_eof_separately_from_ready():
    install = smoke_test.Install(False)
    event = {
        "event": "shutdown",
        "phase": "cancellation",
        "seconds": 20,
        "outcome": "deadline",
    }
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(event) + "\n"))
    said = smoke_test.Said(process, time.monotonic(), install)
    assert said.finished.wait(2)
    assert said.events == [event]


def test_lifecycle_smoke_still_requires_enough_advertised_grace(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    failures = smoke_test.stop_the_runs(
        Process(20, 57, grace=60), DESKTOP, tmp_path, install
    )
    assert any("shell was asked" in failure for failure in failures)


@pytest.mark.parametrize("hurried", [False, True])
def test_lifecycle_smoke_total_deadline_includes_the_hurry_delay(
    tmp_path, lifecycle_stop, hurried
):
    install, _, Process = lifecycle_stop
    install.stopping.hurried = hurried
    process = Process(3 if hurried else 20, 102)
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert process.killed and process.waits == [98 if hurried else 101]
    assert failures == [
        "the runtime did not stop within 101s of being asked to stop the runs"
    ]


@pytest.mark.parametrize("status,unfinished", [("RUNNING", 0), ("TERMINATED", 1)])
def test_lifecycle_smoke_keeps_the_persisted_termination_oracle(
    tmp_path, lifecycle_stop, monkeypatch, status, unfinished
):
    install, _, Process = lifecycle_stop
    monkeypatch.setattr(smoke_test, "persisted_run", lambda *args: (status, unfinished))
    failures = smoke_test.stop_the_runs(Process(20, 57), DESKTOP, tmp_path, install)
    assert any("persisted" in failure for failure in failures)


def test_lifecycle_smoke_keeps_leftover_failures(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=123, poll=lambda: 0)

    def popen(*args, **kwargs):
        return process

    monkeypatch.setattr(
        smoke_test, "subprocess", SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        smoke_test, "psutil", SimpleNamespace(Process=lambda _: object())
    )
    monkeypatch.setattr(smoke_test, "Family", lambda *args: object())
    monkeypatch.setattr(
        smoke_test,
        "Said",
        lambda *args: SimpleNamespace(wait_for_ready=lambda _: "url"),
    )
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["bundle-python"])
    monkeypatch.setattr(smoke_test, "stop", lambda _: [])
    monkeypatch.setattr(smoke_test, "stopped_gracefully", lambda *args: [])
    monkeypatch.setattr(
        smoke_test, "left_running", lambda _: ["owned process left running"]
    )
    failures = smoke_test.run(
        DESKTOP, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
    )
    assert failures == ["owned process left running"]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="permission bits are POSIX's"
)
posix_only_signals = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no hang-up signal"
)


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "frontend" / ".next" / "server").mkdir(parents=True)
    (root / "frontend" / "server.js").write_text("server")
    (root / "prisma" / "node_modules" / "@prisma" / "engines").mkdir(parents=True)
    return root


def test_a_directory_a_run_creates_is_a_change_even_when_empty(runtime: Path):
    before = settled(runtime)
    assert smoke_test.changed_files(before, smoke_test.snapshot(runtime)) == []

    (runtime / "frontend" / ".next" / "cache" / "images").mkdir(parents=True)

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    # The two new directories, and the one they were created in.
    assert "changed 3 file(s) or directories" in failure
    assert os.path.join("frontend", ".next", "cache") + os.sep in failure


def test_a_file_that_came_and_went_during_the_run_is_a_change(runtime: Path):
    """A temporary file, a lock, a pid file: gone by the time the run is
    over, and a write an installed bundle would have refused."""
    before = settled(runtime)
    passing = runtime / "frontend" / ".next" / "server" / "x.tmp"
    make_later_than(
        runtime / "frontend" / ".next" / "server", lambda: passing.write_text("x")
    )
    passing.unlink()

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert "changed 1 file(s) or directories" in failure
    assert os.path.join("frontend", ".next", "server") + os.sep in failure


def test_a_file_in_the_top_of_the_bundle_that_came_and_went_is_a_change(runtime: Path):
    before = settled(runtime)
    make_later_than(runtime, lambda: (runtime / "lock").write_text("x"))
    (runtime / "lock").unlink()
    assert len(smoke_test.changed_files(before, smoke_test.snapshot(runtime))) == 1


def settled(runtime: Path) -> dict:
    """A snapshot of a bundle that was not touched a moment ago. A real one
    was built minutes before it is run; here it was made in the same tick of
    a clock that, on Windows, moves every 16 ms, and a directory changed
    again within that tick keeps the time it had."""
    past = time.time() - 600
    for directory in [runtime, *(path for path in runtime.rglob("*") if path.is_dir())]:
        if not directory.is_symlink() and not is_junction(directory):
            os.utime(directory, (past, past))
    return smoke_test.snapshot(runtime)


def is_junction(path: Path) -> bool:
    return sys.platform == "win32" and bool(
        os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def make_later_than(directory: Path, change) -> None:
    """Make `change`, and make sure the directory's modification time is
    seen to move on a file system whose clock is coarser than this test."""
    was = directory.stat().st_mtime_ns
    change()
    if directory.stat().st_mtime_ns == was:
        os.utime(directory, ns=(was + 1_000_000, was + 1_000_000))


def test_a_link_a_run_creates_or_repoints_is_a_change(runtime: Path, tmp_path: Path):
    one, other = tmp_path / "one", tmp_path / "other"
    one.mkdir()
    other.mkdir()
    link = runtime / "frontend" / "link"
    make_link(link, one)
    before = settled(runtime)
    assert before[os.path.join("frontend", "link")][0] == "link"
    assert not any(
        name.startswith(os.path.join("frontend", "link") + os.sep) for name in before
    )

    remove_link(link)
    make_link(link, other)

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert os.path.join("frontend", "link") in failure


@posix_only
def test_a_socket_or_a_pipe_a_run_leaves_is_a_change(runtime: Path):
    before = settled(runtime)
    os.mkfifo(runtime / "frontend" / "pipe")
    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert os.path.join("frontend", "pipe") in failure


def make_link(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        from autogpt_desktop import winlinks

        winlinks.create_junction(target, link)  # a symlink needs a privilege
    else:
        link.symlink_to(target, target_is_directory=True)


def remove_link(link: Path) -> None:
    if sys.platform == "win32":
        link.rmdir()
    else:
        link.unlink()


def test_a_file_added_changed_or_removed_is_a_change(runtime: Path):
    before = settled(runtime)
    (runtime / "frontend" / "server.js").write_text("rewritten")
    (runtime / "prisma" / "libquery_engine.so.node").write_text("downloaded")
    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    # Both files, and the directory the new one is in.
    assert "changed 3 file(s)" in failure
    assert os.path.join("frontend", "server.js") in failure
    assert os.path.join("prisma", "libquery_engine.so.node") in failure

    again = settled(runtime)
    (runtime / "prisma" / "libquery_engine.so.node").unlink()
    (failure,) = smoke_test.changed_files(again, smoke_test.snapshot(runtime))
    assert os.path.join("prisma", "libquery_engine.so.node") in failure


def test_what_an_earlier_run_left_in_the_bundle_fails_the_run(runtime: Path, capsys):
    assert smoke_test.left_by_earlier_runs(runtime) == []

    (runtime / "frontend" / ".next" / "cache").mkdir()
    engines = runtime / "prisma" / "node_modules" / "@prisma" / "engines"
    (engines / "libquery_engine-debian-openssl-1.1.x.so.node").write_text("downloaded")

    (failure,) = smoke_test.left_by_earlier_runs(runtime)
    assert "libquery_engine-debian-openssl-1.1.x.so.node" in failure
    assert "cache" in failure and "--only seal" in failure


def test_a_log_that_speaks_of_a_refused_write_fails_a_read_only_run(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "api.log").write_text("INFO started\nINFO serving\n")
    assert smoke_test.refused_writes(tmp_path) == []

    (logs / "frontend.log").write_text(
        " ⨯ Failed to write image to cache abc Error: EACCES: permission denied, mkdir\n",
        encoding="utf-8",
    )
    (logs / "rabbitmq.log").write_text(
        "could not open: Read-only file system\nagain: EROFS\n"
    )

    failures = smoke_test.refused_writes(tmp_path)
    assert [failure.split(" ", 1)[0] for failure in failures] == [
        "frontend.log",
        "rabbitmq.log",
    ]
    assert "2 line(s)" in failures[1]

    # The backend's own file logs are a folder further down (LOG_DIR), and
    # the runtime's is beside the services'.
    (logs / "backend").mkdir()
    (logs / "backend" / "rest.log").write_text(
        "PermissionError: [Errno 13] Permission denied: 'x'\n"
    )
    (logs / "runtime.log").write_text("OSError: [Errno 30] Read-only file system\n")
    names = [
        failure.split(" ", 1)[0] for failure in smoke_test.refused_writes(tmp_path)
    ]
    assert names == ["backend/rest.log", "frontend.log", "rabbitmq.log", "runtime.log"]


def test_the_temporary_data_directory_is_removed_with_what_is_in_it(tmp_path: Path):
    data = tmp_path / "autogpt-smoke-x"
    (data / "postgres" / "base").mkdir(parents=True)
    (data / "postgres" / "PG_VERSION").write_text("18")
    (data / "rabbitmq").mkdir()
    cookie = data / "rabbitmq" / ".erlang.cookie"  # as rabbitmq.prepare writes it
    cookie.write_text("COOKIE")
    cookie.chmod(0o400)

    smoke_test.remove_data(data)

    assert not data.exists()
    smoke_test.remove_data(data)  # already gone: nothing to do, no error


def test_read_only_mode_is_refused_where_it_proves_nothing(monkeypatch):
    monkeypatch.setattr(smoke_test.sys, "platform", "win32")
    assert "macOS and Linux" in str(smoke_test.cannot_run_read_only())
    if hasattr(os, "geteuid"):
        monkeypatch.setattr(smoke_test.sys, "platform", "linux")
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        assert "root" in str(smoke_test.cannot_run_read_only())
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        assert smoke_test.cannot_run_read_only() is None


@posix_only
def test_the_bundle_is_unwritable_for_the_run_and_as_it_was_afterwards(runtime: Path):
    script = runtime / "frontend" / "server.js"
    script.chmod(0o755)
    before = {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in [runtime, *runtime.rglob("*")]
    }

    with pytest.raises(RuntimeError, match="the run failed"), smoke_test.unwritable(
        runtime
    ):
        assert stat.S_IMODE(script.stat().st_mode) == 0o555
        if os.geteuid() != 0:
            with pytest.raises(PermissionError):
                (runtime / "frontend" / ".next" / "cache").mkdir()
            with pytest.raises(PermissionError):
                script.write_text("changed")
        raise RuntimeError("the run failed")

    after = {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in [runtime, *runtime.rglob("*")]
    }
    assert after == before


@posix_only
def test_a_run_being_ended_can_put_the_permissions_back_first_and_only_once(
    runtime: Path,
):
    script = runtime / "frontend" / "server.js"
    with smoke_test.unwritable(runtime) as give_back:
        assert not os.access(runtime / "frontend", os.W_OK) or os.geteuid() == 0
        give_back()
        assert stat.S_IMODE(script.stat().st_mode) & 0o200
        script.chmod(0o400)  # what happens after is not undone a second time
        give_back()
    assert stat.S_IMODE(script.stat().st_mode) == 0o400


@pytest.fixture
def handlers():
    """This test process's own answers to the signals, put back afterwards."""
    before = {number: signal.getsignal(number) for number in smoke_test.END_SIGNALS}
    yield before
    for number, handler in before.items():
        signal.signal(number, handler)


def test_being_told_to_end_is_an_interrupt_so_that_everything_is_cleared_up(handlers):
    """Left to Python, SIGTERM ends the process where it stands: no
    `finally` runs, the data directory stays, a read-only bundle stays
    read-only."""
    smoke_test.end_on_signals()
    for number in smoke_test.END_SIGNALS:
        if number == signal.SIGINT:
            continue
        with pytest.raises(KeyboardInterrupt, match=signal.Signals(number).name):
            signal.getsignal(number)(number, None)


@posix_only
def test_a_signal_that_arrives_while_permissions_go_back_does_not_stop_that(handlers):
    smoke_test.end_on_signals()
    with smoke_test.deaf_to_signals():
        os.kill(os.getpid(), signal.SIGTERM)  # and no KeyboardInterrupt
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    with pytest.raises(KeyboardInterrupt):  # heard again afterwards
        os.kill(os.getpid(), signal.SIGTERM)
        signal.pause()


# A stand-in runtime: starts a service that outlives it, says which, and
# stays until its stdin closes.
RUNTIME = """
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, flush=True)
sys.stdin.read()
"""


def test_only_what_the_runtime_started_counts_as_left_running(tmp_path: Path):
    """The bundle's interpreter also runs the unit tests, perhaps right now
    and for somebody else: a process is a leftover because the runtime
    started it, not because of where its executable is."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    runtime = subprocess.Popen(
        [sys.executable, "-c", RUNTIME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert runtime.stdout and runtime.stdin
        service = psutil.Process(int(runtime.stdout.readline()))
        family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path, BUNDLE)
        family.look()
        runtime.stdin.close()
        runtime.wait(20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert not smoke_test.is_alive(service)  # ended, so the machine is left clean
        assert bystander.poll() is None
    finally:
        for process in (bystander, runtime):
            process.kill()
            process.wait(10)


def test_a_runtime_that_took_everything_with_it_left_nothing(tmp_path: Path):
    runtime = subprocess.Popen([sys.executable, "-c", "pass"])
    family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path, BUNDLE)
    runtime.wait(20)
    assert smoke_test.left_running(family) == []


def test_a_process_that_left_the_family_before_it_was_seen_is_still_found(
    tmp_path: Path, monkeypatch
):
    """A daemon: by the time anyone looks, its parent has gone and it is
    nobody's child. It runs from the bundle, was started after the runtime,
    and inherited what the runtime was started with; the developer's tests,
    running from the same bundle, did not, and are left alone."""
    monkeypatch.setattr(smoke_test.Family, "look", lambda self: None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    runtime = subprocess.Popen(
        [sys.executable, "-c", RUNTIME],
        env={**os.environ, "AUTOGPT_DESKTOP_DATA_DIR": str(tmp_path / "data")},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    daemon = None
    try:
        assert runtime.stdout and runtime.stdin
        daemon = psutil.Process(int(runtime.stdout.readline()))
        family = smoke_test.Family(
            psutil.Process(runtime.pid), tmp_path / "data", BUNDLE
        )
        runtime.stdin.close()
        runtime.wait(20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert not smoke_test.is_alive(daemon)
        assert bystander.poll() is None
    finally:
        for process in (bystander, runtime):
            process.kill()
            process.wait(10)
        if daemon and smoke_test.is_alive(daemon):
            daemon.kill()


def test_a_process_started_with_this_runs_data_directory_is_of_this_run(
    tmp_path: Path, monkeypatch
):
    """Whoever its parent is: here, this test, which is alive and no part of
    the run."""
    monkeypatch.setattr(smoke_test.Family, "look", lambda self: None)
    runtime = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    sleep = [sys.executable, "-c", "import time; time.sleep(120)"]
    family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path / "data", BUNDLE)
    of_this_run = subprocess.Popen(
        sleep, env={**os.environ, "RABBITMQ_BASE": str(tmp_path / "data" / "rabbitmq")}
    )
    bystander = subprocess.Popen(
        sleep, env={**os.environ, "RABBITMQ_BASE": str(tmp_path / "other")}
    )
    try:
        runtime.communicate(timeout=20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert of_this_run.wait(10) is not None
        assert bystander.poll() is None
    finally:
        for process in (bystander, of_this_run, runtime):
            process.kill()
            process.wait(10)


def test_a_process_working_in_the_data_directory_is_of_this_run(tmp_path: Path):
    """Valkey is started with nothing but a working directory to tell."""
    (tmp_path / "data" / "valkey").mkdir(parents=True)
    sleep = [sys.executable, "-c", "import time; time.sleep(120)"]
    inside = subprocess.Popen(sleep, cwd=tmp_path / "data" / "valkey")
    outside = subprocess.Popen(sleep, cwd=tmp_path)
    try:
        marks = [str(tmp_path / "data")]
        assert smoke_test.mentions(psutil.Process(inside.pid), marks)
        assert not smoke_test.mentions(psutil.Process(outside.pid), marks)
    finally:
        for process in (inside, outside):
            process.kill()
            process.wait(10)


# Where this interpreter's image is, as the check sees it for the processes
# started below. Not sys.executable: a framework build on macOS (GitHub's
# runners have one) runs Python.app from another directory.
BUNDLE = Path(psutil.Process().exe()).resolve().parent


@posix_only_signals
def test_a_run_started_with_nohup_outlives_its_terminal():
    """nohup runs a program with the hang-up signal ignored. The smoke test
    turns signals that would kill it into a tidy end, and must leave that one
    alone, or the run ends when the terminal it was started from closes."""
    probe = (
        "import signal, sys; sys.path.insert(0, sys.argv[1]); import smoke_test; "
        "signal.signal(signal.SIGHUP, signal.SIG_IGN); smoke_test.end_on_signals(); "
        "print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN, "
        "callable(signal.getsignal(signal.SIGTERM)))"
    )
    answer = subprocess.run(
        [sys.executable, "-c", probe, str(DESKTOP / "build")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert answer.stdout.split() == ["True", "True"], answer.stderr


# --- the links a run leaves outside the data directory ------------------------


@pytest.fixture
def link_roots(tmp_path: Path, monkeypatch) -> list[Path]:
    roots = [
        tmp_path / "cache" / "AutoGPT" / "links",
        tmp_path / "shared" / "AutoGPT-user",
    ]
    (tmp_path / "shared").mkdir()
    monkeypatch.setattr(smoke_test, "alias_roots", lambda: roots)
    return roots


def test_the_links_a_run_made_to_its_data_and_to_the_bundle_go_with_it(
    runtime: Path, tmp_path: Path, link_roots: list[Path]
):
    """With --data-dir too: the directory is the caller's, the link the
    runtime made to it is not."""
    data = tmp_path / "autogpt smoke"
    (data / "rabbitmq").mkdir(parents=True)
    (runtime / "erlang").mkdir()
    aliases = smoke_test.Aliases(runtime, data)

    for root in link_roots:
        root.mkdir(parents=True)
    make_link(link_roots[0] / "to-data", data / "rabbitmq")
    make_link(link_roots[0] / "to-erlang", runtime / "erlang")
    make_link(link_roots[1] / "to-data-too", data / "rabbitmq")
    aliases.remove_new()

    assert not any(os.path.lexists(root) for root in link_roots)
    assert not (
        tmp_path / "cache" / "AutoGPT"
    ).exists()  # made for the links, and empty
    assert (tmp_path / "shared").is_dir()
    assert (data / "rabbitmq").is_dir() and (runtime / "erlang").is_dir()


def test_links_that_were_there_or_lead_elsewhere_are_somebody_elses(
    runtime: Path, tmp_path: Path, link_roots: list[Path]
):
    data = tmp_path / "autogpt smoke"
    (data / "rabbitmq").mkdir(parents=True)
    installed = tmp_path / "the installed app's data"
    installed.mkdir()
    root = link_roots[0]
    root.mkdir(parents=True)
    make_link(root / "from-an-earlier-run", data / "rabbitmq")
    (root / "a-folder").mkdir()
    aliases = smoke_test.Aliases(runtime, data)

    make_link(root / "the-installed-apps", installed)  # it was started meanwhile
    make_link(root / "this-runs", data / "rabbitmq")
    aliases.remove_new()

    assert sorted(path.name for path in root.iterdir()) == [
        "a-folder",
        "from-an-earlier-run",
        "the-installed-apps",
    ]


def test_a_link_is_known_by_where_it_goes_not_by_how_the_path_is_spelled(
    tmp_path: Path,
):
    data = tmp_path / "data"
    (data / "rabbitmq").mkdir(parents=True)
    make_link(tmp_path / "another way in", tmp_path / "data")
    make_link(tmp_path / "link", tmp_path / "another way in" / "rabbitmq")

    assert smoke_test.leads_into(tmp_path / "link", data)
    assert not smoke_test.leads_into(tmp_path / "link", tmp_path / "elsewhere")


def test_a_stale_postmaster_pid_is_not_taken_for_this_runs_server(tmp_path: Path):
    """It names a process that is gone, or by now somebody else's, which
    must not be counted, let alone ended."""
    (tmp_path / "postgres").mkdir()
    (tmp_path / "postgres" / "postmaster.pid").write_text(f"{os.getppid()}\n")
    me = psutil.Process()
    assert smoke_test.nobodys_children(tmp_path, me) == []
    (tmp_path / "postgres" / "postmaster.pid").write_text("not a pid\n")
    assert smoke_test.nobodys_children(tmp_path, me) == []


# --- how long a start takes ----------------------------------------------------


def start(name: str, to_ready: float | None, **phases: float) -> "smoke_test.Start":
    return smoke_test.Start(
        name, to_ready, {"phase": "total", "phases": phases} if phases else None
    )


def test_a_restart_over_its_budget_fails_the_run_and_a_first_start_does_not():
    """A first start also creates the database; a restart is what every
    start of the app costs."""
    starts = [
        start("fresh_install", 300.0, database=40.0),
        start("restart", 95.0, queue=70.0, services=20.0),
        start("restart_with_a_password_reset", 30.0, queue=6.0),
    ]

    (failure,) = smoke_test.slow_restarts(starts, 90)

    assert "`restart` took 95s" in failure
    assert "budget is 90s (--restart-budget)" in failure
    assert "queue 70.0, services 20.0" in failure
    assert smoke_test.slow_restarts(starts, 120) == []


def test_a_start_that_never_became_ready_is_not_also_a_slow_one():
    """It is already a failure, in its own words."""
    assert smoke_test.slow_restarts([start("restart", None)], 90) == []


def test_the_table_of_starts_has_every_phase_and_a_dash_for_one_that_did_not_run(
    capsys,
):
    smoke_test.print_starts(
        [
            start("fresh_install", 24.1, config=0.03, database=6.5, queue=7.0),
            start("restart", None),
        ]
    )
    printed = capsys.readouterr().out.splitlines()
    header, first, second = printed[-3:]
    assert header.split() == ["start", "to", "ready", *smoke_test.PHASES]
    assert first.split() == [
        "fresh",
        "install",
        "24.1",
        "0.0",
        "6.5",
        "-",
        "7.0",
        "-",
        "-",
        "-",
    ]
    assert second.split()[:2] == ["restart", "never"]


def test_the_phases_the_table_shows_are_the_ones_the_runtime_reports():
    sys.path.insert(0, str(DESKTOP / "runtime"))
    timing = importlib.import_module("autogpt_desktop.timing")
    assert smoke_test.PHASES == timing.PHASES


class NoAliases:
    def __init__(self, runtime: Path, data: Path) -> None:
        pass

    def remove_new(self) -> None:
        pass


def test_the_restart_budget_is_a_parameter(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        smoke_test,
        "smoke",
        lambda runtime, data, args, give_back=None: seen.update(vars(args)) or 0,
    )
    monkeypatch.setattr(smoke_test, "end_on_signals", lambda: None)
    monkeypatch.setattr(smoke_test, "Aliases", NoAliases)
    monkeypatch.setattr(sys, "path", list(sys.path))  # main() adds the runtime to it
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_test.py",
            str(DESKTOP),
            "--restart-budget",
            "35",
            "--data-dir",
            str(DESKTOP),
        ],
    )
    assert smoke_test.main() == 0
    assert seen["restart_budget"] == 35
    monkeypatch.setattr(
        sys, "argv", ["smoke_test.py", str(DESKTOP), "--data-dir", str(DESKTOP)]
    )
    smoke_test.main()
    assert seen["restart_budget"] == smoke_test.RESTART_BUDGET_SECONDS


@pytest.mark.parametrize("record", ["stopped", "ended", "both"])
def test_a_stopped_turn_keeps_its_exact_identity_until_the_next_start(
    tmp_path: Path, record: str
):
    (tmp_path / "run").mkdir()
    stopping = smoke_test.Stopping("graph/run", "workers", "chat", "workers", None)
    if record in {"stopped", "both"}:
        (tmp_path / "run" / "stopped-runs.json").write_text(
            json.dumps(
                {
                    "graph_runs": ["run"],
                    "copilot_turns": ["chat"],
                    "copilot_turn_ids": {"chat": "turn"},
                }
            )
        )
    if record in {"ended", "both"}:
        (tmp_path / "run" / "ended-turns.json").write_text(
            json.dumps(
                [
                    {"session_id": "chat", "turn_id": "turn"},
                ]
            )
        )
    report = smoke_test.Report()
    smoke_test.check_stop_records(tmp_path, stopping, "turn", report)
    assert report.failures == []


@pytest.mark.parametrize(
    "record", [None, [], ["chat"], [{"session_id": "chat", "turn_id": "later"}]]
)
def test_missing_or_wrong_turn_intent_cannot_pass_the_stop_smoke_check(
    tmp_path: Path, record
):
    (tmp_path / "run").mkdir()
    if record is not None:
        (tmp_path / "run" / "ended-turns.json").write_text(json.dumps(record))
    stopping = smoke_test.Stopping("graph/run", "workers", "chat", "workers", None)
    report = smoke_test.Report()
    smoke_test.check_stop_records(tmp_path, stopping, "turn", report)
    assert report.failures


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_runs", ["other-run"]),
        ("copilot_turns", ["other-chat"]),
        ("copilot_turn_ids", {"chat": "later-turn"}),
    ],
)
def test_retained_stop_intent_cannot_name_different_work(
    tmp_path: Path, field: str, value
):
    (tmp_path / "run").mkdir()
    record = {
        "graph_runs": ["run"],
        "copilot_turns": ["chat"],
        "copilot_turn_ids": {"chat": "turn"},
    }
    record[field] = value
    (tmp_path / "run" / "stopped-runs.json").write_text(json.dumps(record))
    report = smoke_test.Report()
    smoke_test.check_stop_records(
        tmp_path,
        smoke_test.Stopping("graph/run", "workers", "chat", "workers", None),
        "turn",
        report,
    )
    assert report.failures


def test_persisted_run_refuses_a_cluster_that_is_still_running(tmp_path: Path):
    (tmp_path / "postgres").mkdir()
    (tmp_path / "postgres/postmaster.pid").write_text("123")
    with pytest.raises(RuntimeError, match="while PostgreSQL is running"):
        smoke_test.persisted_run(DESKTOP, tmp_path, "run")


@pytest.mark.skipif(
    not os.environ.get("AUTOGPT_TEST_POSTGRES_RUNTIME"),
    reason="requires an explicitly selected existing PostgreSQL runtime",
)
@pytest.mark.parametrize(
    "graph_status,node_status,expected",
    [
        ("TERMINATED", "FAILED", ("TERMINATED", 0)),
        ("RUNNING", "RUNNING", ("RUNNING", 1)),
    ],
)
def test_stopped_run_oracle_reads_durable_state_without_reconciliation(
    tmp_path: Path,
    graph_status: str,
    node_status: str,
    expected: tuple[str, int],
):
    from autogpt_desktop import ports, postgres
    from autogpt_desktop.layout import Bundle, DataDir
    from autogpt_desktop.process import wait_until

    bundle = Bundle(Path(os.environ["AUTOGPT_TEST_POSTGRES_RUNTIME"]))
    data = DataDir(tmp_path / "source")
    data.prepare()
    password = "synthetic-smoke-oracle-only"
    data.runtime_env.write_text(f"POSTGRES_PASSWORD={password}\n")
    postgres.initialize(bundle, data, password)
    port = ports.allocate(data.ports_file, ("postgres",))["postgres"]
    server = postgres.process(bundle, data, port)
    connection = None

    def connected():
        nonlocal connection
        try:
            connection = smoke_test.database(data.root)
            return True
        except smoke_test.psycopg2.OperationalError:
            return False

    try:
        server.start()
        assert wait_until(connected, 30, interval=0.1)
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA platform")
            cursor.execute(
                'CREATE TABLE platform."AgentGraphExecution" (id text, "executionStatus" text)'
            )
            cursor.execute(
                'CREATE TABLE platform."AgentNodeExecution" ("agentGraphExecutionId" text, "executionStatus" text)'
            )
            cursor.execute(
                'INSERT INTO platform."AgentGraphExecution" VALUES (%s, %s)',
                ("run", graph_status),
            )
            cursor.execute(
                'INSERT INTO platform."AgentNodeExecution" VALUES (%s, %s)',
                ("run", node_status),
            )
        connection.close()
        connection = None
    finally:
        if connection:
            connection.close()
        server.stop()
    before = smoke_test.snapshot(data.postgres)
    assert smoke_test.persisted_run(bundle.root, data.root, "graph/run") == expected
    assert smoke_test.changed_files(before, smoke_test.snapshot(data.postgres)) == []
    assert not (data.postgres / "postmaster.pid").exists()
