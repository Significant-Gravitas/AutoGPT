"""Tests for how the local test runner shares the docker test stack.

Every checkout's stack is the same compose project, so a run in one worktree
must not recreate or remove the containers another worktree's run is using.
"""

import os
import subprocess
from unittest.mock import patch

from scripts import run_tests

OTHER_CHECKOUT = os.path.join(os.sep, "code", "other-worktree", "backend")


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def _container(working_dir: str, compose_file: str = "docker-compose.test.yaml"):
    """A `docker ps` line: the container's compose working dir and config files."""
    return f"{working_dir}\t{os.path.join(working_dir, compose_file)}\n"


def test_other_stack_owners_leaves_out_this_checkout():
    listing = (
        _container(run_tests.BACKEND_DIR)
        + _container(OTHER_CHECKOUT)
        + _container(OTHER_CHECKOUT)
    )
    with patch("scripts.run_tests.subprocess.run", return_value=_completed(listing)):
        owners = run_tests.other_stack_owners()

    assert owners == {os.path.normcase(os.path.normpath(OTHER_CHECKOUT))}


def test_other_stack_owners_ignores_other_projects_named_backend():
    """Compose names a project after its directory, so an unrelated app
    started from some other `backend` directory is a `backend` project too.
    It isn't a test run, so it must neither be waited for nor block teardown."""
    other_app = os.path.join(os.sep, "code", "some-other-app", "backend")
    listing = _container(other_app, compose_file="docker-compose.yml")
    with patch("scripts.run_tests.subprocess.run", return_value=_completed(listing)):
        assert run_tests.other_stack_owners() == set()


def test_a_hung_docker_ps_counts_as_no_other_owner():
    """The compose command that runs next reports a Docker outage better than
    a traceback from the ownership check would."""
    with patch(
        "scripts.run_tests.subprocess.run",
        side_effect=subprocess.TimeoutExpired("docker", 10),
    ):
        assert run_tests.other_stack_owners() == set()


def test_waits_until_another_checkouts_run_has_finished():
    owners = iter([{OTHER_CHECKOUT}, {OTHER_CHECKOUT}, set()])
    with patch(
        "scripts.run_tests.other_stack_owners", side_effect=lambda: next(owners)
    ), patch("scripts.run_tests.time.sleep") as sleep:
        assert run_tests.wait_for_stack_to_be_free(max_wait=60) is True

    assert sleep.call_count == 2


def test_gives_up_while_another_checkout_still_holds_the_stack():
    with patch(
        "scripts.run_tests.other_stack_owners", return_value={OTHER_CHECKOUT}
    ), patch("scripts.run_tests.time.sleep"):
        assert run_tests.wait_for_stack_to_be_free(max_wait=0) is False


def test_tear_down_leaves_a_stack_another_checkout_has_taken_over():
    with patch(
        "scripts.run_tests.other_stack_owners", return_value={OTHER_CHECKOUT}
    ), patch("scripts.run_tests.run_command") as run_command:
        run_tests.tear_down_stack()

    run_command.assert_not_called()


def test_tear_down_takes_its_own_stack_down():
    with patch("scripts.run_tests.other_stack_owners", return_value=set()), patch(
        "scripts.run_tests.run_command"
    ) as run_command:
        run_tests.tear_down_stack()

    run_command.assert_called_once_with(
        ["docker", "compose", "-f", "docker-compose.test.yaml", "down"]
    )


def test_postgres_is_ready_only_once_the_postgres_role_answers():
    """During a fresh data directory's init, the server already accepts
    connections (pg_isready passes) before the `postgres` role exists."""
    role_missing = subprocess.CalledProcessError(
        2, "psql", stderr='FATAL:  role "postgres" does not exist'
    )
    with patch(
        "scripts.run_tests.subprocess.run",
        side_effect=[role_missing, _completed("1\n")],
    ) as run, patch("scripts.run_tests.time.sleep"):
        assert run_tests.wait_for_postgres() is True

    assert run.call_count == 2
    probe = run.call_args.args[0]
    assert "psql" in probe
    # No TTY, so nothing but psql's own output is compared with "1".
    assert "-T" in probe
