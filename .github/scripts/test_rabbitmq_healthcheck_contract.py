"""Every RabbitMQ healthcheck leaves the broker's Erlang cookie to the broker.

Docker runs a healthcheck as the image's user, root, and the RabbitMQ CLI
writes the Erlang cookie when it finds none. A probe that gets there before a
slow-booting broker leaves the cookie root-owned and 0400, and the broker,
which runs as `rabbitmq`, then fails its boot with `.erlang.cookie: eacces`.
So each healthcheck checks that the broker has written the cookie before it
runs the CLI; until then there is nothing to ping anyway.

The check is deliberately simple. A health command must start with the guard,
so an equivalent guard written another way is refused, and nothing after the
first CLI call is checked (a trailing `; rabbitmqctl ...` would pass). It reads
Compose healthchecks and workflow service containers' `--health-cmd`, not a
step's own `docker run` or a composite action.
"""

import re
import shlex
import subprocess
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKIE_GUARD = "test -f /var/lib/rabbitmq/.erlang.cookie && "
# Each of these starts an Erlang node, which writes a missing cookie.
RABBITMQ_CLI = re.compile(
    r"\brabbitmq(?:ctl|-diagnostics|-plugins|-queues|-streams|-upgrade)\b"
)
HOW_TO_COMPLY = (
    f"start the health command with `{COOKIE_GUARD}`, so a root probe never "
    "writes the cookie before the broker does (see "
    "autogpt_platform/docker-compose.platform.yml)"
)


def compose_healthchecks() -> dict[str, str]:
    """Every Compose service's health command, as the shell runs it."""
    commands = {}
    for path in _compose_files():
        services = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "services"
        ) or {}
        for name, service in services.items():
            test = ((service or {}).get("healthcheck") or {}).get("test")
            if isinstance(test, list):
                test = test[1] if test[:1] == ["CMD-SHELL"] else shlex.join(test[1:])
            if test:
                commands[f"{path.relative_to(REPO_ROOT)}: {name}"] = test
    return commands


def workflow_healthchecks() -> dict[str, str]:
    """Every workflow service container's `--health-cmd`."""
    commands = {}
    for path in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (workflow.get("jobs") or {}).items():
            for name, service in ((job or {}).get("services") or {}).items():
                options = str((service or {}).get("options", ""))
                for command in health_commands(options):
                    key = f"{path.relative_to(REPO_ROOT)}: {job_name}.{name}"
                    commands[key] = command
    return commands


def health_commands(options: str) -> list[str]:
    """The `--health-cmd` values in `docker create` options, in either form."""
    tokens = shlex.split(options)
    commands = []
    for flag, value in zip(tokens, tokens[1:] + [""]):
        if flag.startswith("--health-cmd="):
            flag, value = flag.split("=", 1)
        if flag == "--health-cmd":
            commands.append(value)
    return commands


def _compose_files() -> list[Path]:
    """The Compose files git tracks under autogpt_platform/. A developer's
    uncommitted local override isn't the repository's to check."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "autogpt_platform"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    return sorted(
        REPO_ROOT / path
        for path in tracked
        if path and re.fullmatch(r"docker-compose.*\.ya?ml", Path(path).name)
    )


class RabbitMQHealthcheckContractTest(unittest.TestCase):
    def test_compose_healthchecks_leave_the_cookie_to_the_broker(self) -> None:
        self._assert_guarded(compose_healthchecks())

    def test_ci_service_healthchecks_leave_the_cookie_to_the_broker(self) -> None:
        self._assert_guarded(workflow_healthchecks())

    def test_health_commands_are_read_in_either_option_form(self) -> None:
        options = '--health-cmd "a b" --health-interval 30s --health-cmd="c d"'
        self.assertEqual(health_commands(options), ["a b", "c d"])

    def _assert_guarded(self, commands: dict[str, str]) -> None:
        probes = {
            where: command
            for where, command in commands.items()
            if RABBITMQ_CLI.search(command)
        }
        self.assertTrue(
            probes,
            "found no healthcheck that runs the RabbitMQ CLI; if none should, "
            "this test has nothing left to check and can go",
        )
        unguarded = {
            where: command
            for where, command in probes.items()
            if not command.startswith(COOKIE_GUARD)
        }
        self.assertEqual(unguarded, {}, HOW_TO_COMPLY)


if __name__ == "__main__":
    unittest.main()
