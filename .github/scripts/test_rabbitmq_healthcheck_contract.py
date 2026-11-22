"""Every RabbitMQ healthcheck leaves the broker's Erlang cookie to the broker.

Docker runs a healthcheck as the image's user, root, and the RabbitMQ CLI
writes the Erlang cookie when it finds none. A probe that gets there before a
slow-booting broker leaves the cookie root-owned and 0400, and the broker,
which runs as `rabbitmq`, then fails its boot with `.erlang.cookie: eacces`.
So each healthcheck checks that the broker has written the cookie before it
runs the CLI; until then there is nothing to ping anyway.
"""

import re
import shlex
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKIE_GUARD = "test -f /var/lib/rabbitmq/.erlang.cookie && "
HOW_TO_COMPLY = (
    f"start the health command with `{COOKIE_GUARD}`, so a root probe never "
    "writes the cookie before the broker does (see "
    "autogpt_platform/docker-compose.platform.yml)"
)


def compose_healthchecks() -> dict[str, str]:
    """The health command of every RabbitMQ service in a Compose file."""
    commands = {}
    for path in sorted(REPO_ROOT.glob("autogpt_platform/**/docker-compose*.y*ml")):
        if "node_modules" in path.parts:
            continue
        services = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "services"
        ) or {}
        for name, service in services.items():
            test = ((service or {}).get("healthcheck") or {}).get("test")
            if test and str(service.get("image", "")).startswith("rabbitmq"):
                if isinstance(test, list):
                    test = test[1] if test[0] == "CMD-SHELL" else shlex.join(test[1:])
                commands[f"{path.relative_to(REPO_ROOT)}: {name}"] = test
    return commands


def workflow_healthchecks() -> dict[str, str]:
    """The health command of every RabbitMQ service container in a workflow."""
    commands = {}
    for path in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (workflow.get("jobs") or {}).items():
            for name, service in ((job or {}).get("services") or {}).items():
                if not str((service or {}).get("image", "")).startswith("rabbitmq"):
                    continue
                options = shlex.split(str(service.get("options", "")))
                for flag, value in zip(options, options[1:]):
                    if flag == "--health-cmd":
                        key = f"{path.relative_to(REPO_ROOT)}: {job_name}.{name}"
                        commands[key] = value
    return commands


class RabbitMQHealthcheckContractTest(unittest.TestCase):
    def test_compose_healthchecks_leave_the_cookie_to_the_broker(self) -> None:
        self._assert_guarded(compose_healthchecks())

    def test_ci_service_healthchecks_leave_the_cookie_to_the_broker(self) -> None:
        self._assert_guarded(workflow_healthchecks())

    def _assert_guarded(self, commands: dict[str, str]) -> None:
        self.assertTrue(commands, "found no RabbitMQ healthcheck to check")
        unguarded = {
            where: command
            for where, command in commands.items()
            if re.search(r"\brabbitmq(?:-diagnostics|ctl)\b", command)
            and not command.startswith(COOKIE_GUARD)
        }
        self.assertEqual(unguarded, {}, HOW_TO_COMPLY)


if __name__ == "__main__":
    unittest.main()
