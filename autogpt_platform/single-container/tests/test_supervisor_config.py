from __future__ import annotations

import configparser
import re
import unittest
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = ASSET_DIR / "supervisor" / "supervisord.conf"
HEALTHCHECK_PATH = ASSET_DIR / "healthcheck.sh"

# Unraid ships Docker's stock stop timeout and operators must not have to raise
# it host-wide to run this appliance, so the whole supervised shutdown has to
# finish well inside this budget or Docker SIGKILLs the container (exit 137)
# with the data stores still running.
DOCKER_STOP_TIMEOUT_SECONDS = 10
# Headroom for supervisor's own event loop, process reaping and final exit.
SHUTDOWN_MARGIN_SECONDS = 2

# Programs that hold no durable state are signalled together, then the data
# stores. `fatal-exit` is an event listener; supervisor always places those in
# their own group, and its priority keeps it alive until everything it reports
# on has stopped.
RUNTIME_PROGRAMS = {
    "bootstrap",
    "database-manager",
    "scheduler",
    "batch-executor",
    "notification",
    "executor",
    "copilot-executor",
    "copilot-bot",
    "platform-linking-manager",
    "websocket",
    "rest",
    "next",
    "nginx",
    "watchdog",
}
STATE_PROGRAMS = {
    "postgres",
    "valkey-0",
    "valkey-1",
    "valkey-2",
    "rabbitmq",
    "falkordb",
}
GROUPS = {"runtime": RUNTIME_PROGRAMS, "state": STATE_PROGRAMS}
EVENT_LISTENERS = {"fatal-exit"}

# One-shot; it has normally already exited, so the healthcheck does not require
# it to be RUNNING.
ONE_SHOT_PROGRAMS = {"bootstrap"}


def load_supervisor_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    with SUPERVISOR_PATH.open(encoding="utf-8") as config_file:
        config.read_file(config_file)
    return config


def section_names(config: configparser.ConfigParser, prefix: str) -> set[str]:
    return {
        section.removeprefix(f"{prefix}:")
        for section in config.sections()
        if section.startswith(f"{prefix}:")
    }


class SupervisorShutdownTierTest(unittest.TestCase):
    def test_every_program_belongs_to_exactly_one_group(self) -> None:
        config = load_supervisor_config()

        self.assertEqual(section_names(config, "group"), set(GROUPS))
        self.assertEqual(
            section_names(config, "program"),
            RUNTIME_PROGRAMS | STATE_PROGRAMS,
            "a program outside both groups becomes its own stop tier",
        )
        for group, expected in GROUPS.items():
            with self.subTest(group=group):
                declared = {
                    program.strip()
                    for program in config[f"group:{group}"]["programs"].split(",")
                    if program.strip()
                }
                self.assertEqual(declared, expected)

    def test_state_services_stop_after_everything_that_uses_them(self) -> None:
        config = load_supervisor_config()

        runtime = config["group:runtime"].getint("priority")
        state = config["group:state"].getint("priority")
        listener = config["eventlistener:fatal-exit"].getint("priority")

        # Supervisor stops the highest priority group first.
        self.assertGreater(runtime, state)
        self.assertGreater(state, listener)

    def test_worst_case_shutdown_fits_inside_the_docker_stop_timeout(self) -> None:
        config = load_supervisor_config()

        def stop_wait(program: str, section: str) -> int:
            # Supervisor's built-in default is 10s, which alone exhausts the
            # budget, so every program must set this explicitly.
            self.assertIn(
                "stopwaitsecs",
                config[f"{section}:{program}"],
                f"{program} inherits supervisor's 10s default stopwaitsecs",
            )
            return config[f"{section}:{program}"].getint("stopwaitsecs")

        # Groups stop one after another, and a group is only done once its
        # slowest member has stopped, so the tiers add up.
        budget = sum(
            max(stop_wait(program, "program") for program in programs)
            for programs in GROUPS.values()
        )
        budget += sum(
            stop_wait(listener, "eventlistener") for listener in EVENT_LISTENERS
        )

        self.assertLessEqual(
            budget,
            DOCKER_STOP_TIMEOUT_SECONDS - SHUTDOWN_MARGIN_SECONDS,
            f"worst-case supervised shutdown is {budget}s; Docker SIGKILLs the "
            f"container at {DOCKER_STOP_TIMEOUT_SECONDS}s",
        )

    def test_postgres_uses_fast_shutdown(self) -> None:
        config = load_supervisor_config()

        # SIGTERM is PostgreSQL's *smart* shutdown: it waits for every client to
        # disconnect and so never completes on a deadline. SIGINT is the fast
        # shutdown - roll back open transactions, checkpoint, exit.
        self.assertEqual(config["program:postgres"]["stopsignal"], "INT")

    def test_supervisor_activity_log_reaches_container_logs_once(self) -> None:
        config = load_supervisor_config()

        # Supervisor's activity log names the program that stalled a shutdown,
        # so it has to reach `docker logs`. Under nodaemon it already mirrors
        # that log to stdout, so pointing `logfile` at stdout as well installs a
        # second handler on the same descriptor and prints every line twice.
        self.assertTrue(config["supervisord"].getboolean("nodaemon"))
        self.assertEqual(config["supervisord"]["logfile"], "/dev/null")


class HealthcheckSupervisorNamesTest(unittest.TestCase):
    def test_healthcheck_matches_grouped_program_names(self) -> None:
        healthcheck = HEALTHCHECK_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"local programs=\(\n(?P<programs>.*?)\n  \)", healthcheck, re.DOTALL
        )
        self.assertIsNotNone(match)
        assert match is not None

        checked = set(match.group("programs").split())
        expected = set(EVENT_LISTENERS) | {
            f"{group}:{program}"
            for group, programs in GROUPS.items()
            for program in programs - ONE_SHOT_PROGRAMS
        }
        # Grouping renames status lines to `group:program`; an un-updated list
        # would silently match nothing and pass every program.
        self.assertEqual(checked, expected)


if __name__ == "__main__":
    unittest.main()
