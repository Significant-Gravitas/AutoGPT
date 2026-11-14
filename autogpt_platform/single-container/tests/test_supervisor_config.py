from __future__ import annotations

import configparser
import re
import unittest
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = ASSET_DIR / "supervisor" / "supervisord.conf"
HEALTHCHECK_PATH = ASSET_DIR / "healthcheck.sh"
WATCHDOG_PATH = ASSET_DIR / "watchdog.sh"
SMOKE_PATH = (
    ASSET_DIR.parents[1] / ".github" / "scripts" / "platform-single-container-smoke.sh"
)
WORKFLOW_PATH = (
    ASSET_DIR.parents[1]
    / ".github"
    / "workflows"
    / "platform-single-container-docker.yml"
)

# Unraid ships Docker's stock stop timeout and operators must not have to raise
# it host-wide to run this appliance, so the whole supervised shutdown has to
# finish well inside this budget or Docker SIGKILLs the container (exit 137)
# with the data stores still running.
DOCKER_STOP_TIMEOUT_SECONDS = 10
# Supervisor stops one group per phase, and each phase costs more than its
# stopwaitsecs: `runforever()` polls with timeout=1 and needs at least one more
# iteration to reap what `ordered_stop_groups_phase_1` just stopped. Measured
# against supervisor 4.2.5 with every program ignoring SIGTERM, wall time came
# out at sum(stopwaitsecs) + ~1.4s across 2-, 3- and 4-phase layouts, so charge
# each phase for that rather than assuming stopwaitsecs is the whole cost.
# Measured flat, not per phase: 2 phases cost +1.30s, 3 cost +1.38s, 4 cost
# +1.27s. Charging per phase happens to fit at three and under-charges at two.
SUPERVISOR_SHUTDOWN_OVERHEAD_SECONDS = 1.5
SHUTDOWN_MARGIN_SECONDS = 1

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

    def test_event_listeners_are_accounted_for(self) -> None:
        config = load_supervisor_config()

        # Supervisor groups each event listener on its own, so an undeclared one
        # is an extra stop phase the budget never charged for.
        self.assertEqual(section_names(config, "eventlistener"), EVENT_LISTENERS)

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
        waits = [
            max(stop_wait(program, "program") for program in programs)
            for programs in GROUPS.values()
        ]
        waits += [stop_wait(listener, "eventlistener") for listener in EVENT_LISTENERS]
        budget = sum(waits) + SUPERVISOR_SHUTDOWN_OVERHEAD_SECONDS

        self.assertLessEqual(
            budget,
            DOCKER_STOP_TIMEOUT_SECONDS - SHUTDOWN_MARGIN_SECONDS,
            f"worst-case supervised shutdown is {sum(waits)}s of stopwaitsecs "
            f"plus {SUPERVISOR_SHUTDOWN_OVERHEAD_SECONDS}s of supervisor "
            f"overhead = {budget}s; Docker SIGKILLs the container at "
            f"{DOCKER_STOP_TIMEOUT_SECONDS}s",
        )

    def test_state_tier_holds_the_larger_share_of_the_budget(self) -> None:
        config = load_supervisor_config()

        def wait(program: str) -> int:
            return config[f"program:{program}"].getint("stopwaitsecs")

        # The sum alone would let the tiers be inverted. PostgreSQL's shutdown
        # checkpoint measured 3.2s on a seeded database, so the drainable tier
        # has to keep the larger cap.
        self.assertGreater(
            min(wait(program) for program in STATE_PROGRAMS),
            max(wait(program) for program in RUNTIME_PROGRAMS),
        )

    def test_postgres_uses_fast_shutdown(self) -> None:
        config = load_supervisor_config()

        # SIGTERM is PostgreSQL's *smart* shutdown: it waits for every client to
        # disconnect and so never completes on a deadline. SIGINT is the fast
        # shutdown - roll back open transactions, checkpoint, exit.
        self.assertEqual(config["program:postgres"]["stopsignal"], "INT")
        # run-service.sh execs the postmaster, so supervisor's child *is* the
        # postmaster. killpg'ing SIGINT would also hit backends, where INT means
        # cancel-query, racing the postmaster's own orchestrated shutdown.
        self.assertEqual(config["program:postgres"]["stopasgroup"], "false")

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


class WatchdogCadenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.watchdog = WATCHDOG_PATH.read_text(encoding="utf-8")
        self.smoke = SMOKE_PATH.read_text(encoding="utf-8")

    def _constant(self, name: str) -> int:
        match = re.search(rf"^readonly {name}=(\d+)$", self.watchdog, re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        return int(match.group(1))

    def test_failure_limit_remains_three(self) -> None:
        self.assertEqual(self._constant("FAILURE_LIMIT"), 3)

        force_check = self.smoke.split("force_watchdog_check() {", maxsplit=1)[1].split(
            "\n}\n", maxsplit=1
        )[0]
        count_position = force_check.index(
            'previous_count="$(count_container_log_evidence "${expected}")"'
        )
        signal_position = force_check.index(
            'docker exec "${CONTAINER_NAME}" kill -USR1 "${watchdog_pid}"'
        )
        wait_position = force_check.index(
            'wait_for_new_container_log_evidence "${expected}" "${previous_count}"'
        )
        self.assertLess(count_position, signal_position)
        self.assertLess(signal_position, wait_position)

        evidence_wait = self.smoke.split(
            "wait_for_new_container_log_evidence() {", maxsplit=1
        )[1].split("\n}\n", maxsplit=1)[0]
        self.assertIn("((current_count > previous_count))", evidence_wait)

    def test_steady_state_cadence_remains_thirty_seconds(self) -> None:
        self.assertEqual(self._constant("CHECK_INTERVAL"), 30)
        self.assertIn("while true; do\n    wait_for_next_check", self.watchdog)
        self.assertIn('sleep "${CHECK_INTERVAL}" &', self.watchdog)
        self.assertIn("trap queue_forced_check USR1", self.watchdog)

    def test_initial_health_poll_is_faster_than_steady_state_poll(self) -> None:
        self.assertEqual(self._constant("INITIAL_CHECK_INTERVAL"), 1)

        initial_wait = self.watchdog.split("wait_for_initial_health()", maxsplit=1)[
            1
        ].split("stop_appliance()", maxsplit=1)[0]
        self.assertIn('sleep "${INITIAL_CHECK_INTERVAL}"', initial_wait)
        self.assertNotIn('sleep "${CHECK_INTERVAL}"', initial_wait)


class WorkflowFailurePropagationTest(unittest.TestCase):
    def test_scans_join_smoke_and_fail_from_recorded_outcomes(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        validation_job = workflow.split("  build-and-scan:", maxsplit=1)[1].split(
            "  publish-platform-digests:", maxsplit=1
        )[0]

        self.assertIn(
            "      - name: Scan for fixable critical vulnerabilities\n"
            "        id: vulnerability_scan\n"
            "        continue-on-error: true",
            validation_job,
        )
        self.assertIn(
            "      - name: Scan image filesystem for embedded secrets\n"
            "        id: secret_scan\n"
            "        continue-on-error: true",
            validation_job,
        )
        self.assertIn(
            "      - name: Wait for complete-image smoke test\n"
            "        wait: complete-image-smoke",
            validation_job,
        )
        self.assertIn(
            "VULNERABILITY_SCAN_OUTCOME: ${{ steps.vulnerability_scan.outcome }}",
            validation_job,
        )
        self.assertIn(
            "SECRET_SCAN_OUTCOME: ${{ steps.secret_scan.outcome }}", validation_job
        )
        self.assertIn("((scan_failed == 0))", validation_job)


if __name__ == "__main__":
    unittest.main()
