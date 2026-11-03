"""The Docker Compose stack's data services: reachable from this machine only,
and logged into with the passwords `make init-env` generates per install."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PLATFORM_ROOT = Path(__file__).parents[2]
COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.platform.yml",
    "backend/docker-compose.test.yaml",
)
DATA_SERVICES = {
    "db",
    "redis-0",
    "redis-1",
    "redis-2",
    "rabbitmq",
    "falkordb",
    "clamav",
}
DB_ENTRYPOINT = "/usr/local/bin/autogpt-db-entrypoint.sh"
# The image's own entrypoint, which db/entrypoint.sh sources for its helpers,
# stubbed so each records what it was asked to do.
IMAGE_ENTRYPOINT_STUB = """
docker_setup_env() { declare -g DATABASE_ALREADY_EXISTS="${STUB_EXISTS:-}"; }
docker_create_db_directories() { :; }
docker_temp_server_start() {
  echo temp-start >> "$STUB_LOG"
  [ -z "${STUB_START_FAILS:-}" ]
}
docker_temp_server_stop() { echo temp-stop >> "$STUB_LOG"; }
docker_process_sql() {
  cat > /dev/null
  echo password-set >> "$STUB_LOG"
  [ -z "${STUB_SQL_FAILS:-}" ]
}
"""
# redis-cli, stubbed: each call is logged; the cluster has its slots assigned
# unless STUB_SLOTS says otherwise, forms once its shards have met if
# STUB_HEALS is set, and redis-1 holds STUB_KEYS keys, or doesn't answer if
# that is "none".
REDIS_CLI_STUB = """
host=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h) host=$2; shift 2 ;;
    -p) shift 2 ;;
    *) break ;;
  esac
done
echo "$host $*" >> "$STUB_LOG"
case "$*" in
  ping) echo PONG ;;
  "cluster info")
    if [ -n "${STUB_HEALS:-}" ] && grep -q "cluster meet" "$STUB_LOG"; then
      echo cluster_state:ok
    else
      echo cluster_state:fail
    fi
    echo "cluster_slots_assigned:${STUB_SLOTS:-16384}" ;;
  dbsize)
    if [ "$host" != redis-1 ]; then echo 0
    elif [ "${STUB_KEYS:-0}" = none ]; then exit 1
    else echo "${STUB_KEYS:-0}"
    fi ;;
esac
"""

SPEC = importlib.util.spec_from_file_location(
    "single_container_runtime_config", Path(__file__).parents[1] / "runtime_config.py"
)
assert SPEC is not None and SPEC.loader is not None
runtime_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_config)


class ComposeCredentialsTest(unittest.TestCase):
    def test_data_services_publish_ports_on_loopback_only(self) -> None:
        published = {
            (name, service, port)
            for name in COMPOSE_FILES
            for service, ports in _published_ports(PLATFORM_ROOT / name).items()
            if service in DATA_SERVICES
            for port in ports
        }

        self.assertTrue(published)
        self.assertEqual(
            [
                entry
                for entry in sorted(published)
                if not entry[2].startswith("127.0.0.1:")
            ],
            [],
        )

    def test_connection_strings_take_the_generated_database_password(self) -> None:
        for name in ("docker-compose.yml", "docker-compose.platform.yml"):
            text = (PLATFORM_ROOT / name).read_text(encoding="utf-8")
            for password in re.findall(r"postgresql://[^:@\s/]+:([^@\s]*)@", text):
                self.assertTrue(
                    password.startswith("${POSTGRES_PASSWORD:?"), f"{name}: {password}"
                )

    def test_service_passwords_are_never_written_into_compose(self) -> None:
        names = "|".join(sorted(runtime_config.CREDENTIALS - {"DATABASE_URL"}))
        assignment = re.compile(rf"^\s*-?\s*({names})\s*[:=]\s*(\S.*)$")
        for name in COMPOSE_FILES:
            for line in (PLATFORM_ROOT / name).read_text(encoding="utf-8").splitlines():
                match = assignment.match(line)
                if match:
                    self.assertTrue(
                        match.group(2).startswith("${"), f"{name}: {line.strip()}"
                    )

    def test_backend_ci_tests_log_in_with_the_passwords_its_services_require(
        self,
    ) -> None:
        workflow = (
            PLATFORM_ROOT.parent / ".github/workflows/platform-backend-ci.yml"
        ).read_text(encoding="utf-8")
        # The `test` job alone, up to the next job: what another job sets
        # reaches neither its services nor its tests.
        (job,) = re.findall(r"(?ms)^  test:\n(.*?)(?=^  \S|\Z)", workflow)
        (env,) = re.findall(r"(?m)^    env:\n((?:^      .*\n)+)", job)
        job_env = dict(re.findall(r'(?m)^      ([A-Z_]+): "([^"]*)"$', env))
        (falkordb,) = re.findall(r'REDIS_ARGS: "--requirepass ([^"]+)"', job)

        self.assertEqual(job_env.get("GRAPHITI_FALKORDB_PASSWORD"), falkordb)
        self.assertIn("RABBITMQ_DEFAULT_PASS: ${{ env.RABBITMQ_DEFAULT_PASS }}", job)
        self.assertIn("RABBITMQ_DEFAULT_PASS", job_env)

    def test_database_containers_run_the_password_entrypoint(self) -> None:
        for name, script in (
            ("docker-compose.yml", "./db/entrypoint.sh"),
            ("backend/docker-compose.test.yaml", "../db/entrypoint.sh"),
        ):
            text = (PLATFORM_ROOT / name).read_text(encoding="utf-8")
            self.assertIn(f'entrypoint: ["bash", "{DB_ENTRYPOINT}"]', text, name)
            self.assertIn(f"- {script}:{DB_ENTRYPOINT}:ro", text, name)
            self.assertTrue((PLATFORM_ROOT / name).parent.joinpath(script).is_file())

    @unittest.skipIf(sys.platform == "win32", "bash there is WSL's, outside this path")
    def test_database_entrypoint_parses(self) -> None:
        subprocess.run(
            ["bash", "-n", str(PLATFORM_ROOT / "db/entrypoint.sh")], check=True
        )


@unittest.skipIf(sys.platform == "win32", "bash there is WSL's, outside this path")
class DatabaseEntrypointTest(unittest.TestCase):
    """What db/entrypoint.sh decides, with the image's helpers stubbed out.
    That the password really changes takes a real database to show."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        image_entrypoint = self.root / "image-entrypoint.sh"
        image_entrypoint.write_text(IMAGE_ENTRYPOINT_STUB, encoding="utf-8")
        (self.root / "bin").mkdir()
        for command, body in (
            ("docker-entrypoint.sh", 'echo "stock $*" >> "$STUB_LOG"'),
            # As the postgres user, whoever runs the test, unless STUB_UID says.
            ("id", 'echo "${STUB_UID:-999}"'),
            # Takes root's place as gosu does, after making the script
            # unreadable, as a checkout's file can be to the postgres user.
            (
                "gosu",
                'echo "gosu $1" >> "$STUB_LOG"; shift; chmod 000 "$STUB_SCRIPT"; '
                'export STUB_UID=999; exec "$@"',
            ),
        ):
            path = self.root / "bin" / command
            path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
            path.chmod(0o755)
        script = (PLATFORM_ROOT / "db/entrypoint.sh").read_text(encoding="utf-8")
        self.script = self.root / "entrypoint.sh"
        self.script.write_text(
            script.replace(
                "/usr/local/bin/docker-entrypoint.sh", str(image_entrypoint), 1
            ),
            encoding="utf-8",
        )

    def test_an_existing_database_takes_the_password_before_it_starts(self) -> None:
        self.assertEqual(
            self._run("postgres", STUB_EXISTS="1"),
            (0, ["temp-start", "password-set", "temp-stop", "stock postgres"]),
        )

    def test_a_database_whose_password_cant_be_set_is_not_started(self) -> None:
        """It would go on accepting only its old password, which may be one
        the repository published."""
        for failure in ("STUB_SQL_FAILS", "STUB_START_FAILS"):
            with self.subTest(failure=failure):
                code, log = self._run("postgres", STUB_EXISTS="1", **{failure: "1"})

                self.assertEqual(code, 1)
                self.assertNotIn("stock postgres", log)
                self.assertIn("so it is not started", self.stderr)

    def test_a_new_data_directory_is_left_to_the_image(self) -> None:
        self.assertEqual(self._run("postgres", STUB_EXISTS=""), (0, ["stock postgres"]))

    @unittest.skipIf(
        os.name == "posix" and os.geteuid() == 0, "root reads an unreadable file"
    )
    def test_started_as_root_it_runs_as_postgres_without_rereading_its_file(
        self,
    ) -> None:
        self.assertEqual(
            self._run(
                "postgres", STUB_EXISTS="1", STUB_UID="0", STUB_SCRIPT=str(self.script)
            ),
            (
                0,
                [
                    "gosu postgres",
                    "temp-start",
                    "password-set",
                    "temp-stop",
                    "stock postgres",
                ],
            ),
        )

    def test_printing_help_or_a_setting_starts_no_server(self) -> None:
        for arguments in (
            ["-C", "max_connections"],
            ["-Cmax_connections"],
            ["-sC", "max_connections"],
            ["--help"],
            ["-V"],
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    self._run("postgres", *arguments, STUB_EXISTS="1"),
                    (0, [f"stock postgres {' '.join(arguments)}"]),
                )

    def test_an_image_without_the_helpers_starts_no_existing_database(self) -> None:
        """Say an image renamed one of them: the password could not be
        changed, so the database is not started on the one it has."""
        stub = self.root / "image-entrypoint.sh"
        stub.write_text(
            IMAGE_ENTRYPOINT_STUB.replace("docker_temp_server_start", "renamed"),
            encoding="utf-8",
        )
        data = self.root / "data"
        data.mkdir()

        self.assertEqual(
            self._run("postgres", PGDATA=str(data)), (0, ["stock postgres"])
        )
        (data / "PG_VERSION").write_text("15\n", encoding="utf-8")
        code, log = self._run("postgres", PGDATA=str(data))

        self.assertEqual((code, log), (1, []))
        self.assertIn("so it is not started", self.stderr)

    def test_an_option_value_that_looks_like_dash_c_is_not_taken_for_it(self) -> None:
        for arguments in (
            ["-c", "log_line_prefix= -C "],
            ["-D", "-Cdata"],
            # Several short options in one argument, the last taking a value.
            ["-Fk", "-Csocket"],
            ["--cluster-name=-Cname", "-Fp", "5432"],
        ):
            with self.subTest(arguments=arguments):
                _, log = self._run("postgres", *arguments, STUB_EXISTS="1")

                self.assertEqual(log[:3], ["temp-start", "password-set", "temp-stop"])

    def _run(self, *arguments: str, **stubs: str) -> tuple[int, list[str]]:
        log = self.root / "log"
        log.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(self.script), *arguments],
            env={
                "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
                "STUB_LOG": str(log),
                "POSTGRES_PASSWORD": "new-password",
                **stubs,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.stderr = result.stderr
        return result.returncode, log.read_text(encoding="utf-8").splitlines()


@unittest.skipIf(sys.platform == "win32", "bash there is WSL's, outside this path")
class RedisInitTest(unittest.TestCase):
    """What redis-init does to the cluster, with redis-cli stubbed. Work in
    progress lives in Redis, so a shard that holds any is never emptied."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "bin").mkdir()
        for command, body in (
            ("redis-cli", REDIS_CLI_STUB),
            ("getent", 'echo "10.0.0.9 $2"'),
            ("sleep", ":"),
        ):
            path = self.root / "bin" / command
            path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
            path.chmod(0o755)

    def test_shards_compose_recreated_are_joined_back_up(self) -> None:
        code, log = self._run(STUB_HEALS="1")

        self.assertEqual(code, 0)
        self.assertTrue(any("cluster meet" in line for line in log))
        self.assertFalse([line for line in log if "reset" in line or "create" in line])

    def test_shards_that_hold_data_are_never_reset(self) -> None:
        """Nor are they when one shard doesn't say it holds nothing."""
        for keys in ("5", "none"):
            with self.subTest(keys=keys):
                code, log = self._run(STUB_KEYS=keys)

                self.assertEqual(code, 1)
                self.assertFalse(
                    [
                        line
                        for line in log
                        if any(w in line for w in ("reset", "flush", "create"))
                    ]
                )
                self.assertIn("so nothing is reset", self.stderr)

    def test_a_cluster_with_nothing_in_it_is_created_afresh(self) -> None:
        for slots in ("0", "16384"):
            with self.subTest(slots=slots):
                code, log = self._run(STUB_SLOTS=slots)

                self.assertEqual(code, 0)
                self.assertEqual(sum("cluster reset hard" in line for line in log), 3)
                self.assertIn(
                    " --cluster create redis-0:17000 redis-1:17001 redis-2:17002"
                    " --cluster-replicas 0 --cluster-yes",
                    log,
                )

    def _run(self, **stubs: str) -> tuple[int, list[str]]:
        log = self.root / "log"
        log.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["sh", "-c", _redis_init_script()],
            env={
                "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
                "STUB_LOG": str(log),
                "REDIS_PASSWORD": "a-password",
                **stubs,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.stderr = result.stderr
        return result.returncode, log.read_text(encoding="utf-8").splitlines()


def _redis_init_script() -> str:
    """The script redis-init hands `sh -c`: the block under its `command:`,
    with each `$$` read as `$`, as Compose reads it."""
    lines = (
        (PLATFORM_ROOT / "docker-compose.platform.yml")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    start = lines.index("      - |", lines.index("  redis-init:")) + 1
    script = []
    for line in lines[start:]:
        if line.strip() and not line.startswith(" " * 8):
            break
        script.append(line[8:])
    return "\n".join(script).replace("$$", "$") + "\n"


def _published_ports(path: Path) -> dict[str, list[str]]:
    """`ports:` entries per service, read the way these files are laid out:
    services two spaces in, their keys four, list items under them."""
    ports: dict[str, list[str]] = {}
    in_services = False
    service = None
    ports_indent = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_services, service, ports_indent = line == "services:", None, None
            continue
        if in_services and indent == 2 and line.endswith(":"):
            service, ports_indent = line.strip()[:-1], None
            continue
        if ports_indent is not None:
            if indent > ports_indent and line.lstrip().startswith("- "):
                entry = line.lstrip()[2:].strip().strip("\"'")
                ports.setdefault(service or "", []).append(entry)
                continue
            ports_indent = None
        if service and line.strip() == "ports:":
            ports_indent = indent
    return ports


if __name__ == "__main__":
    unittest.main()
