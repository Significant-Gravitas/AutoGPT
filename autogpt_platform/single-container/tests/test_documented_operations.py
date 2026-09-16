from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
DOC_PATH = ASSET_DIR.parents[1] / "docs" / "platform" / "single-container.md"
COMMON_PATH = ASSET_DIR / "common.sh"
ENTRYPOINT_PATH = ASSET_DIR / "entrypoint.sh"
SUPERVISOR_PATH = ASSET_DIR / "supervisor" / "supervisord.conf"


def extract_bash_block(marker: str) -> str:
    blocks = re.findall(
        r"^```bash\n(.*?)\n```$",
        DOC_PATH.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    matching_blocks = [block for block in blocks if marker in block]
    if len(matching_blocks) != 1:
        raise AssertionError(
            f"Expected one documented Bash block containing {marker!r}, "
            f"found {len(matching_blocks)}"
        )
    return matching_blocks[0]


COLD_BACKUP_BLOCK = extract_bash_block("BACKUP_IMAGE_ID=")
RESTORE_BLOCK = extract_bash_block("RESTORE_CREATED=false")
STRUCTURAL_VALIDATION_BLOCK = extract_bash_block("test -s /data/config/runtime.env")
RESTORED_LAUNCH_BLOCK = extract_bash_block(
    ': "${ENV_FILE:?Set ENV_FILE to the recorded host environment-file path}"'
)


def extract_restore_requirements(block: str) -> tuple[tuple[str, str], ...]:
    requirements = re.findall(
        r"^\s*test (-[sd]) /data/([^\s]+)\s*$", block, flags=re.MULTILINE
    )
    if not requirements:
        raise AssertionError("Documented structural validation has no requirements")
    return tuple(
        (relative_path, "directory" if predicate == "-d" else "file")
        for predicate, relative_path in requirements
    )


REQUIRED_RESTORE_PATHS = extract_restore_requirements(STRUCTURAL_VALIDATION_BLOCK)

FAKE_DOCKER_PATH = Path(__file__).with_name("fake_docker.py")
POSTGRESQL_CONF = "listen_addresses = '127.0.0.1'\n"
PG_HBA_CONF = (
    "local all all peer\n"
    "host all all 127.0.0.1/32 scram-sha-256\n"
    "host all all ::1/128 scram-sha-256\n"
    "local replication all peer\n"
    "host replication all 127.0.0.1/32 scram-sha-256\n"
    "host replication all ::1/128 scram-sha-256\n"
)


class DocumentedOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temporary_directory.name)
        self.fake_root = self.work_dir / "fake-docker"
        self.volume_root = self.fake_root / "volumes"
        self.source_volume = self.volume_root / "autogpt-data"
        self.state_dir = self.fake_root / "state"
        self.bin_dir = self.work_dir / "bin"
        self.log_path = self.fake_root / "commands.jsonl"
        self.state_dir.mkdir(parents=True)
        self.source_volume.mkdir(parents=True)
        self.bin_dir.mkdir()
        self._set_running(True)
        self._seed_source_volume()
        fake_docker = self.bin_dir / "docker"
        shutil.copyfile(FAKE_DOCKER_PATH, fake_docker)
        fake_docker.chmod(0o755)
        fake_date = self.bin_dir / "date"
        fake_date.write_text(
            "#!/bin/sh\n"
            'if [ -n "${FAKE_DATE_OUTPUT:-}" ]; then\n'
            "  printf '%s\\n' \"${FAKE_DATE_OUTPUT}\"\n"
            "else\n"
            '  exec /bin/date "$@"\n'
            "fi\n",
            encoding="utf-8",
        )
        fake_date.chmod(0o755)
        host_environment = {
            name: os.environ[name]
            for name in ("LANG", "LC_ALL", "TMPDIR", "TZ")
            if name in os.environ
        }
        self.environment = {
            **host_environment,
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', os.defpath)}",
            "FAKE_DOCKER_ROOT": str(self.fake_root),
            "FAKE_DOCKER_VOLUME": "autogpt-data",
            "FAKE_DOCKER_LABELS": "{}",
            "FAKE_DOCKER_IMAGE_ID": "sha256:local-image-id",
            "FAKE_DOCKER_IMAGE_REF": "ghcr.io/example/autogpt:v1.2.3",
            "FAKE_DOCKER_IMAGE_DIGEST": (
                "ghcr.io/example/autogpt@sha256:"
                "0123456789abcdef0123456789abcdef"
                "0123456789abcdef0123456789abcdef"
            ),
        }
        resolved_docker = shutil.which("docker", path=self.environment["PATH"])
        self.assertIsNotNone(resolved_docker)
        self.assertEqual(Path(resolved_docker or "").resolve(), fake_docker.resolve())
        execution_probe = subprocess.run(
            [str(fake_docker)],
            cwd=self.work_dir,
            env=self.environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=10,
        )
        self.assertEqual(execution_probe.returncode, 2, execution_probe.stderr)
        self.assertIn("missing fake docker command", execution_probe.stderr)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_backup_restore_validation_and_restored_launch(self) -> None:
        backup = self._run(COLD_BACKUP_BLOCK)

        self.assertEqual(backup.returncode, 0, backup.stderr)
        self.assertTrue(self._is_running())
        self.assertIn("Image reference: ghcr.io/example/autogpt:v1.2.3", backup.stdout)
        self.assertIn("Image digest: ghcr.io/example/autogpt@sha256:", backup.stdout)
        backup_commands = self._commands()
        image_inspect = next(
            command
            for command in backup_commands
            if command[:2] == ["image", "inspect"]
        )
        self.assertEqual(image_inspect[-1], self.environment["FAKE_DOCKER_IMAGE_ID"])
        backup_tar = next(command for command in backup_commands if "-czf" in command)
        self.assertIn("--exclude=./cache", backup_tar)
        self.assertIn(self.environment["FAKE_DOCKER_IMAGE_ID"], backup_tar)
        backup_checksum = next(
            command
            for command in backup_commands
            if "sha256sum" in command and command[-1].endswith(".partial")
        )
        self.assertIn(self.environment["FAKE_DOCKER_IMAGE_ID"], backup_checksum)
        backup_tar_index = backup_commands.index(backup_tar)
        restart_index = next(
            index
            for index, command in enumerate(backup_commands)
            if command[0] == "start"
        )
        checksum_index = backup_commands.index(backup_checksum)
        self.assertLess(backup_tar_index, restart_index)
        self.assertLess(restart_index, checksum_index)
        archive_path, checksum_path = self._backup_artifacts()
        self.assertEqual(
            checksum_path.read_text(encoding="utf-8").split()[0],
            hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
        with tarfile.open(archive_path, "r:gz") as archive:
            archived_paths = {
                Path(member.name).parts[0]
                for member in archive.getmembers()
                if Path(member.name).parts
            }
        self.assertNotIn("cache", archived_paths)
        self.assertIn("config", archived_paths)
        self.assertIn("workspaces", archived_paths)

        restore = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=str(archive_path.parent),
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )

        self.assertEqual(restore.returncode, 0, restore.stderr)
        restore_commands = self._commands()[len(backup_commands) :]
        create_index = next(
            index
            for index, command in enumerate(restore_commands)
            if command[:2] == ["volume", "create"]
        )
        create_command = restore_commands[create_index]
        self.assertIn("--label", create_command)
        self.assertRegex(
            create_command[create_command.index("--label") + 1],
            r"^org[.]agpt[.]restore[.]owner=restore-",
        )
        owner_inspect_indexes = [
            index
            for index, command in enumerate(restore_commands)
            if command[:2] == ["volume", "inspect"] and "--format" in command
        ]
        self.assertEqual(len(owner_inspect_indexes), 1)
        extract_index = next(
            index for index, command in enumerate(restore_commands) if "-xzf" in command
        )
        self.assertLess(create_index, owner_inspect_indexes[0])
        self.assertLess(owner_inspect_indexes[0], extract_index)
        match = re.search(r"Restored .* into volume (\S+)", restore.stdout)
        self.assertIsNotNone(match, restore.stdout)
        restore_volume = match.group(1) if match else ""
        restored_data = self.volume_root / restore_volume
        self.assertTrue((restored_data / "config" / "runtime.env").is_file())
        self.assertFalse((restored_data / "cache").exists())
        self.assertEqual(
            (restored_data / "workspaces" / "example.txt").read_text(encoding="utf-8"),
            "durable workspace\n",
        )

        validation = self._run(
            STRUCTURAL_VALIDATION_BLOCK,
            RESTORE_VOLUME=restore_volume,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertEqual(validation.returncode, 0, validation.stderr)
        validation_command = self._commands()[-1]
        self.assertIn("--network", validation_command)
        self.assertEqual(
            validation_command[validation_command.index("--network") + 1], "none"
        )

        environment_file = self.work_dir / "autogpt.env"
        environment_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
        launch = self._run(
            RESTORED_LAUNCH_BLOCK,
            ENV_FILE=str(environment_file),
            RESTORE_VOLUME=restore_volume,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
            PUBLISH_SPEC="127.0.0.1:3300:3000",
            ADD_HOST_SPEC="host.docker.internal:host-gateway",
        )
        self.assertEqual(launch.returncode, 0, launch.stderr)
        launch_command = self._commands()[-1]
        self.assertIn(f"{restore_volume}:/data", launch_command)
        self.assertIn("127.0.0.1:3300:3000", launch_command)
        self.assertIn("host.docker.internal:host-gateway", launch_command)
        self.assertIn(str(environment_file), launch_command)

        for index, (relative_path, path_type) in enumerate(REQUIRED_RESTORE_PATHS):
            with self.subTest(required_path=relative_path):
                incomplete_volume = f"incomplete-restore-{index}"
                incomplete_data = self.volume_root / incomplete_volume
                shutil.copytree(restored_data, incomplete_data)
                missing_path = incomplete_data / relative_path
                if path_type == "file":
                    missing_path.unlink()
                else:
                    shutil.rmtree(missing_path)
                invalid = self._run(
                    STRUCTURAL_VALIDATION_BLOCK,
                    RESTORE_VOLUME=incomplete_volume,
                    RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
                )
                self.assertNotEqual(
                    invalid.returncode,
                    0,
                    f"missing {relative_path} unexpectedly passed",
                )
                self.assertIn(f"/data/{relative_path}", invalid.stderr)

        invalid_postgres_settings = (
            (
                "later-listen-override",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "Listen_Addresses '*'\n",
            ),
            (
                "auto-conf-listen-override",
                "postgres/postgresql.auto.conf",
                "listen_addresses = '*'\n",
            ),
            (
                "postgres-config-include",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "InClUdE = 'unsafe.conf'\n",
            ),
            (
                "postgres-hba-file",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "hba_file = '/tmp/unsafe-hba.conf'\n",
            ),
            (
                "auto-conf-hba-file",
                "postgres/postgresql.auto.conf",
                "HBA_FILE '/tmp/unsafe-hba.conf'\n",
            ),
            (
                "shared-preload-library",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "SHARED_PRELOAD_LIBRARIES 'unsafe'\n",
            ),
            (
                "local-preload-library",
                "postgres/postgresql.auto.conf",
                "local_preload_libraries = 'unsafe'\n",
            ),
            (
                "session-preload-library",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "session_preload_libraries 'unsafe'\n",
            ),
            (
                "archive-mode",
                "postgres/postgresql.auto.conf",
                "Archive_Mode = always\n",
            ),
            (
                "archive-command",
                "postgres/postgresql.conf",
                POSTGRESQL_CONF + "archive_command 'unsafe'\n",
            ),
            (
                "earlier-broad-trust-rule",
                "postgres/pg_hba.conf",
                "host all all 0.0.0.0/0 trust\n" + PG_HBA_CONF,
            ),
            (
                "replication-trust-rule",
                "postgres/pg_hba.conf",
                PG_HBA_CONF.replace(
                    "host replication all 127.0.0.1/32 scram-sha-256",
                    "host replication all 127.0.0.1/32 trust",
                ),
            ),
            (
                "hba-include",
                "postgres/pg_hba.conf",
                "include 'unsafe.conf'\n" + PG_HBA_CONF,
            ),
        )
        for index, (case, relative_path, content) in enumerate(
            invalid_postgres_settings
        ):
            with self.subTest(invalid_postgres_setting=case):
                invalid_volume = f"invalid-postgres-settings-{index}"
                invalid_data = self.volume_root / invalid_volume
                shutil.copytree(restored_data, invalid_data)
                (invalid_data / relative_path).write_text(content, encoding="utf-8")
                invalid = self._run(
                    STRUCTURAL_VALIDATION_BLOCK,
                    RESTORE_VOLUME=invalid_volume,
                    RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
                )
                self.assertNotEqual(
                    invalid.returncode,
                    0,
                    f"{case} unexpectedly passed:\n{invalid.stdout}\n{invalid.stderr}",
                )

    def test_restore_race_preserves_foreign_volume(self) -> None:
        backup = self._run(COLD_BACKUP_BLOCK)
        self.assertEqual(backup.returncode, 0, backup.stderr)
        archive_path, _ = self._backup_artifacts()
        command_count = len(self._commands())
        restore_volume = "contended-restore-volume"

        result = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=str(archive_path.parent),
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
            RESTORE_VOLUME=restore_volume,
            FAKE_DOCKER_PRECREATE_VOLUME_OWNER="other-restore-run",
            FAKE_DOCKER_FAIL="extract",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to populate", result.stderr)
        foreign_volume = self.volume_root / restore_volume
        self.assertTrue(foreign_volume.is_dir())
        self.assertEqual(
            (foreign_volume / "other-owner.txt").read_text(encoding="utf-8"),
            "preserve this volume\n",
        )
        self.assertFalse((foreign_volume / "config").exists())
        labels = json.loads(
            (self.state_dir / "volume-labels" / f"{restore_volume}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(labels, {"org.agpt.restore.owner": "other-restore-run"})
        restore_commands = self._commands()[command_count:]
        volume_operations = [
            command
            for command in restore_commands
            if command[:2] in (["volume", "inspect"], ["volume", "create"])
        ]
        self.assertEqual(
            [command[:2] for command in volume_operations],
            [
                ["volume", "inspect"],
                ["volume", "create"],
                ["volume", "inspect"],
            ],
        )
        self.assertFalse(any("-xzf" in command for command in restore_commands))
        self.assertFalse(
            any(command[:2] == ["volume", "rm"] for command in restore_commands)
        )

    def test_fake_validation_rewrites_only_data_mount_root(self) -> None:
        restore_volume = "boundary-aware-validation"
        shutil.copytree(self.source_volume, self.volume_root / restore_volume)
        validation_block = STRUCTURAL_VALIDATION_BLOCK.replace(
            '      quote="$(printf "\\047")"',
            "      literal=/database\n"
            '      test "${#literal}" -eq 9\n'
            '      test -s "/data/config/runtime.env"\n'
            '      quote="$(printf "\\047")"',
        )
        self.assertNotEqual(validation_block, STRUCTURAL_VALIDATION_BLOCK)

        result = self._run(
            validation_block,
            RESTORE_VOLUME=restore_volume,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backup_can_leave_the_container_stopped_for_upgrade(self) -> None:
        result = self._run(COLD_BACKUP_BLOCK, RESTART_AFTER_BACKUP="false")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self._is_running())
        self._backup_artifacts()
        self.assertFalse(any(command[0] == "start" for command in self._commands()))

    def test_backup_failures_clean_up_and_attempt_safe_recovery(self) -> None:
        expected_errors = {
            "stop": "injected stop failure",
            "tar": "injected tar failure",
            "checksum": "injected checksum failure",
            "malformed-checksum": "Backup checksum is not a valid SHA-256 digest",
            "start": "injected start failure",
        }
        for operation, expected_error in expected_errors.items():
            with self.subTest(operation=operation):
                self._reset_backup_state()
                result = self._run(COLD_BACKUP_BLOCK, FAKE_DOCKER_FAIL=operation)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self._assert_no_backup_artifacts()
                self.assertEqual(self._is_running(), operation != "start")
                if operation in {"tar", "checksum", "malformed-checksum", "start"}:
                    self.assertTrue(
                        any(command[0] == "start" for command in self._commands())
                    )

    def test_reset_backup_state_clears_stale_lock(self) -> None:
        lock_path = self.state_dir / "backup-lock"
        lock_path.write_text("stale", encoding="utf-8")

        self._reset_backup_state()

        self.assertFalse(lock_path.exists())

    def test_upgrade_mode_failure_restarts_and_cleans_up(self) -> None:
        expected_errors = {
            "tar": "injected tar failure",
            "checksum": "injected checksum failure",
            "malformed-checksum": "Backup checksum is not a valid SHA-256 digest",
        }
        for operation, expected_error in expected_errors.items():
            with self.subTest(operation=operation):
                self._reset_backup_state()
                result = self._run(
                    COLD_BACKUP_BLOCK,
                    FAKE_DOCKER_FAIL=operation,
                    RESTART_AFTER_BACKUP="false",
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self._assert_no_backup_artifacts()
                self.assertTrue(self._is_running())
                self.assertFalse((self.state_dir / "backup-lock").exists())
                self.assertTrue(
                    any(command[0] == "start" for command in self._commands())
                )

    def test_term_during_archive_cleans_up_and_restarts(self) -> None:
        result = self._run(COLD_BACKUP_BLOCK, FAKE_DOCKER_FAIL="signal-term")

        self.assertNotEqual(result.returncode, 0)
        self._assert_no_backup_artifacts()
        self.assertTrue(self._is_running())
        self.assertTrue(any(command[0] == "start" for command in self._commands()))

    def test_anonymous_volume_guards_and_unlabeled_named_volume(self) -> None:
        anonymous = self._run(
            COLD_BACKUP_BLOCK,
            FAKE_DOCKER_LABELS='{"com.docker.volume.anonymous":""}',
        )
        self.assertNotEqual(anonymous.returncode, 0)
        self.assertIn("anonymous volume", anonymous.stderr)
        self.assertTrue(self._is_running())

        hexadecimal_name = "a" * 64
        (self.volume_root / hexadecimal_name).mkdir()
        hexadecimal = self._run(
            COLD_BACKUP_BLOCK,
            FAKE_DOCKER_VOLUME=hexadecimal_name,
            FAKE_DOCKER_LABELS="null",
        )
        self.assertNotEqual(hexadecimal.returncode, 0)
        self.assertIn("anonymous volume", hexadecimal.stderr)
        self.assertTrue(self._is_running())

        named = self._run(COLD_BACKUP_BLOCK, FAKE_DOCKER_LABELS="null")
        self.assertEqual(named.returncode, 0, named.stderr)
        self.assertTrue(self._is_running())
        self._backup_artifacts()

    def test_backup_refuses_nonrunning_and_existing_artifacts(self) -> None:
        self._set_running(False)
        nonrunning = self._run(COLD_BACKUP_BLOCK)

        self.assertNotEqual(nonrunning.returncode, 0)
        self.assertIn("container is not running", nonrunning.stderr)
        self.assertFalse(self._is_running())
        self.assertFalse(any(command[0] == "start" for command in self._commands()))
        self.assertFalse((self.state_dir / "backup-lock").exists())

        self._set_running(True)
        backup_dir = self.work_dir / "autogpt-backups"
        backup_dir.mkdir(exist_ok=True)
        existing = backup_dir / "autogpt-data-20260825T123456Z.tgz"
        existing.write_text("existing backup", encoding="utf-8")
        collision = self._run(
            COLD_BACKUP_BLOCK,
            FAKE_DATE_OUTPUT="20260825T123456Z",
        )

        self.assertNotEqual(collision.returncode, 0)
        self.assertIn("Refusing to overwrite", collision.stderr)
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing backup")
        self.assertFalse((self.state_dir / "backup-lock").exists())

    def test_documented_restore_layout_matches_service_owners(self) -> None:
        entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")
        supervisor = SUPERVISOR_PATH.read_text(encoding="utf-8")
        common = COMMON_PATH.read_text(encoding="utf-8")
        installed_paths = {
            match.removeprefix("/data/")
            for match in re.findall(
                r"^\s*install -d [^\n]* (/data/[^\s]+)$",
                entrypoint,
                flags=re.MULTILINE,
            )
            if not match.startswith("/data/cache")
        }
        documented_paths = {
            relative_path for relative_path, _ in REQUIRED_RESTORE_PATHS
        }
        self.assertEqual(
            {path.split("/", maxsplit=1)[0] for path in installed_paths},
            {path.split("/", maxsplit=1)[0] for path in documented_paths},
        )
        implementation_leaf_directories = {
            path
            for path in installed_paths
            if path not in {"config", "postgres", "rabbitmq", "valkey"}
        }
        # RabbitMQ creates mnesia itself, so supervisor owns this durable path.
        implementation_leaf_directories.add("rabbitmq/mnesia")
        documented_directories = {
            relative_path
            for relative_path, path_type in REQUIRED_RESTORE_PATHS
            if path_type == "directory"
        }
        self.assertEqual(implementation_leaf_directories, documented_directories)
        self.assertIn(
            "RABBITMQ_MNESIA_BASE=/data/rabbitmq/mnesia",
            supervisor,
        )
        self.assertIn(
            'AUTOGPT_RUNTIME_ENV="${AUTOGPT_RUNTIME_ENV:-/data/config/runtime.env}"',
            common,
        )
        self.assertIn("local path=/data/config/backend.json", entrypoint)
        self.assertIn("${PGDATA}/PG_VERSION", entrypoint)
        self.assertIn("listen_addresses = '127.0.0.1'", entrypoint)
        self.assertIn("--auth-local=peer", entrypoint)
        self.assertIn("--auth-host=scram-sha-256", entrypoint)

    def test_restored_launch_supports_bash_3_when_available(self) -> None:
        stock_bash = Path("/bin/bash")
        if not stock_bash.is_file():
            self.skipTest("/bin/bash is unavailable")
        version = subprocess.run(
            [str(stock_bash), "--version"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=10,
        )
        if version.returncode != 0 or not re.search(
            r"version 3[.]", version.stdout.splitlines()[0]
        ):
            self.skipTest("/bin/bash is not Bash 3.x")
        environment_file = self.work_dir / "stock-bash.env"
        environment_file.write_text(
            "AUTOGPT_PUBLIC_URL=http://localhost:3000\n", encoding="utf-8"
        )

        result = subprocess.run(
            [str(stock_bash), "-c", RESTORED_LAUNCH_BLOCK],
            cwd=self.work_dir,
            env={
                **self.environment,
                "ENV_FILE": str(environment_file),
                "RESTORE_VOLUME": "autogpt-data",
                "RESTORE_IMAGE": self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
            },
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_concurrent_backup_is_refused_without_deleting_the_winner(self) -> None:
        first = subprocess.Popen(
            ["bash", "-c", COLD_BACKUP_BLOCK],
            cwd=self.work_dir,
            env={
                **self.environment,
                "BACKUP_DIR": "first-backups",
                "FAKE_DOCKER_DELAY_TAR": "0.5",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        try:
            lock_path = self.state_dir / "backup-lock"
            deadline = time.monotonic() + 5
            while not lock_path.is_file() and first.poll() is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertTrue(lock_path.is_file())

            second = self._run(COLD_BACKUP_BLOCK, BACKUP_DIR="second-backups")
            first_stdout, first_stderr = first.communicate(timeout=10)
        finally:
            if first.poll() is None:
                first.kill()
            first.wait(timeout=10)

        self.assertEqual(first.returncode, 0, first_stderr)
        self.assertIn("Backup written", first_stdout)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("another backup may be running", second.stderr)
        archive_path, checksum_path = self._backup_artifacts(
            self.work_dir / "first-backups"
        )
        self.assertEqual(
            checksum_path.read_text(encoding="utf-8").split()[0],
            hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
        second_backup_dir = self.work_dir / "second-backups"
        self.assertTrue(second_backup_dir.is_dir())
        self.assertFalse(list(second_backup_dir.iterdir()))
        self.assertFalse(lock_path.exists())

    def test_relative_backup_directory_and_restore_rejections(self) -> None:
        relative_backup_dir = "relative-backups"
        backup = self._run(COLD_BACKUP_BLOCK, BACKUP_DIR=relative_backup_dir)

        self.assertEqual(backup.returncode, 0, backup.stderr)
        backup_dir = self.work_dir / relative_backup_dir
        archive_path = next(backup_dir.glob("*.tgz"))
        checksum_path = next(backup_dir.glob("*.tgz.sha256"))
        archive_command = next(
            command for command in self._commands() if "-czf" in command
        )
        archive_mounts = [
            archive_command[index + 1]
            for index, argument in enumerate(archive_command[:-1])
            if argument == "--volume"
        ]
        canonical_backup_dir = backup_dir.resolve()
        self.assertIn(f"{canonical_backup_dir}:/backup", archive_mounts)

        restore = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=relative_backup_dir,
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertEqual(restore.returncode, 0, restore.stderr)
        restore_checksum = next(
            command
            for command in reversed(self._commands())
            if "sha256sum" in command and command[-1].endswith(".tgz")
        )
        self.assertIn(f"{canonical_backup_dir}:/backup:ro", restore_checksum)

        volume_create_count = sum(
            command[:2] == ["volume", "create"] for command in self._commands()
        )
        invalid_name = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=relative_backup_dir,
            BACKUP_FILE="../outside.tgz",
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertNotEqual(invalid_name.returncode, 0)
        self.assertIn("must be a filename", invalid_name.stderr)

        original_checksum = checksum_path.read_text(encoding="utf-8")
        checksum_path.write_text("not-a-digest\n", encoding="utf-8")
        invalid_checksum = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=relative_backup_dir,
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertNotEqual(invalid_checksum.returncode, 0)
        self.assertIn("not a valid SHA-256", invalid_checksum.stderr)

        checksum_path.write_text(f"{'0' * 64}  {archive_path.name}\n", encoding="utf-8")
        mismatch = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=relative_backup_dir,
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("checksum verification failed", mismatch.stderr)
        self.assertEqual(
            sum(command[:2] == ["volume", "create"] for command in self._commands()),
            volume_create_count,
        )
        checksum_path.write_text(original_checksum, encoding="utf-8")

        volumes_before_failure = {
            path.name for path in self.volume_root.iterdir() if path.is_dir()
        }
        failure_command_count = len(self._commands())
        failed_extract = self._run(
            RESTORE_BLOCK,
            BACKUP_DIR=relative_backup_dir,
            BACKUP_FILE=archive_path.name,
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
            FAKE_DOCKER_FAIL="extract",
        )
        self.assertNotEqual(failed_extract.returncode, 0)
        self.assertIn("partial restore volume was removed", failed_extract.stderr)
        self.assertEqual(
            {path.name for path in self.volume_root.iterdir() if path.is_dir()},
            volumes_before_failure,
        )
        failed_restore_commands = self._commands()[failure_command_count:]
        owner_inspect_indexes = [
            index
            for index, command in enumerate(failed_restore_commands)
            if command[:2] == ["volume", "inspect"] and "--format" in command
        ]
        remove_index = next(
            index
            for index, command in enumerate(failed_restore_commands)
            if command[:2] == ["volume", "rm"]
        )
        self.assertEqual(len(owner_inspect_indexes), 2)
        self.assertLess(owner_inspect_indexes[-1], remove_index)

        environment_file = self.work_dir / "restore.env"
        environment_file.write_text(
            "AUTOGPT_PUBLIC_URL=http://localhost:3000\n", encoding="utf-8"
        )
        missing_volume = self._run(
            RESTORED_LAUNCH_BLOCK,
            ENV_FILE=str(environment_file),
            RESTORE_VOLUME="missing-restored-volume",
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
        )
        self.assertNotEqual(missing_volume.returncode, 0)
        self.assertIn("Restored volume does not exist", missing_volume.stderr)

        existing_container = self._run(
            RESTORED_LAUNCH_BLOCK,
            ENV_FILE=str(environment_file),
            RESTORE_VOLUME="autogpt-data",
            RESTORE_IMAGE=self.environment["FAKE_DOCKER_IMAGE_DIGEST"],
            FAKE_DOCKER_EXISTING_CONTAINER="true",
        )
        self.assertNotEqual(existing_container.returncode, 0)
        self.assertIn("existing container named autogpt", existing_container.stderr)

    def _seed_source_volume(self) -> None:
        files = {
            "config/runtime.env": "POSTGRES_PASSWORD=test\n",
            "config/backend.json": '{"config": true}\n',
            "postgres/PG_VERSION": "16\n",
            "postgres/postgresql.conf": POSTGRESQL_CONF,
            "postgres/pg_hba.conf": PG_HBA_CONF,
            "workspaces/example.txt": "durable workspace\n",
            "cache/regenerable.txt": "do not archive\n",
        }
        for relative_path, content in files.items():
            path = self.source_volume / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for relative_path in (
            "rabbitmq/mnesia",
            "valkey/17000",
            "valkey/17001",
            "valkey/17002",
            "falkordb",
            "home",
            "frontend-home",
        ):
            (self.source_volume / relative_path).mkdir(parents=True)

    def _run(self, block: str, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", block],
            cwd=self.work_dir,
            env={**self.environment, **environment},
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=10,
        )

    def _backup_artifacts(self, backup_dir: Path | None = None) -> tuple[Path, Path]:
        backup_dir = backup_dir or self.work_dir / "autogpt-backups"
        archives = list(backup_dir.glob("*.tgz"))
        checksums = list(backup_dir.glob("*.tgz.sha256"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(len(checksums), 1)
        self.assertFalse(list(backup_dir.glob("*.partial")))
        return archives[0], checksums[0]

    def _assert_no_backup_artifacts(self) -> None:
        backup_dir = self.work_dir / "autogpt-backups"
        self.assertFalse(list(backup_dir.iterdir()) if backup_dir.exists() else [])

    def _reset_backup_state(self) -> None:
        backup_dir = self.work_dir / "autogpt-backups"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        (self.state_dir / "backup-lock").unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)
        self._set_running(True)

    def _commands(self) -> list[list[str]]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
        ]

    def _is_running(self) -> bool:
        return (self.state_dir / "running").read_text(encoding="utf-8") == "true"

    def _set_running(self, value: bool) -> None:
        (self.state_dir / "running").write_text(
            "true" if value else "false", encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
