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

REQUIRED_RESTORE_PATHS = (
    ("config/runtime.env", "file"),
    ("config/backend.json", "file"),
    ("postgres/PG_VERSION", "file"),
    ("rabbitmq/mnesia", "directory"),
    ("valkey/17000", "directory"),
    ("valkey/17001", "directory"),
    ("valkey/17002", "directory"),
    ("falkordb", "directory"),
    ("workspaces", "directory"),
    ("home", "directory"),
    ("frontend-home", "directory"),
)

FAKE_DOCKER = r"""#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(os.environ["FAKE_DOCKER_ROOT"])
STATE_DIR = ROOT / "state"
VOLUME_DIR = ROOT / "volumes"
LOG_PATH = ROOT / "commands.jsonl"
ARGS = sys.argv[1:]


def finish(status: int, output: str = "", error: str = "") -> None:
    if output:
        print(output)
    if error:
        print(error, file=sys.stderr)
    raise SystemExit(status)


def record() -> None:
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(ARGS) + "\n")


def should_fail(operation: str) -> bool:
    failures = os.environ.get("FAKE_DOCKER_FAIL", "").split(",")
    return operation in failures


def require_network_none() -> None:
    if "--network" not in ARGS:
        finish(2, error="documented helper did not disable networking")
    if ARGS[ARGS.index("--network") + 1] != "none":
        finish(2, error="documented helper used an unexpected network mode")


def running() -> bool:
    return (STATE_DIR / "running").read_text(encoding="utf-8").strip() == "true"


def set_running(value: bool) -> None:
    (STATE_DIR / "running").write_text(
        "true" if value else "false", encoding="utf-8"
    )


def mounted_paths() -> dict[str, Path]:
    mounts: dict[str, Path] = {}
    for index, argument in enumerate(ARGS[:-1]):
        if argument != "--volume":
            continue
        parts = ARGS[index + 1].split(":")
        source, destination = parts[:2]
        mounts[destination] = (
            Path(source) if source.startswith("/") else VOLUME_DIR / source
        )
    return mounts


def host_path(container_path: str, mounts: dict[str, Path]) -> Path:
    requested = PurePosixPath(container_path)
    for destination, source in sorted(
        mounts.items(), key=lambda item: len(item[0]), reverse=True
    ):
        mount_path = PurePosixPath(destination)
        try:
            relative = requested.relative_to(mount_path)
        except ValueError:
            continue
        return source.joinpath(*relative.parts)
    finish(2, error=f"No fake mount covers {container_path}")


def archive_volume(mounts: dict[str, Path]) -> None:
    require_network_none()
    archive_argument = ARGS[ARGS.index("-czf") + 1]
    archive_path = host_path(archive_argument, mounts)
    data_path = mounts["/data"]
    if os.environ["FAKE_DOCKER_IMAGE_ID"] not in ARGS:
        finish(2, error="backup tar did not use the inspected local image ID")
    if should_fail("signal-term"):
        os.kill(os.getppid(), signal.SIGTERM)
        time.sleep(0.1)
        finish(143, error="injected TERM during tar")
    if should_fail("tar"):
        finish(1, error="injected tar failure")
    time.sleep(float(os.environ.get("FAKE_DOCKER_DELAY_TAR", "0")))
    exclude_cache = "--exclude=./cache" in ARGS
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(data_path.iterdir()):
            if child.name == "cache" and exclude_cache:
                continue
            archive.add(child, arcname=child.name)


def extract_archive(mounts: dict[str, Path]) -> None:
    require_network_none()
    if os.environ["RESTORE_IMAGE"] not in ARGS:
        finish(2, error="restore tar did not use RESTORE_IMAGE")
    archive_argument = ARGS[ARGS.index("-xzf") + 1]
    archive_path = host_path(archive_argument, mounts)
    data_path = mounts["/data"]
    if should_fail("extract"):
        finish(1, error="injected restore extraction failure")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                finish(2, error="unsafe test archive member")
            destination = data_path.joinpath(*member_path.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                finish(2, error="unsupported test archive member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                finish(2, error="unreadable test archive member")
            with source, destination.open("wb") as destination_file:
                shutil.copyfileobj(source, destination_file)


def checksum(mounts: dict[str, Path]) -> None:
    require_network_none()
    if should_fail("checksum"):
        finish(1, error="injected checksum failure")
    if should_fail("malformed-checksum"):
        finish(0, output="not-a-sha256  requested-file")
    requested_path = ARGS[-1]
    expected_image = (
        os.environ["FAKE_DOCKER_IMAGE_ID"]
        if requested_path.endswith(".partial")
        else os.environ["RESTORE_IMAGE"]
    )
    if expected_image not in ARGS:
        finish(2, error="checksum did not use the expected image")
    file_path = host_path(requested_path, mounts)
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    finish(0, output=f"{digest}  {requested_path}")


def validate_layout(mounts: dict[str, Path]) -> None:
    require_network_none()
    if os.environ["RESTORE_IMAGE"] not in ARGS:
        finish(2, error="validation did not use RESTORE_IMAGE")
    script = ARGS[-1]
    requirements = re.findall(r"test (-[sd]) (/data/[^\s]+)", script)
    if not requirements:
        finish(2, error="documented validation supplied no requirements")
    for predicate, container_path in requirements:
        path = host_path(container_path, mounts)
        valid = path.is_dir() if predicate == "-d" else path.is_file() and path.stat().st_size > 0
        if not valid:
            finish(1, error=f"missing required restore path: {container_path}")


record()
if not ARGS:
    finish(2, error="missing fake docker command")

if ARGS[0] == "inspect":
    template = ARGS[ARGS.index("--format") + 1]
    if ".Config.Image" in template:
        finish(0, output=os.environ["FAKE_DOCKER_IMAGE_REF"])
    if ".Image" in template:
        finish(0, output=os.environ["FAKE_DOCKER_IMAGE_ID"])
    if ".Mounts" in template:
        finish(0, output=os.environ["FAKE_DOCKER_VOLUME"])
    if ".State.Running" in template:
        finish(0, output="true" if running() else "false")
    if ".State.Status" in template:
        finish(0, output="running" if running() else "exited")
    finish(2, error=f"unsupported inspect template: {template}")

if ARGS[:2] == ["image", "inspect"]:
    if ARGS[-1] != os.environ["FAKE_DOCKER_IMAGE_ID"]:
        finish(2, error="image metadata lookup did not use the local image ID")
    finish(0, output=os.environ["FAKE_DOCKER_IMAGE_DIGEST"])

if ARGS[0] == "create":
    name = ARGS[ARGS.index("--name") + 1]
    if name != "autogpt-backup-lock":
        finish(2, error=f"unsupported fake lock container: {name}")
    lock_path = STATE_DIR / "backup-lock"
    if lock_path.exists():
        finish(1, error="lock container already exists")
    if os.environ["FAKE_DOCKER_IMAGE_ID"] not in ARGS:
        finish(2, error="lock container did not use the inspected image ID")
    lock_path.write_text("locked", encoding="utf-8")
    finish(0, output="fake-lock-container-id")

if ARGS[0] == "rm" and ARGS[-1] == "autogpt-backup-lock":
    if "--volumes" not in ARGS:
        finish(2, error="lock cleanup did not remove anonymous volumes")
    lock_path = STATE_DIR / "backup-lock"
    if not lock_path.exists():
        finish(1, error="lock container does not exist")
    lock_path.unlink()
    finish(0, output="autogpt-backup-lock")

if ARGS[:2] == ["container", "inspect"]:
    exists = os.environ.get("FAKE_DOCKER_EXISTING_CONTAINER") == "true"
    finish(0 if exists else 1)

if ARGS[:2] == ["volume", "inspect"]:
    if "--format" in ARGS:
        finish(0, output=os.environ.get("FAKE_DOCKER_LABELS", "{}"))
    volume_name = ARGS[-1]
    finish(0 if (VOLUME_DIR / volume_name).is_dir() else 1)

if ARGS[:2] == ["volume", "create"]:
    volume_name = ARGS[-1]
    (VOLUME_DIR / volume_name).mkdir(parents=True)
    finish(0, output=volume_name)

if ARGS[:2] == ["volume", "rm"]:
    volume_name = ARGS[-1]
    volume_path = VOLUME_DIR / volume_name
    if not volume_path.is_dir():
        finish(1, error="volume does not exist")
    shutil.rmtree(volume_path)
    finish(0, output=volume_name)

if ARGS[0] == "stop":
    if should_fail("stop"):
        finish(1, error="injected stop failure")
    set_running(False)
    finish(0, output="autogpt")

if ARGS[0] == "start":
    if should_fail("start"):
        finish(1, error="injected start failure")
    set_running(True)
    finish(0, output="autogpt")

if ARGS[0] == "run":
    mounts = mounted_paths()
    entrypoint = ARGS[ARGS.index("--entrypoint") + 1] if "--entrypoint" in ARGS else ""
    if entrypoint == "tar" and "-czf" in ARGS:
        archive_volume(mounts)
        finish(0)
    if entrypoint == "tar" and "-xzf" in ARGS:
        extract_archive(mounts)
        finish(0)
    if entrypoint == "sha256sum":
        checksum(mounts)
    if entrypoint == "/bin/sh":
        validate_layout(mounts)
        finish(0)
    if "--detach" in ARGS and "--name" in ARGS:
        if os.environ["RESTORE_IMAGE"] not in ARGS:
            finish(1, error="restored launch did not use RESTORE_IMAGE")
        environment_file = Path(ARGS[ARGS.index("--env-file") + 1])
        if not environment_file.is_file():
            finish(1, error="launch environment file does not exist")
        if not mounts["/data"].is_dir():
            finish(1, error="launch restore volume does not exist")
        finish(0, output="fake-container-id")

finish(2, error=f"unsupported fake docker invocation: {ARGS}")
"""


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
        fake_docker.write_text(FAKE_DOCKER, encoding="utf-8")
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
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
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
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn(f"/data/{relative_path}", invalid.stderr)

    def test_backup_can_leave_the_container_stopped_for_upgrade(self) -> None:
        result = self._run(COLD_BACKUP_BLOCK, RESTART_AFTER_BACKUP="false")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self._is_running())
        self._backup_artifacts()
        self.assertFalse(any(command[0] == "start" for command in self._commands()))

    def test_backup_failures_clean_up_and_attempt_safe_recovery(self) -> None:
        for operation in ("stop", "tar", "checksum", "malformed-checksum", "start"):
            with self.subTest(operation=operation):
                self._reset_backup_state()
                result = self._run(COLD_BACKUP_BLOCK, FAKE_DOCKER_FAIL=operation)

                self.assertNotEqual(result.returncode, 0)
                self._assert_no_backup_artifacts()
                self.assertEqual(self._is_running(), operation != "start")
                if operation in {"tar", "checksum", "malformed-checksum", "start"}:
                    self.assertTrue(
                        any(command[0] == "start" for command in self._commands())
                    )

    def test_upgrade_mode_failure_restarts_and_cleans_up(self) -> None:
        for operation in ("tar", "checksum", "malformed-checksum"):
            with self.subTest(operation=operation):
                self._reset_backup_state()
                result = self._run(
                    COLD_BACKUP_BLOCK,
                    FAKE_DOCKER_FAIL=operation,
                    RESTART_AFTER_BACKUP="false",
                )

                self.assertNotEqual(result.returncode, 0)
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
        for relative_path, path_type in REQUIRED_RESTORE_PATHS:
            if path_type == "directory" and relative_path != "rabbitmq/mnesia":
                self.assertIn(f"/data/{relative_path}", entrypoint)
        supervisor = SUPERVISOR_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "RABBITMQ_MNESIA_BASE=/data/rabbitmq/mnesia",
            supervisor,
        )

    def test_restored_launch_supports_stock_macos_bash(self) -> None:
        stock_bash = Path("/bin/bash")
        if not stock_bash.is_file():
            self.skipTest("stock /bin/bash is unavailable")
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
        lock_path = self.state_dir / "backup-lock"
        deadline = time.monotonic() + 5
        while not lock_path.is_file() and first.poll() is None:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertTrue(lock_path.is_file())

        second = self._run(COLD_BACKUP_BLOCK, BACKUP_DIR="second-backups")
        first_stdout, first_stderr = first.communicate(timeout=10)

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
