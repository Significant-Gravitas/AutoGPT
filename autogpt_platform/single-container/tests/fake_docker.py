#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(os.environ["FAKE_DOCKER_ROOT"])
STATE_DIR = ROOT / "state"
VOLUME_DIR = ROOT / "volumes"
VOLUME_LABEL_DIR = STATE_DIR / "volume-labels"
LOG_PATH = ROOT / "commands.jsonl"
ARGS = sys.argv[1:]
RESTORE_OWNER_LABEL = "org.agpt.restore.owner"


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


def volume_label_path(volume_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", volume_name):
        finish(2, error=f"invalid fake volume name: {volume_name}")
    return VOLUME_LABEL_DIR / f"{volume_name}.json"


def read_volume_labels(volume_name: str) -> dict[str, str]:
    metadata_path = volume_label_path(volume_name)
    if not metadata_path.is_file():
        return {}
    labels = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        finish(2, error=f"invalid fake volume labels: {volume_name}")
    return {str(key): str(value) for key, value in labels.items()}


def write_volume_labels(volume_name: str, labels: dict[str, str]) -> None:
    VOLUME_LABEL_DIR.mkdir(exist_ok=True)
    metadata_path = volume_label_path(volume_name)
    temporary_path = metadata_path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(labels, sort_keys=True), encoding="utf-8")
    temporary_path.replace(metadata_path)


def requested_volume_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, argument in enumerate(ARGS[:-1]):
        if argument == "--label":
            label = ARGS[index + 1]
        elif argument.startswith("--label="):
            label = argument.removeprefix("--label=")
        else:
            continue
        key, separator, value = label.partition("=")
        if not separator or not key:
            finish(2, error=f"invalid fake volume label: {label}")
        labels[key] = value
    return labels


def require_network_none() -> None:
    if "--network" not in ARGS:
        finish(2, error="documented helper did not disable networking")
    if ARGS[ARGS.index("--network") + 1] != "none":
        finish(2, error="documented helper used an unexpected network mode")


def running() -> bool:
    return (STATE_DIR / "running").read_text(encoding="utf-8").strip() == "true"


def set_running(value: bool) -> None:
    (STATE_DIR / "running").write_text("true" if value else "false", encoding="utf-8")


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
    missing_requirements = []
    for predicate, container_path in requirements:
        path = host_path(container_path, mounts)
        valid = (
            path.is_dir()
            if predicate == "-d"
            else path.is_file() and path.stat().st_size > 0
        )
        if not valid:
            missing_requirements.append(container_path)
    data_path = mounts["/data"]
    try:
        relative_data_path = data_path.relative_to(ROOT)
    except ValueError:
        finish(2, error="fake validation data path escaped its root")
    if not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in relative_data_path.parts
    ):
        finish(2, error="fake validation data path is not shell-safe")
    shell_data_path = f"./{relative_data_path.as_posix()}"
    host_script = re.sub(
        r"(?<![A-Za-z0-9_./-])/data(?=/|[\s:;,)\"']|$)",
        shell_data_path,
        script,
    )
    validation = subprocess.run(
        ["/bin/sh", "-ceu", host_script],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    if validation.returncode == 0 and missing_requirements:
        finish(2, error="documented validation did not reject a missing path")
    if validation.returncode != 0:
        error_parts = [
            part.strip()
            for part in (validation.stderr, validation.stdout)
            if part.strip()
        ]
        error_parts.extend(
            f"missing required restore path: {container_path}"
            for container_path in missing_requirements
        )
        finish(
            validation.returncode,
            error="\n".join(error_parts) or "documented restore validation failed",
        )


if not ARGS:
    finish(2, error="missing fake docker command")
record()

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
    volume_name = ARGS[-1]
    volume_path = VOLUME_DIR / volume_name
    if not volume_path.is_dir():
        finish(1)
    if "--format" in ARGS:
        template = ARGS[ARGS.index("--format") + 1]
        metadata_path = volume_label_path(volume_name)
        if "json .Labels" in template:
            if (
                not metadata_path.is_file()
                and volume_name == os.environ["FAKE_DOCKER_VOLUME"]
            ):
                finish(0, output=os.environ.get("FAKE_DOCKER_LABELS", "{}"))
            finish(0, output=json.dumps(read_volume_labels(volume_name)))
        label_match = re.search(r'index \.Labels "([^"]+)"', template)
        if label_match:
            finish(
                0, output=read_volume_labels(volume_name).get(label_match.group(1), "")
            )
        finish(2, error=f"unsupported fake volume inspect template: {template}")
    finish(0)

if ARGS[:2] == ["volume", "create"]:
    volume_name = ARGS[-1]
    volume_path = VOLUME_DIR / volume_name
    race_owner = os.environ.get("FAKE_DOCKER_PRECREATE_VOLUME_OWNER")
    if race_owner and not volume_path.exists():
        try:
            volume_path.mkdir()
        except FileExistsError:
            pass
        else:
            write_volume_labels(volume_name, {RESTORE_OWNER_LABEL: race_owner})
            (volume_path / "other-owner.txt").write_text(
                "preserve this volume\n", encoding="utf-8"
            )
    try:
        volume_path.mkdir()
    except FileExistsError:
        pass
    else:
        write_volume_labels(volume_name, requested_volume_labels())
    finish(0, output=volume_name)

if ARGS[:2] == ["volume", "rm"]:
    volume_name = ARGS[-1]
    volume_path = VOLUME_DIR / volume_name
    if not volume_path.is_dir():
        finish(1, error="volume does not exist")
    shutil.rmtree(volume_path)
    volume_label_path(volume_name).unlink(missing_ok=True)
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
