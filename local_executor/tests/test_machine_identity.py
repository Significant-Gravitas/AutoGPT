"""The machine ID is the shim's own.

HELLO's `machine_id`, which every audit record and folder grant carries too,
was the operating system's own identifier as read (IOPlatformUUID,
/etc/machine-id, MachineGuid), or the hostname and a random UUID where none
could be read. It is derived now from that identifier and a random value the
shim keeps in its own folder, so that neither leaves the computer, and so
that two computers never share an ID, however alike they are.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import types
import uuid
from pathlib import Path
from typing import Any

import keyring.core
import pytest
from keyring.backends import fail
from pydantic import ValidationError

import autogpt_local_executor.cli as cli
import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor import machine_identity, platform_info, session_grants
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig, load_config
from autogpt_local_executor.daemon import DaemonPreflightError, ShimDaemon

KEY = b"a" * 32
SEED_A = bytes(range(32))
SEED_B = bytes(range(32, 64))
IDENTIFIER_1 = "0123456789abcdef0123456789abcdef"
IDENTIFIER_2 = "fedcba9876543210fedcba9876543210"

# Each operating system's identifier, as its source gives it.
MACOS_UUID = "5F2B8E0C-91D4-4A6B-B3C7-0E9A1D2F4C68"
LINUX_MACHINE_ID = "9c1e4b7a2d5f4038b6a1c3e5f7092b4d"
WINDOWS_GUID = "3d7a9f10-6c2e-4b85-a1f4-8e0b2c6d9a37"


# The shim's reader of the operating system's identifier, which reads once
# for a process. Kept by name: some tests put a stand-in in its place.
_READ = platform_info.os_machine_identifier


@pytest.fixture(autouse=True)
def _read_afresh():
    """Each test here reads the identifier afresh, and leaves the next test
    none of its own."""
    _READ.cache_clear()
    yield
    _READ.cache_clear()


@pytest.fixture(autouse=True)
def _a_short_wait_for_a_value_being_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that holds the start of a value is waited for, in case another
    start is still writing it. The tests that put one there mean it to stay
    that way, and the test of the wait sets its own."""
    monkeypatch.setattr(machine_identity, "_BEING_WRITTEN_SECONDS", 0.05, raising=False)


def _fallback_folder() -> Path:
    return Path(os.environ["AUTOGPT_SHIM_FALLBACK_DIR"])


def _spellings(identifier: str) -> set[str]:
    """`identifier` as it could be written: with and without its dashes, in
    either letter case."""
    bare = identifier.replace("-", "")
    dashed = f"{bare[0:8]}-{bare[8:12]}-{bare[12:16]}-{bare[16:20]}-{bare[20:32]}"
    return {text for form in (bare, dashed) for text in (form.lower(), form.upper())}


def _this_computer_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> str:
    """Have the shim read the identifier of a `source` computer, through
    that system's own way of reading it, and return what it read. The rest
    of the test runs as the OS it is on."""
    with monkeypatch.context() as during_the_read:
        if source == "macos":
            during_the_read.setattr(platform_info, "detect_platform", lambda: "darwin")
            during_the_read.setattr(
                platform_info.subprocess,
                "check_output",
                lambda *_args, **_kwargs: (
                    "+-o J316sAP  <class IOPlatformExpertDevice, id 0x100000243>\n"
                    "    {\n"
                    '      "IOPlatformSerialNumber" = "C02XXXXXXXXX"\n'
                    f'      "IOPlatformUUID" = "{MACOS_UUID}"\n'
                    "    }\n"
                ),
            )
            identifier = MACOS_UUID
        elif source == "linux":
            machine_id = tmp_path / "machine-id"
            machine_id.write_text(LINUX_MACHINE_ID + "\n", encoding="utf-8")
            during_the_read.setattr(platform_info, "detect_platform", lambda: "linux")
            during_the_read.setattr(
                platform_info, "_LINUX_MACHINE_ID_FILES", (str(machine_id),), raising=False
            )
            identifier = LINUX_MACHINE_ID
        elif source == "windows":
            during_the_read.setattr(platform_info, "detect_platform", lambda: "windows")
            during_the_read.setitem(sys.modules, "winreg", _registry({"MachineGuid": WINDOWS_GUID}))
            identifier = WINDOWS_GUID
        else:
            assert source == "none"
            during_the_read.setattr(platform_info, "detect_platform", lambda: "linux")
            during_the_read.setattr(platform_info, "_LINUX_MACHINE_ID_FILES", (), raising=False)
            identifier = ""
        read = platform_info.os_machine_identifier()
    assert (read or "") == identifier
    return identifier


def _registry(values: dict[str, str]) -> types.ModuleType:
    """A stand-in for winreg holding HKLM\\SOFTWARE\\Microsoft\\Cryptography."""
    module = types.ModuleType("winreg")

    class Key:
        def __enter__(self) -> Key:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def open_key(root: object, path: str) -> Key:
        assert root == "HKLM" and path == r"SOFTWARE\Microsoft\Cryptography"
        return Key()

    def query_value(_key: Key, name: str) -> tuple[str, int]:
        if name not in values:
            raise FileNotFoundError(errno.ENOENT, "no such value")
        return values[name], 1

    module.HKEY_LOCAL_MACHINE = "HKLM"  # type: ignore[attr-defined]
    module.OpenKey = open_key  # type: ignore[attr-defined]
    module.QueryValueEx = query_value  # type: ignore[attr-defined]
    return module


def _config(tmp_path: Path, **overrides: Any) -> ShimConfig:
    return ShimConfig(
        **{
            "allowed_root": tmp_path / "workspace",
            "audit_log_path": tmp_path / "audit.log",
            "platform_url": "http://localhost:9999",
            **overrides,
        }
    )


def _daemon(config: ShimConfig) -> ShimDaemon:
    audit = AuditWriter(path=config.audit_log_path, audit_key=KEY)
    return ShimDaemon(config, token_store=object(), audit=audit)


# ── The derivation ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("identifier", "seed", "machine_id"),
    [
        (IDENTIFIER_1, SEED_A, "5d2cc977-5546-4d98-83f8-1e1a7062e46c"),
        (IDENTIFIER_1, SEED_B, "75ae8304-8ff4-4b0f-a9df-b9b4eddc5071"),
        (IDENTIFIER_2, SEED_A, "fb810855-005a-4e1c-9870-e507a15454d1"),
        (None, SEED_A, "01c29023-9574-4aa9-9eb6-7a6b95176266"),
        (None, SEED_B, "55956252-283c-4597-bbc5-8e0b151a12f7"),
    ],
)
def test_the_derivation_is_pinned(identifier: str | None, seed: bytes, machine_id: str) -> None:
    """These can't change: the platform knows a computer by its ID, and a
    chat is bound to it, so another derivation would cut every chat off from
    its computer. Both inputs are in each vector."""
    assert machine_identity.derive_machine_id(identifier, seed) == machine_id


def test_the_id_is_written_as_a_uuid() -> None:
    machine_id = machine_identity.derive_machine_id(IDENTIFIER_1, SEED_A)

    assert str(uuid.UUID(machine_id)) == machine_id
    assert uuid.UUID(machine_id).version == 4


@pytest.mark.parametrize(
    "spelling",
    [
        "0123456789ABCDEF0123456789ABCDEF",
        "01234567-89ab-cdef-0123-456789abcdef",
        "01234567-89AB-CDEF-0123-456789ABCDEF",
        "{01234567-89AB-CDEF-0123-456789ABCDEF}",
        "  0123456789abcdef0123456789abcdef\n",
        "\t01234567-89ab-cdef-0123-456789abcdef \r\n",
    ],
)
def test_two_spellings_of_one_identifier_give_one_id(spelling: str) -> None:
    """One computer can't get two IDs from how its identifier is written:
    letter case, dashes, braces and the space around it don't count."""
    assert machine_identity.derive_machine_id(
        spelling, SEED_A
    ) == machine_identity.derive_machine_id(IDENTIFIER_1, SEED_A)


@pytest.mark.parametrize(
    "not_an_identifier",
    [
        "",
        "   ",
        "uninitialized",
        "0123",
        IDENTIFIER_1 + "0",
        "g" * 32,
        "0123456789abcdef-0123456789abcde",
    ],
)
def test_what_is_no_identifier_counts_as_none(not_an_identifier: str) -> None:
    """Each source holds 128 bits in hex. Anything else (systemd's
    `uninitialized` before a first boot is done, an empty file) is no
    identifier, and the ID comes from the stored value alone."""
    assert machine_identity.canonical_os_identifier(not_an_identifier) is None
    assert machine_identity.derive_machine_id(
        not_an_identifier, SEED_A
    ) == machine_identity.derive_machine_id(None, SEED_A)


def test_one_os_identifier_with_two_stored_values_gives_two_ids() -> None:
    """Two computers made from one image hold the same identifier."""
    seeds = [os.urandom(32) for _ in range(2)]

    first, second = (machine_identity.derive_machine_id(IDENTIFIER_1, seed) for seed in seeds)

    assert first != second


def test_one_stored_value_with_two_os_identifiers_gives_two_ids() -> None:
    """Two computers that share a home folder share the stored value."""
    seed = os.urandom(32)

    first, second = (
        machine_identity.derive_machine_id(identifier, seed)
        for identifier in (IDENTIFIER_1, IDENTIFIER_2)
    )

    assert first != second


def test_no_identifier_with_two_stored_values_gives_two_ids() -> None:
    seeds = [os.urandom(32) for _ in range(2)]

    first, second = (machine_identity.derive_machine_id(None, seed) for seed in seeds)

    assert first != second
    assert machine_identity.derive_machine_id(None, seeds[0]) != machine_identity.derive_machine_id(
        IDENTIFIER_1, seeds[0]
    )


def test_the_id_holds_neither_input() -> None:
    machine_id = machine_identity.derive_machine_id(IDENTIFIER_1, SEED_A)

    assert machine_id.replace("-", "") != IDENTIFIER_1
    assert IDENTIFIER_1[:12] not in machine_id.replace("-", "")
    assert SEED_A.hex()[:12] not in machine_id.replace("-", "")


# ── The operating system's identifier ────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("macos", "5f2b8e0c91d44a6bb3c70e9a1d2f4c68"),
        ("linux", LINUX_MACHINE_ID),
        ("windows", "3d7a9f106c2e4b85a1f48e0b2c6d9a37"),
        ("none", None),
    ],
)
def test_each_systems_identifier_is_read_and_put_in_one_form(
    monkeypatch, tmp_path: Path, source: str, canonical: str | None
) -> None:
    identifier = _this_computer_is(monkeypatch, tmp_path, source)

    assert machine_identity.canonical_os_identifier(identifier) == canonical


def test_linux_falls_back_to_dbus_machine_id(monkeypatch, tmp_path: Path) -> None:
    empty, dbus = tmp_path / "machine-id", tmp_path / "dbus-machine-id"
    empty.write_text("\n", encoding="utf-8")
    dbus.write_text(LINUX_MACHINE_ID + "\n", encoding="utf-8")
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(
        platform_info,
        "_LINUX_MACHINE_ID_FILES",
        (str(tmp_path / "missing"), str(empty), str(dbus)),
    )

    assert platform_info.os_machine_identifier() == LINUX_MACHINE_ID


def test_a_wsl2_distribution_reads_its_own_identifier_not_its_hosts(
    monkeypatch, tmp_path: Path
) -> None:
    """A Windows host and a WSL2 distribution on it stay two computers: the
    distribution has a machine-id of its own, and the registry isn't read."""
    machine_id = tmp_path / "machine-id"
    machine_id.write_text(LINUX_MACHINE_ID + "\n", encoding="utf-8")
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "wsl2")
    monkeypatch.setattr(platform_info, "_LINUX_MACHINE_ID_FILES", (str(machine_id),))
    monkeypatch.setitem(sys.modules, "winreg", _registry({"MachineGuid": WINDOWS_GUID}))

    assert platform_info.os_machine_identifier() == LINUX_MACHINE_ID


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(errno.ENOENT, "ioreg"),
        subprocess.TimeoutExpired("ioreg", 5),
        subprocess.CalledProcessError(1, "ioreg"),
    ],
)
def test_a_macs_identifier_that_could_not_be_read_is_not_taken_for_none(
    monkeypatch, failure: Exception
) -> None:
    """Every Mac has one. `ioreg` failing, or not answering in its five
    seconds at a busy login, is that one read failing. Taken for a computer
    with no identifier, the start went on under another machine ID."""

    def fails(*_args: object, **_kwargs: object) -> str:
        raise failure

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(platform_info.subprocess, "check_output", fails)

    with pytest.raises(platform_info.MachineIdentifierUnreadable):
        platform_info.os_machine_identifier()


def test_a_mac_whose_answer_holds_no_identifier_has_none(monkeypatch) -> None:
    """`ioreg` answered, so nothing failed: the same answer at every start."""
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(
        platform_info.subprocess, "check_output", lambda *_args, **_kwargs: "+-o Root\n  {\n  }\n"
    )

    assert platform_info.os_machine_identifier() is None


def test_a_linux_identifier_file_that_may_not_be_read_is_still_none(
    monkeypatch, tmp_path: Path
) -> None:
    """A file the account may not read stays that way from one start to the
    next (a sandbox, a hardened /etc), so the ID from the stored value alone
    is the same each time. Stopping there would keep the shim from ever
    starting."""

    def no_access(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(
        platform_info, "_LINUX_MACHINE_ID_FILES", (str(tmp_path / "machine-id"),), raising=False
    )
    monkeypatch.setattr("builtins.open", no_access)

    assert platform_info.os_machine_identifier() is None


@pytest.mark.parametrize(
    "failure",
    [OSError(errno.EIO, "Input/output error"), OSError(errno.ESTALE, "Stale file handle")],
    ids=["io-error", "stale-handle"],
)
def test_a_linux_identifier_whose_read_fails_this_once_is_not_taken_for_none(
    monkeypatch, tmp_path: Path, failure: OSError
) -> None:
    """A file that isn't there, or isn't this account's to read, stays that
    way. A read that fails any other way may work at the next start, and
    the ID without it would be another one for this start."""

    def fails(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(
        platform_info, "_LINUX_MACHINE_ID_FILES", (str(tmp_path / "machine-id"),), raising=False
    )
    monkeypatch.setattr("builtins.open", fails)

    with pytest.raises(platform_info.MachineIdentifierUnreadable):
        platform_info.os_machine_identifier()


def test_a_registry_read_that_fails_this_once_is_not_taken_for_none(monkeypatch) -> None:
    registry = _registry({"MachineGuid": WINDOWS_GUID})

    def fails(_key: object, _name: str) -> tuple[str, int]:
        raise OSError(errno.EIO, "The registry could not be read")

    registry.QueryValueEx = fails  # type: ignore[attr-defined]
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "windows")
    monkeypatch.setitem(sys.modules, "winreg", registry)

    with pytest.raises(platform_info.MachineIdentifierUnreadable):
        platform_info.os_machine_identifier()


def test_a_registry_the_account_may_not_read_is_none(monkeypatch) -> None:
    registry = _registry({"MachineGuid": WINDOWS_GUID})

    def refused(_key: object, _name: str) -> tuple[str, int]:
        raise PermissionError(errno.EACCES, "Access is denied")

    registry.QueryValueEx = refused  # type: ignore[attr-defined]
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "windows")
    monkeypatch.setitem(sys.modules, "winreg", registry)

    assert platform_info.os_machine_identifier() is None


def test_a_registry_without_the_value_is_none(monkeypatch) -> None:
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "windows")
    monkeypatch.setitem(sys.modules, "winreg", _registry({}))

    assert platform_info.os_machine_identifier() is None


# ── The stored value ─────────────────────────────────────────────────────────


def test_the_stored_value_is_kept_in_the_shims_own_folder(monkeypatch, tmp_path: Path) -> None:
    """Beside the audit key's file and the locks every shim of the account
    takes: AUTOGPT_SHIM_FALLBACK_DIR, by default ~/.autogpt-local-executor."""
    assert machine_identity.seed_path() == _fallback_folder() / "machine-id-seed"

    monkeypatch.delenv("AUTOGPT_SHIM_FALLBACK_DIR")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    assert machine_identity.seed_path() == (
        tmp_path / "home" / ".autogpt-local-executor" / "machine-id-seed"
    )


def test_the_stored_value_is_made_once_and_read_after(tmp_path: Path) -> None:
    path = machine_identity.seed_path()
    assert not path.exists()

    first = machine_identity.stored_seed()

    written = path.read_bytes()
    assert written == first.hex().encode("ascii") + b"\n"
    assert len(first) == 32
    assert machine_identity.stored_seed() == first
    assert path.read_bytes() == written
    # Nothing else is left beside it.
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


def test_the_stored_value_is_made_in_a_folder_that_is_not_there_yet(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path / "not" / "there" / "yet"))

    seed = machine_identity.stored_seed()

    assert machine_identity.seed_path().read_bytes() == seed.hex().encode("ascii") + b"\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_stored_values_file_is_its_owners_alone() -> None:
    """Whatever the umask: the file is made with its mode, not given it
    after, so it is never open to another account, even for a moment."""
    before = os.umask(0)
    try:
        machine_identity.stored_seed()
    finally:
        os.umask(before)

    assert stat.S_IMODE(machine_identity.seed_path().stat().st_mode) == 0o600


def test_two_starts_at_once_end_with_one_stored_value() -> None:
    """Each start that finds no value makes one. Only one of them is kept,
    and every start uses the one that was."""
    starts = 16
    together = threading.Barrier(starts)
    seeds: list[bytes] = []
    failures: list[BaseException] = []

    def start() -> None:
        try:
            together.wait(timeout=30)
            seeds.append(machine_identity.stored_seed())
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=start) for _ in range(starts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert failures == []
    assert len(seeds) == starts
    assert len(set(seeds)) == 1
    path = machine_identity.seed_path()
    assert path.read_bytes() == seeds[0].hex().encode("ascii") + b"\n"
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


def test_a_start_that_loses_to_another_takes_the_others_value(monkeypatch) -> None:
    """Another start's value appears between this start's look for one and
    its putting its own in place: this start's is dropped, never put over
    the other's, and it goes on with the one that is there."""
    path = machine_identity.seed_path()
    other = SEED_B.hex().encode("ascii") + b"\n"
    place = machine_identity._place

    def another_start_got_there_first(temp: Path, final: Path) -> None:
        final.write_bytes(other)
        place(temp, final)

    monkeypatch.setattr(machine_identity, "_place", another_start_got_there_first)

    assert machine_identity.stored_seed() == SEED_B
    assert path.read_bytes() == other
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\n",
        SEED_A.hex().encode("ascii")[:40],
        SEED_A.hex().encode("ascii")[:63] + b"\n",
        SEED_A.hex().encode("ascii") + b"00\n",
        b"z" * 64 + b"\n",
        b"\x00" * 65,
        SEED_A.hex().upper().encode("ascii") + b"\n",
    ],
    ids=["empty", "newline", "cut-short", "one-short", "too-long", "not-hex", "zeros", "upper"],
)
def test_a_file_that_is_not_a_whole_value_is_never_read_as_one(content: bytes) -> None:
    """A value is written whole to a file of its own and only then given its
    name, so a file under that name that isn't one wasn't written by the
    shim. It isn't used, and it isn't replaced: that would give the computer
    a new ID without a word."""
    path = machine_identity.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    with pytest.raises(machine_identity.MachineIdError) as refused:
        machine_identity.stored_seed()

    assert str(path) in str(refused.value)
    assert refused.value.error is None
    assert path.read_bytes() == content


def test_a_value_without_its_newline_is_whole() -> None:
    path = machine_identity.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SEED_A.hex().encode("ascii"))

    assert machine_identity.stored_seed() == SEED_A


def test_a_folder_that_cannot_hold_the_value_is_an_error_not_a_new_value(
    monkeypatch, tmp_path: Path
) -> None:
    """With nowhere to keep it, a value made now would be another at the
    next start, and the platform would see a new computer each time."""
    in_the_way = tmp_path / "a-file"
    in_the_way.write_text("", encoding="utf-8")
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(in_the_way))

    with pytest.raises(machine_identity.MachineIdError) as refused:
        machine_identity.stored_seed()

    assert isinstance(refused.value.error, OSError)
    assert str(machine_identity.seed_path()) in str(refused.value)


def test_a_value_that_cannot_be_put_in_place_leaves_nothing_behind(monkeypatch) -> None:
    def full(_temp: Path, _final: Path) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(machine_identity, "_place", full)

    with pytest.raises(machine_identity.MachineIdError) as refused:
        machine_identity.stored_seed()

    assert refused.value.error is not None and refused.value.error.errno == errno.ENOSPC
    assert list(machine_identity.seed_path().parent.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="hard links to a file of one's own are POSIX's way")
def test_without_hard_links_the_value_is_still_put_in_place_once(monkeypatch) -> None:
    """A filesystem without hard links (FAT, some network shares): the value
    gets its name by a rename that refuses to replace."""

    def no_links(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(machine_identity.os, "link", no_links)

    first = machine_identity.stored_seed()
    path = machine_identity.seed_path()
    assert path.read_bytes() == first.hex().encode("ascii") + b"\n"
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]

    path.unlink()
    other = SEED_B.hex().encode("ascii") + b"\n"
    place = machine_identity._place

    def another_start_got_there_first(temp: Path, final: Path) -> None:
        final.write_bytes(other)
        place(temp, final)

    monkeypatch.setattr(machine_identity, "_place", another_start_got_there_first)
    assert machine_identity.stored_seed() == SEED_B
    assert path.read_bytes() == other


def _neither_links_nor_a_one_step_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem with no hard links, on a system whose rename has no way
    to refuse to replace: exFAT or a FAT stick, an SMB share without links,
    sshfs."""
    from autogpt_local_executor import fsops

    def no_links(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(machine_identity.os, "link", no_links)
    monkeypatch.setattr(fsops, "_native_no_replace_rename", lambda _src, _dst: False)


@pytest.mark.skipif(os.name == "nt", reason="Windows' own rename never replaces")
def test_without_links_or_a_one_step_rename_another_starts_value_is_not_replaced(
    monkeypatch,
) -> None:
    """There the value got its name by a look for the name and then a
    rename, two steps. Another start's value that took the name between
    them was replaced, after that start had read it: the two went on under
    two machine IDs, and the first one's changed at its next start."""
    from autogpt_local_executor import fsops

    _neither_links_nor_a_one_step_rename(monkeypatch)
    path = machine_identity.seed_path()
    other = SEED_B.hex().encode("ascii") + b"\n"
    arrived: list[bool] = []

    def another_start_takes_the_name() -> None:
        if not arrived:
            arrived.append(True)
            path.write_bytes(other)

    # Wherever this start's placing of its own value looks or acts on the
    # name, the other start's value has just got there.
    lexists, open_ = fsops.os.path.lexists, machine_identity.os.open

    def looked_just_before(name: object) -> bool:
        found = lexists(name)
        if Path(os.fspath(name)) == path:  # type: ignore[arg-type]
            another_start_takes_the_name()
        return found

    def opened_just_after(name: object, flags: int, *rest: object) -> int:
        if Path(os.fspath(name)) == path and flags & os.O_EXCL:  # type: ignore[arg-type]
            another_start_takes_the_name()
        return open_(name, flags, *rest)  # type: ignore[arg-type]

    monkeypatch.setattr(fsops.os.path, "lexists", looked_just_before)
    monkeypatch.setattr(machine_identity.os, "open", opened_just_after)

    seed = machine_identity.stored_seed()

    assert arrived == [True]
    assert path.read_bytes() == other
    assert seed == SEED_B
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


@pytest.mark.skipif(os.name == "nt", reason="Windows' own rename never replaces")
def test_without_links_or_a_one_step_rename_two_starts_at_once_end_with_one_value(
    monkeypatch,
) -> None:
    _neither_links_nor_a_one_step_rename(monkeypatch)
    starts = 16
    together = threading.Barrier(starts)
    seeds: list[bytes] = []
    failures: list[BaseException] = []

    def start() -> None:
        try:
            together.wait(timeout=30)
            seeds.append(machine_identity.stored_seed())
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=start) for _ in range(starts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert failures == []
    assert len(seeds) == starts
    assert len(set(seeds)) == 1
    path = machine_identity.seed_path()
    assert path.read_bytes() == seeds[0].hex().encode("ascii") + b"\n"
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


@pytest.mark.skipif(os.name == "nt", reason="Windows' own rename never replaces")
def test_without_links_or_a_one_step_rename_the_value_is_still_made_and_kept(monkeypatch) -> None:
    _neither_links_nor_a_one_step_rename(monkeypatch)

    first = machine_identity.stored_seed()

    path = machine_identity.seed_path()
    assert path.read_bytes() == first.hex().encode("ascii") + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert machine_identity.stored_seed() == first
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


@pytest.mark.skipif(os.name == "nt", reason="Windows' own rename never replaces")
def test_a_whole_value_another_start_may_have_read_is_not_taken_away_again(monkeypatch) -> None:
    """Where the name is made first, the value is under it from the moment
    it is written, before it is synced. A sync that then failed had the
    file removed again. Another start that had read it by then went on
    under an ID no later start would have: the next one made a new value."""
    _neither_links_nor_a_one_step_rename(monkeypatch)
    path = machine_identity.seed_path()
    fsync = machine_identity.os.fsync
    synced = 0
    read_by_another_start: list[bytes] = []

    def fails_for_the_value_under_its_name(descriptor: int) -> None:
        nonlocal synced
        synced += 1
        if synced == 1:
            # The file the value is first written to, of this start's own.
            fsync(descriptor)
            return
        read_by_another_start.append(machine_identity.stored_seed())
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(machine_identity.os, "fsync", fails_for_the_value_under_its_name)
    with pytest.raises(machine_identity.MachineIdError):
        machine_identity.stored_seed()
    monkeypatch.setattr(machine_identity.os, "fsync", fsync)

    assert len(read_by_another_start) == 1
    # The next start has the value the other one read.
    assert machine_identity.stored_seed() == read_by_another_start[0]
    assert [entry.name for entry in path.parent.iterdir()] == ["machine-id-seed"]


@pytest.mark.skipif(os.name == "nt", reason="Windows' own rename never replaces")
def test_a_value_that_was_not_written_whole_leaves_no_file(monkeypatch) -> None:
    _neither_links_nor_a_one_step_rename(monkeypatch)
    path = machine_identity.seed_path()
    fdopen = machine_identity.os.fdopen
    opened = 0

    class CutShort:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> CutShort:
            return self

        def __exit__(self, *_exc: object) -> None:
            self._handle.close()

        def write(self, data: bytes) -> int:
            self._handle.write(data[:10])
            raise OSError(errno.ENOSPC, "No space left on device")

    def second_file_fills_the_disk(descriptor: int, mode: str) -> Any:
        nonlocal opened
        opened += 1
        handle = fdopen(descriptor, mode)
        return handle if opened == 1 else CutShort(handle)

    monkeypatch.setattr(machine_identity.os, "fdopen", second_file_fills_the_disk)

    with pytest.raises(machine_identity.MachineIdError):
        machine_identity.stored_seed()

    assert list(path.parent.iterdir()) == []


def test_a_value_another_start_is_still_writing_is_waited_for() -> None:
    """Where the name is made first and the value written into it, another
    start can find the name with less than a value under it. It waits for
    the rest, a short while, and doesn't call the file broken."""
    path = machine_identity.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = SEED_A.hex().encode("ascii") + b"\n"
    path.write_bytes(value[:20])
    seen_cut_short = threading.Event()
    read = machine_identity._read
    reads: list[str] = []

    def read_and_say(name: Path, **how: Any) -> bytes | None:
        try:
            found = read(name, **how)
        except machine_identity._StillBeingWritten:
            # Only now does the other start go on: this one has seen the
            # value cut short, and has to come back for the rest.
            reads.append("cut short")
            seen_cut_short.set()
            raise
        reads.append("whole")
        return found

    def the_other_start_finishes() -> None:
        assert seen_cut_short.wait(30)
        path.write_bytes(value)

    writer = threading.Thread(target=the_other_start_finishes)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(machine_identity, "_read", read_and_say)
        patched.setattr(machine_identity, "_BEING_WRITTEN_SECONDS", 60.0)
        writer.start()
        try:
            assert machine_identity.stored_seed() == SEED_A
        finally:
            writer.join(30)
    assert reads[0] == "cut short" and reads[-1] == "whole"


def test_a_value_that_stays_cut_short_is_still_an_error(monkeypatch) -> None:
    path = machine_identity.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SEED_A.hex().encode("ascii")[:20])
    monkeypatch.setattr(machine_identity, "_BEING_WRITTEN_SECONDS", 0.2)

    with pytest.raises(machine_identity.MachineIdError, match="doesn't hold the value"):
        machine_identity.stored_seed()

    assert path.read_bytes() == SEED_A.hex().encode("ascii")[:20]


# ── What the daemon sends, records and prints ────────────────────────────────


def _everything_start_puts_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[str, str]:
    """A start's machine ID, and all it sends, records and prints as text:
    its HELLO, its audit log, a folder grant and what it wrote to the
    terminal."""
    import asyncio

    config = _config(tmp_path)
    daemon = _daemon(config)

    async def hello_and_start_record() -> str:
        hello = await daemon._build_hello()
        await daemon._audit_shim_start()
        return hello.model_dump_json()

    hello = asyncio.run(hello_and_start_record())
    folder = tmp_path / "chosen"
    folder.mkdir()
    grant = daemon._root_grants.issue("chat-1", folder, 1).root_grant
    claims = session_grants._b64decode(grant.split(".")[1])
    if isinstance(claims, bytes):
        claims = claims.decode("utf-8")
    log = config.audit_log_path.read_text(encoding="utf-8")

    class Runs:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            return None

    monkeypatch.setattr(daemon_module, "ShimDaemon", Runs)
    capsys.readouterr()
    assert asyncio.run(cli._cmd_start(_config(tmp_path))) == 0
    printed = capsys.readouterr()
    machine_id = json.loads(hello)["payload"]["machine_id"]
    assert json.loads(claims)["machine_id"] == machine_id
    assert [json.loads(line)["machine_id"] for line in log.splitlines()] == [machine_id]
    assert f"machine_id={machine_id})" in printed.out
    return machine_id, "\n".join([hello, claims, grant, log, printed.out, printed.err])


@pytest.mark.parametrize("source", ["macos", "linux", "windows"])
def test_the_systems_identifier_is_not_the_machine_id_and_leaves_in_nothing(
    monkeypatch, tmp_path: Path, capsys, source: str
) -> None:
    """It went out as read: in HELLO, in every audit record, in every folder
    grant and on the line `start` prints."""
    identifier = _this_computer_is(monkeypatch, tmp_path, source)

    machine_id, put_out = _everything_start_puts_out(tmp_path, monkeypatch, capsys)

    assert machine_id.lower() not in _spellings(identifier)
    assert str(uuid.UUID(machine_id)) == machine_id
    for spelling in _spellings(identifier):
        assert spelling not in put_out
    # Nor the value the shim keeps to derive it from.
    assert machine_identity.seed_path().read_text(encoding="ascii").strip() not in put_out
    canonical = machine_identity.canonical_os_identifier(identifier)
    assert machine_id == machine_identity.derive_machine_id(
        canonical, machine_identity.stored_seed()
    )


def test_with_no_identifier_to_read_the_id_has_no_hostname_in_it(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """It was `<hostname>-<random UUID>`, kept in a file in the home folder.
    The ID comes from the stored value alone now, and the hostname is never
    asked for to make it."""

    def asked() -> str:
        raise AssertionError("the hostname was asked for to make the machine ID")

    # The display name's default keeps the real lookup it was defined with.
    monkeypatch.setattr(socket, "gethostname", asked)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()
    _this_computer_is(monkeypatch, tmp_path, "none")

    machine_id, _put_out = _everything_start_puts_out(tmp_path, monkeypatch, capsys)

    # A UUID and nothing more: no room for a name in front of it.
    assert str(uuid.UUID(machine_id)) == machine_id
    assert machine_id == machine_identity.derive_machine_id(None, machine_identity.stored_seed())
    # The file the old ID was kept in is neither made nor read.
    assert list((tmp_path / "home").iterdir()) == []


def test_the_old_ids_file_is_not_read(monkeypatch, tmp_path: Path) -> None:
    """No compatibility: an ID kept by an earlier build isn't taken over."""
    home = tmp_path / "home"
    home.mkdir()
    old = home / ".autogpt-local-executor-machine-id"
    old.write_text("alices-laptop-3b0ad8f6-5c6f-4d6c-9c53-0d1f2a3b4c5d", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    _this_computer_is(monkeypatch, tmp_path, "none")

    machine_id = _daemon(_config(tmp_path)).machine_id

    assert "alices-laptop" not in machine_id
    assert old.read_text(encoding="utf-8") == "alices-laptop-3b0ad8f6-5c6f-4d6c-9c53-0d1f2a3b4c5d"


@pytest.mark.parametrize("source", ["macos", "linux", "windows", "none"])
def test_a_computer_keeps_its_id_from_one_start_to_the_next(
    monkeypatch, tmp_path: Path, source: str
) -> None:
    _this_computer_is(monkeypatch, tmp_path, source)
    first = _daemon(_config(tmp_path, audit_log_path=tmp_path / "first.log")).machine_id

    # Another process: the identifier is read again, the value is on disk.
    _READ.cache_clear()
    _this_computer_is(monkeypatch, tmp_path, source)
    second = _daemon(_config(tmp_path, audit_log_path=tmp_path / "second.log")).machine_id

    assert first == second


def test_a_start_that_cannot_read_the_identifier_stops_rather_than_be_another_computer(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The ID is derived from the identifier and the stored value. With the
    identifier unread at one start (`ioreg` not answering in time), the ID
    came from the stored value alone: another ID, so for that start the
    platform saw a new computer, and no chat found its own."""
    import asyncio

    _this_computer_is(monkeypatch, tmp_path, "macos")
    first = _daemon(_config(tmp_path, audit_log_path=tmp_path / "first.log")).machine_id

    # The next start: the read doesn't come back in time.
    _READ.cache_clear()

    def no_answer(*_args: object, **_kwargs: object) -> str:
        raise subprocess.TimeoutExpired("ioreg", 5)

    with monkeypatch.context() as this_start:
        this_start.setattr(platform_info, "detect_platform", lambda: "darwin")
        this_start.setattr(platform_info.subprocess, "check_output", no_answer)
        config = _config(tmp_path, audit_log_path=tmp_path / "second.log")
        with pytest.raises(DaemonPreflightError) as refused:
            _daemon(config)

        assert config.machine_id is None
        # To be started again, which reads it again: not for good.
        assert refused.value.transient is True
        assert "identifier" in str(refused.value)
        assert "AUTOGPT_SHIM_FALLBACK_DIR" not in str(refused.value)
        assert "machine_id" in str(refused.value)
        again = _config(tmp_path, audit_log_path=tmp_path / "b.log")
        assert asyncio.run(cli._cmd_start(again)) == 75
        printed = capsys.readouterr()
        assert "Preflight failed" in printed.err
        assert "machine_id=" not in printed.out

    # The start after that, with the identifier read: the same computer.
    _READ.cache_clear()
    _this_computer_is(monkeypatch, tmp_path, "macos")
    third = _daemon(_config(tmp_path, audit_log_path=tmp_path / "third.log")).machine_id
    assert third == first


def test_a_first_start_that_cannot_read_the_identifier_keeps_no_value(
    monkeypatch, tmp_path: Path
) -> None:
    def no_answer(*_args: object, **_kwargs: object) -> str:
        raise subprocess.TimeoutExpired("ioreg", 5)

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(platform_info.subprocess, "check_output", no_answer)

    with pytest.raises(machine_identity.MachineIdError):
        machine_identity.own_machine_id()

    assert not machine_identity.seed_path().exists()


def test_deleting_the_shims_state_gives_the_computer_a_new_id(monkeypatch, tmp_path: Path) -> None:
    _this_computer_is(monkeypatch, tmp_path, "linux")
    first = _daemon(_config(tmp_path, audit_log_path=tmp_path / "first.log")).machine_id

    machine_identity.seed_path().unlink()
    second = _daemon(_config(tmp_path, audit_log_path=tmp_path / "second.log")).machine_id

    assert first != second


def test_two_accounts_on_one_computer_get_two_ids(monkeypatch, tmp_path: Path) -> None:
    """Each account's shim keeps its own value in its own home folder."""
    _this_computer_is(monkeypatch, tmp_path, "linux")
    ids = []
    for account in ("alice", "bob"):
        monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path / account))
        ids.append(
            _daemon(_config(tmp_path, audit_log_path=tmp_path / f"{account}.log")).machine_id
        )

    assert ids[0] != ids[1]


def test_two_computers_that_share_a_home_folder_get_two_ids(monkeypatch, tmp_path: Path) -> None:
    """One stored value, read by both: their own identifiers tell them apart."""
    _this_computer_is(monkeypatch, tmp_path, "linux")
    first = _daemon(_config(tmp_path, audit_log_path=tmp_path / "first.log")).machine_id

    _READ.cache_clear()
    _this_computer_is(monkeypatch, tmp_path, "macos")
    second = _daemon(_config(tmp_path, audit_log_path=tmp_path / "second.log")).machine_id

    assert first != second


def test_a_machine_id_the_person_set_is_sent_as_given(monkeypatch, tmp_path: Path, capsys) -> None:
    _this_computer_is(monkeypatch, tmp_path, "linux")
    import asyncio

    daemon = _daemon(_config(tmp_path, machine_id="Alices.Laptop:work_1"))

    hello = asyncio.run(daemon._build_hello())

    assert hello.payload.machine_id == "Alices.Laptop:work_1"
    assert daemon._root_grants.machine_id == "Alices.Laptop:work_1"
    # Nothing was derived, so nothing was made to derive it from.
    assert not machine_identity.seed_path().exists()


def test_a_machine_id_from_the_environment_or_the_config_file_is_sent_as_given(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOGPT_SHIM_MACHINE_ID", "from-the-environment")
    assert _daemon(load_config()).machine_id == "from-the-environment"

    monkeypatch.delenv("AUTOGPT_SHIM_MACHINE_ID")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f'machine_id = "from-the-file"\naudit_log_path = "{(tmp_path / "f.log").as_posix()}"\n',
        encoding="utf-8",
    )
    assert _daemon(load_config(config_path=config_file)).machine_id == "from-the-file"
    assert not machine_identity.seed_path().exists()


@pytest.mark.parametrize("machine_id", ["", "x" * 129])
def test_a_machine_id_the_platform_could_not_take_is_a_config_error(machine_id: str) -> None:
    """The platform takes an ID of 1 to 128 characters."""
    with pytest.raises(ValidationError):
        ShimConfig(machine_id=machine_id)


def test_the_display_name_is_the_hostname_unless_the_person_sets_one(
    monkeypatch, tmp_path: Path
) -> None:
    """The label the person sees beside each computer in the platform's
    folder dialog, and how they tell two apart."""
    import asyncio

    default = asyncio.run(_daemon(_config(tmp_path))._build_hello())
    assert default.payload.display_name == socket.gethostname()

    chosen = _config(tmp_path, audit_log_path=tmp_path / "b.log", display_name="Work laptop")
    assert asyncio.run(_daemon(chosen)._build_hello()).payload.display_name == "Work laptop"

    monkeypatch.setenv("AUTOGPT_SHIM_DISPLAY_NAME", "Kitchen PC")
    from_environment = load_config()
    from_environment.audit_log_path = tmp_path / "c.log"
    assert (
        asyncio.run(_daemon(from_environment)._build_hello()).payload.display_name == "Kitchen PC"
    )


def test_a_chats_session_runs_under_its_machines_id(monkeypatch, tmp_path: Path) -> None:
    """A chat's session is made from its machine's config, with the ID the
    machine derived: it derives none of its own."""
    _this_computer_is(monkeypatch, tmp_path, "linux")
    machine = _daemon(_config(tmp_path))

    def derived_again() -> str:
        raise AssertionError("a chat's session derived a machine ID of its own")

    monkeypatch.setattr(machine_identity, "own_machine_id", derived_again)
    chat_config = machine.config.model_copy(deep=True)
    chat_config.session_id = "chat-1"
    chat = ShimDaemon(
        chat_config,
        token_store=object(),
        audit=AuditWriter(path=tmp_path / "chat.log", audit_key=KEY),
    )

    assert chat.machine_id == machine.machine_id
    assert chat._root_grants.machine_id == machine.machine_id


# ── A start that can't keep its value ────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "permanent", "exit_code"),
    [
        (PermissionError(errno.EACCES, "Permission denied"), True, 78),
        (OSError(errno.EROFS, "Read-only file system"), True, 78),
        (OSError(errno.ENOSPC, "No space left on device"), False, 75),
        (OSError(errno.EIO, "Input/output error"), True, 78),
        (TimeoutError(errno.ETIMEDOUT, "Connection timed out"), False, 75),
    ],
)
def test_a_start_that_cannot_keep_its_value_stops_and_names_the_file(
    monkeypatch, tmp_path: Path, capsys, error: OSError, permanent: bool, exit_code: int
) -> None:
    """With the value unsaved, the shim would be a new computer at every
    start. It stops instead: for good where a restart can't help, and to be
    started again (exit 75) where it can."""
    import asyncio

    def fails(_temp: Path, _final: Path) -> None:
        raise error

    monkeypatch.setattr(machine_identity, "_place", fails)

    with pytest.raises(DaemonPreflightError) as refused:
        _daemon(_config(tmp_path))

    assert str(machine_identity.seed_path()) in str(refused.value)
    # And the two ways out: another folder, or an ID of the person's own.
    assert "AUTOGPT_SHIM_FALLBACK_DIR" in str(refused.value)
    assert "machine_id" in str(refused.value)
    assert refused.value.permanent is permanent
    assert refused.value.transient is not permanent
    assert asyncio.run(cli._cmd_start(_config(tmp_path, audit_log_path=tmp_path / "b.log"))) == (
        exit_code
    )
    printed = capsys.readouterr()
    assert "Preflight failed" in printed.err
    assert str(machine_identity.seed_path()) in printed.err
    assert "machine_id=" not in printed.out


def test_a_start_with_a_file_where_the_folder_belongs_stops_for_good(
    monkeypatch, tmp_path: Path
) -> None:
    """No restart moves the file out of the way. POSIX and Windows report
    it differently (ENOTDIR at the look for a value, EEXIST at making the
    folder); it is for good on both."""
    in_the_way = tmp_path / "a-file"
    in_the_way.write_text("", encoding="utf-8")
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(in_the_way))

    with pytest.raises(DaemonPreflightError) as refused:
        _daemon(_config(tmp_path))

    assert refused.value.permanent is True
    assert str(machine_identity.seed_path()) in str(refused.value)
    assert in_the_way.read_text(encoding="utf-8") == ""


def test_a_start_over_a_file_that_is_not_a_value_stops_for_good(tmp_path: Path) -> None:
    path = machine_identity.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0123")

    with pytest.raises(DaemonPreflightError) as refused:
        _daemon(_config(tmp_path))

    assert refused.value.permanent is True
    assert str(path) in str(refused.value)
    assert path.read_bytes() == b"0123"


# ── Which commands make the value ────────────────────────────────────────────


def _run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", *arguments])
    try:
        cli.main()
    except SystemExit:
        pass


@pytest.mark.parametrize(
    "command",
    [
        ["--help"],
        ["status"],
        ["doctor"],
        ["audit", "verify"],
        ["audit", "verify-all"],
        ["start", "--help"],
    ],
    ids=" ".join,
)
@pytest.mark.parametrize("source", ["linux", "none"])
def test_a_command_that_only_reads_makes_nothing(
    monkeypatch, tmp_path: Path, capsys, command: list[str], source: str
) -> None:
    """Loading the config read the identifier and, with none to read, made
    the ID's file in the home folder, whatever the command. Only `start`
    needs the ID, and only `start` makes the value it is derived from."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "state"))
    # A keychain that holds nothing, not that of whoever runs the tests.
    monkeypatch.setattr(keyring.core, "_keyring_backend", fail.Keyring())
    _this_computer_is(monkeypatch, tmp_path, source)
    _READ.cache_clear()

    def read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError(f"`{' '.join(command)}` read the operating system's identifier")

    monkeypatch.setattr(platform_info, "os_machine_identifier", read)

    _run(monkeypatch, *command)

    assert not machine_identity.seed_path().exists()
    assert list(_fallback_folder().iterdir()) == []
    assert not (home / ".autogpt-local-executor-machine-id").exists()
    assert "AssertionError" not in capsys.readouterr().err


def test_loading_the_config_reads_and_makes_nothing(monkeypatch, tmp_path: Path) -> None:
    def read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("loading the config read the operating system's identifier")

    monkeypatch.setattr(platform_info, "os_machine_identifier", read)

    config = load_config()

    assert config.machine_id is None
    assert not machine_identity.seed_path().exists()


def test_start_makes_the_value(monkeypatch, tmp_path: Path, capsys) -> None:
    import asyncio

    class Runs:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            return None

    monkeypatch.setattr(daemon_module, "ShimDaemon", Runs)
    # An identifier of the test's own: with the real one, a read that
    # didn't come back in time on the computer the tests run on would stop
    # this start, which is another test's subject.
    _this_computer_is(monkeypatch, tmp_path, "linux")
    assert not machine_identity.seed_path().exists()

    assert asyncio.run(cli._cmd_start(_config(tmp_path))) == 0

    seed = bytes.fromhex(machine_identity.seed_path().read_text(encoding="ascii").strip())
    expected = machine_identity.derive_machine_id(platform_info.os_machine_identifier(), seed)
    assert f"Starting shim daemon (machine_id={expected})" in capsys.readouterr().out
