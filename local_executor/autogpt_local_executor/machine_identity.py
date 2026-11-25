"""The shim's own ID for the computer it runs on.

HELLO carries `machine_id`, and so do every audit record and every folder
grant. The platform knows a computer by it: a chat is bound to it, and a
second connection under the same ID takes the first one's place for good
(`another_shim_connected`). It was the operating system's own identifier
for the installation, as read (IOPlatformUUID, /etc/machine-id,
MachineGuid), which other software on the computer reads too, and which
machine-id(5) asks applications to hash with a key of their own before it
goes over a network. Where none could be read it was the hostname and a
random UUID.

The ID is now derived from two things together, neither of which is sent,
logged or printed:

- The operating system's identifier, when one can be read
  (platform_info.os_machine_identifier). It tells apart two computers that
  share a home folder, and so the value below.
- A random value the shim makes at the first start that needs an ID, and
  keeps in its own folder (seed_path). It tells apart two computers whose
  operating systems hold the same identifier (made from one image), and
  two accounts on one computer, and it makes the ID something nobody can
  work out from the operating system's identifier.

With no identifier to read, the ID comes from the stored value alone, never
from the hostname. An identifier that is there and couldn't be read this
once is another matter: the ID without it would be another computer's, so
that start stops and the next one reads it again (OsIdentifierUnreadable).

So a computer keeps its ID from one start to the next, and when the program
is reinstalled or upgraded, for as long as the shim's folder is kept and
the operating system's identifier stays the same. Deleting the stored value
gives the computer a new ID.

A `machine_id` the person sets themselves is sent as given, and none of
this runs (daemon.resolve_machine_id).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import secrets
import time
import uuid
from pathlib import Path

from . import platform_info
from .audit import _fallback_path
from .fsops import rename_no_replace_in_one_step

# The stored value's file, in the shim's own folder.
SEED_FILE_NAME = "machine-id-seed"
_SEED_BYTES = 32
_HEX_DIGITS = frozenset("0123456789abcdef")
# How long a start waits for a value another start is still writing under
# the name (_place's last way), and how often it looks.
_BEING_WRITTEN_SECONDS = 2.0
_BEING_WRITTEN_POLL_SECONDS = 0.02

# The derivation below, and the form its inputs take, can't change once
# computers are registered under it: the platform knows a computer by this
# ID and binds each chat to it, so another derivation would cut every chat
# off from its computer. tests/test_machine_identity.py pins it with known
# inputs and outputs. A new one would need a new label and a way to carry
# every computer over.
_DERIVATION = b"autogpt-local-executor machine_id v1"


class MachineIdError(Exception):
    """The value the machine ID is derived from can't be read, made or
    kept. `error` is the storage error behind it, or None for a file that
    doesn't hold a value."""

    def __init__(self, description: str, error: OSError | None = None) -> None:
        super().__init__(f"{description} ({error})" if error is not None else description)
        self.description = description
        self.error = error


class OsIdentifierUnreadable(MachineIdError):
    """The operating system's identifier is there and couldn't be read at
    this start (platform_info.MachineIdentifierUnreadable). The ID derived
    without it would differ from the one every other start derives."""


def canonical_os_identifier(identifier: str | None) -> str | None:
    """`identifier` in the one form the derivation takes: 32 hex digits in
    lower case, without the space around them, braces or dashes, so that one
    computer can't get two IDs from how its identifier is written. Each
    source holds 128 bits; anything else (an empty file, systemd's
    `uninitialized` while a first boot isn't done) is None, no identifier."""
    if identifier is None:
        return None
    text = identifier.strip().strip("{}").replace("-", "").lower()
    if len(text) != 32 or not _HEX_DIGITS.issuperset(text):
        return None
    return text


def derive_machine_id(os_identifier: str | None, seed: bytes) -> str:
    """The machine ID for an operating system's identifier, in any spelling,
    or None for none, and the shim's stored value: a keyed hash of the
    identifier under the stored value (HMAC-SHA256, as
    sd_id128_get_machine_app_specific does with an application's key),
    written as a UUID. Pinned: see _DERIVATION."""
    canonical = canonical_os_identifier(os_identifier) or ""
    message = _DERIVATION + b"\0" + canonical.encode("ascii")
    digest = hmac.new(seed, message, hashlib.sha256).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def seed_path() -> Path:
    """Where the stored value is kept: the shim's own folder under the home
    folder, AUTOGPT_SHIM_FALLBACK_DIR (~/.autogpt-local-executor).

    That folder is the one every shim of an account uses whatever its
    config file and audit log: it holds the lock a new audit key is made
    under, the stored sign-in's locks, and the audit key itself where there
    is no keychain. So an account's shims share one ID however they are
    set up, which the platform's one connection for each computer goes by,
    and the ID sits with the key that signs the folder grants that name it.
    Beside the audit log it would follow `audit_log_path`, and a second
    config would look like a second computer. The setting that moves the
    folder, for a home that can't hold it, moves this with it.

    Not the keychain, where the audit key usually is: Windows' credential
    store can follow an account from one of a domain's computers to another,
    a locked keychain would hold up the ID, and a file can be made in a way
    that lets only one of two starts' values stand.

    The folder is in the home folder, so computers that share one (an NFS
    home, a roaming Windows profile) share the value; their operating
    systems' identifiers tell them apart."""
    return _fallback_path().with_name(SEED_FILE_NAME)


def stored_seed() -> bytes:
    """The shim's stored value, made now if there is none. A write: only a
    start calls this.

    Made once. A new value is written whole to a file of its own (owner-only
    from its creation, synced) and only then given the value's name, in a
    way that fails if the name is taken, always in one step (_place): two
    starts that each found none both go on with the one that got the name
    first. So a file under that name is a whole value, or on a filesystem
    where the name has to be made first, one another start is writing this
    moment, which is waited for (_read_whole). One that stays short of a
    value, or holds anything else, was not written here: it raises
    MachineIdError rather than be read as a value or replaced.

    A storage error raises MachineIdError too. A value made and not kept
    would be another at the next start, and the platform would see a new
    computer each time."""
    path = seed_path()
    seed = _read_whole(path)
    if seed is not None:
        return seed
    temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with os.fdopen(os.open(temp, flags, 0o600), "wb") as handle:
                handle.write(secrets.token_bytes(_SEED_BYTES).hex().encode("ascii") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(FileExistsError):
                # Another start's value has the name: its, not this one's.
                _place(temp, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp)
    except OSError as exc:
        raise MachineIdError(
            f"The value the machine ID is derived from couldn't be kept at {path}", exc
        ) from exc
    seed = _read_whole(path)
    if seed is None:
        raise MachineIdError(
            f"The value the machine ID is derived from was removed from {path} as it was made"
        )
    return seed


def _place(temp: Path, path: Path) -> None:
    """Give the value in the file at `temp` the name `path`, unless
    something has that name already: FileExistsError then, and what has it
    is left alone. In one step, whichever way it takes, so that two starts
    can't both find the name free and one replace the other's value after
    that start has read it.

    - A hard link, where there are hard links: it fails if the name exists,
      on every POSIX filesystem, NFS included.
    - Else a rename that refuses to replace, where the system has one in one
      step: Windows' rename, Linux's and macOS's with their flag for it
      (fsops.rename_no_replace_in_one_step).
    - Else (no links, and a kernel or a filesystem without that flag: FAT
      and exFAT, an SMB share without links, sshfs) the name is made new and
      the value written under it. Making a name new fails if it exists
      everywhere. The value isn't under the name from the first moment
      then, which is why a start waits for one still being written
      (_read_whole).

    Not a look for the name followed by a rename, which those last
    filesystems would otherwise need: another start's value could take the
    name between the two."""
    if os.name != "nt":
        try:
            os.link(temp, path)
            return
        except FileExistsError:
            raise
        except OSError:
            pass
    if rename_no_replace_in_one_step(temp, path):
        return
    _write_under_a_new_name(temp.read_bytes(), path)


def _write_under_a_new_name(value: bytes, path: Path) -> None:
    """Make the name `path`, which fails if something has it, and write
    `value` under it. A write that fails part way takes the name away
    again: this start made it, and it holds no whole value.

    Once the whole value is under the name, it stays, even if syncing it
    then fails: another start may have read it by then and gone on under
    its ID, and with the file removed the next start would make another
    value. The failure is still raised, and this start stops."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    whole = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            whole = True
            os.fsync(handle.fileno())
    except BaseException:
        if not whole:
            with contextlib.suppress(OSError):
                os.unlink(path)
        raise


class _StillBeingWritten(Exception):
    """The file holds the start of a value and nothing else: another start
    may be writing it this moment (_write_under_a_new_name)."""


def _read_whole(path: Path) -> bytes | None:
    """`_read`, waiting a short while for a value another start is still
    writing. One that is still short of a value after that is no value."""
    give_up = time.monotonic() + _BEING_WRITTEN_SECONDS
    while True:
        try:
            return _read(path, may_be_unfinished=time.monotonic() < give_up)
        except _StillBeingWritten:
            time.sleep(_BEING_WRITTEN_POLL_SECONDS)


def _read(path: Path, *, may_be_unfinished: bool = False) -> bytes | None:
    """The value in the file at `path`: None when there is no file,
    MachineIdError when it can't be read or doesn't hold a whole value.
    With `may_be_unfinished`, _StillBeingWritten for a file that holds only
    the start of a value."""
    try:
        with open(path, "rb") as handle:
            # One byte more than a value with its line ending, to tell a
            # longer file from a value.
            data = handle.read(2 * _SEED_BYTES + 3)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MachineIdError(
            f"The value the machine ID is derived from couldn't be read from {path}", exc
        ) from exc
    text = data.decode("ascii", "replace").removesuffix("\n").removesuffix("\r")
    if len(text) != 2 * _SEED_BYTES or not _HEX_DIGITS.issuperset(text):
        if may_be_unfinished and len(data) < 2 * _SEED_BYTES and _HEX_DIGITS.issuperset(text):
            raise _StillBeingWritten
        raise MachineIdError(
            f"{path} doesn't hold the value the machine ID is derived from, which the shim "
            "writes whole or not at all. The shim won't replace it: this computer would get "
            "a new machine ID, and each chat bound to the old one would ask for its computer "
            "and folder again. Move the file aside to start under a new ID"
        )
    return bytes.fromhex(text)


def own_machine_id() -> str:
    """This computer's machine ID, for this account's shim, making the
    stored value if there is none (stored_seed): MachineIdError when it
    can't be read or kept, and OsIdentifierUnreadable when the operating
    system's identifier couldn't be read this once. The identifier comes
    first: a start that stops there has made no value."""
    try:
        identifier = platform_info.os_machine_identifier()
    except platform_info.MachineIdentifierUnreadable as exc:
        raise OsIdentifierUnreadable(
            f"The operating system's identifier for this computer couldn't be read ({exc})"
        ) from exc
    return derive_machine_id(identifier, stored_seed())


__all__ = [
    "SEED_FILE_NAME",
    "MachineIdError",
    "OsIdentifierUnreadable",
    "canonical_os_identifier",
    "derive_machine_id",
    "own_machine_id",
    "seed_path",
    "stored_seed",
]
