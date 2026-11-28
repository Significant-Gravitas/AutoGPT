"""A running writer doesn't carry on from a log that lost its records.

When the log wasn't as a writer left it (another writer appended, or
rotated it), the writer took its next sequence number and chain value from
whatever the file's last line now was, without looking at that line. Cut
the log back to record 4 while the shim had written 10, and its next record
became 5, chained to 4: the log verified, and records 5 to 10 were gone
without a trace.

A running writer knows the last record it wrote, and the first record of
the chain it is on. When the log changed under it:

- its last record is still there, with another writer's records after it:
  it goes on from the last of them;
- the log is still its chain, cut short or rewritten: it keeps the chain it
  had, so the log fails verification from its next record on;
- the log is another one (empty, gone, another chain): a rotation leaves
  that, and the log is taken up. If no rotated file holds its last record,
  its records are no longer on disk, and its first record in the log it
  takes up says so.

Each of the last two is said in the shim's own log.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from autogpt_local_executor import cli
from autogpt_local_executor.audit import AuditLogCorruptError, AuditWriter, list_rotated_files

KEY = b"\x00" * 32


def _writer(tmp_path: Path) -> AuditWriter:
    return AuditWriter(
        path=tmp_path / "audit.log",
        audit_key=KEY,
        shim_version="test-0.0",
        machine_id="test-machine",
        session_id="test-session",
    )


async def _write(writer: AuditWriter, *names: str) -> None:
    for name in names:
        await writer.write("EXECUTE_COMMAND", request_id=name, details={}, result={})


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _kinds(path: Path) -> set[str]:
    return {violation.kind for violation in AuditWriter.verify(path, KEY)}


def _cut_back(path: Path, keep: int, *, how: str) -> None:
    """Leave the log's first `keep` records, the way a person or a tool
    would: in place, or by putting a shorter copy in its place."""
    lines = path.read_bytes().splitlines(keepends=True)
    kept = b"".join(lines[:keep])
    if how == "in-place":
        with open(path, "r+b") as file:
            file.truncate(len(kept))
    else:
        assert how == "replaced"
        shorter = path.with_name("shorter-copy")
        shorter.write_bytes(kept)
        os.replace(shorter, path)


NAMES = [f"r{number}" for number in range(1, 11)]


@pytest.mark.parametrize("how", ["in-place", "replaced"])
@pytest.mark.parametrize("keep", [4, 9, 1])
async def test_a_log_cut_back_under_a_running_writer_fails_verification(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, how: str, keep: int
) -> None:
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    assert AuditWriter.verify(writer.path, KEY) == []
    _cut_back(writer.path, keep, how=how)
    # What is left is a valid, shorter chain: nothing in the file says more.
    assert AuditWriter.verify(writer.path, KEY) == []

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "after")

    # The writer's next record is the eleventh, chained to the tenth, which
    # is gone: the log says so to anyone who verifies it.
    last = _records(writer.path)[-1]
    assert (last["request_id"], last["seq"]) == ("after", 11)
    assert {"sequence_gap", "chain_break"} <= _kinds(writer.path)
    assert str(writer.path) in caplog.text
    assert "record 10" in caplog.text
    # And said once: the writer goes on from its own record.
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "and-on")
    assert caplog.text == ""
    assert [record["seq"] for record in _records(writer.path)[-2:]] == [11, 12]


def _removed_record(record: dict) -> bool:
    return record["op"] == "AUDIT_RECORDS_REMOVED"


@pytest.mark.parametrize("how", ["emptied", "removed", "another-chain"])
async def test_a_log_emptied_removed_or_replaced_under_a_running_writer_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    """The log is no longer this writer's chain, and no rotated file holds
    its records. That isn't a rotation, but it can't be told from one whose
    rotated file was pruned, so the chain isn't broken over it: the log is
    taken up, and the writer's first record in it says its records are no
    longer on disk. It used to carry on as if nothing had happened."""
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    last = _records(writer.path)[-1]
    expected_before = 0
    if how == "emptied":
        with open(writer.path, "r+b") as file:
            file.truncate(0)
    elif how == "removed":
        writer.path.unlink()
    else:
        # Another log put in its place: one the same key signed, as an old
        # rotated file would be.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        other = _writer(elsewhere)
        await _write(other, "o1", "o2", "o3")
        os.replace(other.path, writer.path)
        expected_before = 3

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "after")

    records = _records(writer.path)
    said, after = records[expected_before:]
    assert _removed_record(said)
    assert said["details"] == {"last_seq": 10, "last_hmac": last["hmac"]}
    assert said["result"]["ok"] is False
    assert (after["request_id"], after["seq"]) == ("after", said["seq"] + 1)
    # The log is whole as far as it goes, and a restarted shim takes it.
    assert AuditWriter.verify(writer.path, KEY) == []
    _writer(tmp_path)
    assert "record 10" in caplog.text
    # Said once.
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "and-on")
    assert caplog.text == ""
    assert sum(map(_removed_record, _records(writer.path))) == 1


async def test_a_writer_that_has_not_written_yet_knows_the_record_it_found(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A shim that starts on a log knows its last record from then on."""
    await _write(_writer(tmp_path), *NAMES)
    restarted = _writer(tmp_path)
    _cut_back(restarted.path, 4, how="replaced")

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(restarted, "after")

    assert _records(restarted.path)[-1]["seq"] == 11
    assert {"sequence_gap", "chain_break"} <= _kinds(restarted.path)
    assert "record 10" in caplog.text


async def test_the_next_start_refuses_a_log_that_lost_records(tmp_path: Path) -> None:
    """As it refuses any log that fails verification."""
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    _cut_back(writer.path, 4, how="in-place")
    await _write(writer, "after")

    with pytest.raises(AuditLogCorruptError):
        _writer(tmp_path)


# ── What must still be taken up cleanly ──────────────────────────────────────


async def test_another_writers_records_are_carried_on_from(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two writers share a log (two shims; a shim and an `audit` command):
    each finds the other's records after its own and chains on from the
    last of them."""
    first = _writer(tmp_path)
    await _write(first, "a1", "a2")
    second = _writer(tmp_path)

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(second, "b1", "b2")
        await _write(first, "a3")
        await _write(second, "b3")
        await _write(first, "a4", "a5")
        await _write(second, "b4")

    assert [(record["request_id"], record["seq"]) for record in _records(first.path)] == [
        ("a1", 1), ("a2", 2), ("b1", 3), ("b2", 4), ("a3", 5),
        ("b3", 6), ("a4", 7), ("a5", 8), ("b4", 9),
    ]  # fmt: skip
    assert AuditWriter.verify(first.path, KEY) == []
    assert caplog.text == ""


async def test_another_writers_rotation_is_taken_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The records are whole in the rotated file, and the log is a new
    chain: empty, or already holding what the other writer wrote since."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    daemon = _writer(tmp_path)
    await _write(daemon, *NAMES)

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        assert cli._audit_rotate(daemon.path) == 0
        await _write(daemon, "after-an-empty-log")
        other = _writer(tmp_path)
        assert other.force_rotate() is not None
        await _write(other, "the-others-first")
        await _write(daemon, "after-the-others-record")

    assert caplog.text == ""
    rotated = list_rotated_files(daemon.path)
    assert len(rotated) == 2
    for path in (*rotated, daemon.path):
        assert AuditWriter.verify(path, KEY) == []
    assert [(record["request_id"], record["seq"]) for record in _records(daemon.path)] == [
        ("the-others-first", 1),
        ("after-the-others-record", 2),
    ]


async def test_a_writers_own_rotation_starts_a_new_chain(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        assert writer.force_rotate() is not None
        await _write(writer, "after")

    assert caplog.text == ""
    assert [(record["request_id"], record["seq"]) for record in _records(writer.path)] == [
        ("after", 1)
    ]
    assert AuditWriter.verify(writer.path, KEY) == []


async def test_a_rotated_file_that_does_not_hold_the_writers_record_is_no_rotation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file with a rotated log's name doesn't make an emptied log a
    rotation: the writer's last record has to be in it."""
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    lines = writer.path.read_bytes().splitlines(keepends=True)
    writer.path.with_name("audit.log.20200101-000000").write_bytes(b"".join(lines[:4]))
    with open(writer.path, "r+b") as file:
        file.truncate(0)

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "after")

    said, after = _records(writer.path)
    assert _removed_record(said)
    assert after["request_id"] == "after"
    assert "record 10" in caplog.text


async def test_a_rotation_whose_rotated_file_was_pruned_does_not_break_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`audit rotate` and then a prune that takes the rotated file, before
    the shim's next record: the shim's own commands, and nothing to refuse
    the next start over. The records are gone all the same, and the log
    says so."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    daemon = _writer(tmp_path)
    await _write(daemon, *NAMES)
    assert cli._audit_rotate(daemon.path) == 0
    [rotated] = list_rotated_files(daemon.path)
    rotated.unlink()

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(daemon, "after")

    said, after = _records(daemon.path)
    assert _removed_record(said) and (said["seq"], after["seq"]) == (1, 2)
    assert AuditWriter.verify(daemon.path, KEY) == []
    _writer(tmp_path)


async def test_records_that_do_not_chain_on_from_the_writers_are_not_carried_on_from(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The writer's record is there, but what follows it isn't another
    writer's work: a record put back out of an older copy, say."""
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    lines = writer.path.read_bytes().splitlines(keepends=True)
    with open(writer.path, "ab") as file:
        file.write(lines[2])

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor.audit"):
        await _write(writer, "after")

    assert _records(writer.path)[-1]["seq"] == 11
    assert AuditWriter.verify(writer.path, KEY) != []
    assert "record 10" in caplog.text


async def test_verify_says_what_a_removed_record_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The chain verifies with an AUDIT_RECORDS_REMOVED record in it, so
    `audit verify` and `verify-all` say what the record says."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    assert cli._audit_verify(writer.path) == 0
    assert "note:" not in capsys.readouterr().out
    writer.path.unlink()
    await _write(writer, "after")

    assert cli._audit_verify(writer.path) == 0
    out = capsys.readouterr().out
    assert "audit.log: OK" in out
    assert "note: seq=1 records that the log before it, up to seq=10" in out

    assert cli._audit_verify_all(writer.path) == 0
    assert "up to seq=10" in capsys.readouterr().out


# ── Review: evidence that must outlast a rotation and a failed write ─────────


@pytest.mark.parametrize("how", ["in-place", "replaced"])
async def test_a_cut_back_log_is_not_rotated_away_before_the_record_that_shows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A log that was cut back fails verification at the writer's next
    record, which follows one that is no longer there. A rotation that fell
    due at that write came first: it put the shortened log away, a valid
    chain by itself, and started a new one. Both verified, and nothing
    showed the loss."""
    from autogpt_local_executor import audit as audit_module

    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    _cut_back(writer.path, 4, how=how)
    # The log's first record is now older than a log may get.
    monkeypatch.setattr(audit_module, "ROTATE_MAX_AGE_SECONDS", -1.0)

    await _write(writer, "after-the-cut")

    files = [writer.path, *list_rotated_files(writer.path)]
    broken = [path.name for path in files if {"sequence_gap", "chain_break"} <= _kinds(path)]
    assert len(broken) == 1, {path.name: _kinds(path) for path in files}
    # Rotation goes on as usual once the record is in.
    await _write(writer, "next")
    assert _records(writer.path)[-1]["request_id"] == "next"


async def test_a_removal_whose_record_could_not_be_written_is_recorded_at_the_next_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log was emptied under the writer, and the disk was full when it
    went to say so. The writer had already let go of the last record it
    knew, so its next write, with room again, began a clean chain with no
    AUDIT_RECORDS_REMOVED in it."""
    writer = _writer(tmp_path)
    await _write(writer, *NAMES[:3])
    last = _records(writer.path)[-1]
    writer.path.write_bytes(b"")
    append = writer._append

    def full(_line: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(writer, "_append", full)
    with pytest.raises(OSError, match="No space left"):
        await _write(writer, "while-full")
    monkeypatch.setattr(writer, "_append", append)

    await _write(writer, "room-again")

    records = _records(writer.path)
    assert [record["op"] for record in records] == ["AUDIT_RECORDS_REMOVED", "EXECUTE_COMMAND"]
    assert records[0]["details"] == {"last_seq": last["seq"], "last_hmac": last["hmac"]}
    assert records[1]["request_id"] == "room-again"
    assert AuditWriter.verify(writer.path, KEY) == []


async def test_a_removal_is_recorded_once_when_the_write_after_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    await _write(writer, *NAMES[:3])
    writer.path.write_bytes(b"")
    append = writer._append
    calls = 0

    def full_after_the_first(line: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(28, "No space left on device")
        append(line)

    monkeypatch.setattr(writer, "_append", full_after_the_first)
    with pytest.raises(OSError, match="No space left"):
        await _write(writer, "while-full")

    await _write(writer, "room-again")

    assert [record["op"] for record in _records(writer.path)] == [
        "AUDIT_RECORDS_REMOVED",
        "EXECUTE_COMMAND",
    ]
    assert AuditWriter.verify(writer.path, KEY) == []


async def test_verify_does_not_vouch_for_a_removal_record_in_a_log_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The note for an AUDIT_RECORDS_REMOVED record is read off the file as
    it is. In a log that fails verification the record may be one nobody
    signed (written in by hand, or changed), and `audit verify` printed
    what it says as plainly as for a log that verified."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await _write(writer, *NAMES)
    writer.path.unlink()
    await _write(writer, "after")
    # The record is changed after the shim wrote it: it now names another
    # last record, and its HMAC is no longer its own.
    changed = writer.path.read_bytes().replace(b'"last_seq": 10', b'"last_seq": 3', 1)
    assert changed != writer.path.read_bytes()
    writer.path.write_bytes(changed)

    for verify in (cli._audit_verify, cli._audit_verify_all):
        assert verify(writer.path) == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "records that the log before it" not in out
        assert "note: seq=1 reads as a record that the log before it, up to seq=3" in out
        assert "failed verification" in out
