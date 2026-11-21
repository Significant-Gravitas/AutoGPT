"""PROTOCOL.md says what a share that stops answering costs a running chat.

A control request's looks at a folder are bounded: five seconds, eight left
waiting. A file request on a chat's own connection is not. It runs in one of
the event loop's worker threads, and a thread that waits on a share that has
gone away holds its chat's slot, one of the machine's slots and the thread
until the share answers or the process is killed. Four of them hold every
machine slot, and other chats' file and shell requests are then refused.

Nothing here changes that. The tests hold the numbers PROTOCOL.md gives to
the code's, so that the two change together.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from autogpt_local_executor import daemon as daemon_module
from autogpt_local_executor import handlers, install
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import MachineSlots, ShimDaemon

ROOT = Path(__file__).resolve().parent.parent
_WORDS = {4: "four", 10: "ten"}


def _section() -> str:
    protocol = (ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    start = protocol.index("###### On a chat's own connection")
    return " ".join(protocol[start : protocol.index("\n#### ", start)].split())


def _where(thread: threading.Thread) -> tuple[str, bool]:
    return thread.name, thread.daemon


async def test_file_work_runs_in_the_event_loops_own_worker_threads() -> None:
    """Which is what a stuck request holds, and why the process can't exit
    while one is stuck: asyncio.run and the interpreter both wait for those
    threads, as they don't for a daemon thread."""
    read_in = await handlers._read_in_a_thread(lambda: _where(threading.current_thread()))
    changed_in = await handlers._run_to_completion(lambda: _where(threading.current_thread()))

    for name, is_daemon in (read_in, changed_in):
        assert name.startswith("asyncio_"), name
        assert not is_daemon


async def test_the_machine_has_as_many_slots_as_one_chat(tmp_path: Path) -> None:
    """So one chat's requests can hold them all."""
    config = ShimConfig(
        allowed_root=tmp_path / "workspace",
        audit_log_path=tmp_path / "audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
    )
    machine = ShimDaemon(
        config,
        token_store=object(),
        audit=AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32),
    )

    slots = machine._machine_work_semaphore
    assert isinstance(slots, MachineSlots)
    for _ in range(config.max_concurrent):
        assert not slots.locked()
        await slots.acquire()
    assert slots.locked()
    assert config.max_concurrent == 4


def test_the_section_gives_the_codes_numbers() -> None:
    section = _section()
    slots = ShimConfig.model_fields["max_concurrent"].default

    assert f"(`max_concurrent`, {slots} by default)" in section
    assert f"of which there are {slots} by default as well" in section
    # By the limit the connection settled on, with the default as its example.
    assert "the smaller of the shim's own setting and the `HELLO_ACK.max_concurrent`" in section
    assert (
        "As many requests waiting on such a share as the machine has slots, "
        f"{_WORDS[slots]} by default, therefore hold every one of them" in section
    )
    assert (
        f"wait {_WORDS[int(daemon_module.MACHINE_WORK_WAIT_SECONDS)]} seconds for a slot" in section
    )
    assert f"let go after {_WORDS[int(daemon_module.DISPATCH_DRAIN_SECONDS)]} seconds" in section
    # The event loop's pool: min(32, cores + 4), by the cores Python counts,
    # which from 3.13 are the ones the process may use.
    assert "the CPU cores Python counts, up to 32: 6 with two cores, 12 with eight" in section
    assert "From Python 3.13 it counts the cores the shim's process may use" in section
    assert (min(32, 2 + 4), min(32, 8 + 4)) == (6, 12)
    assert f"With the default {_WORDS[slots]} slots, the held threads leave two" in section
    assert "of a two-core computer's six free" in section
    assert min(32, 2 + 4) - slots == 2


def test_the_section_says_what_is_refused_and_what_still_works() -> None:
    section = _section()

    assert '`SHIM_OVERLOADED` with `details.reason: "machine_busy"`' in section
    assert "`SHIM_OVERLOADED` (`details.max_concurrent`)" in section
    assert "the control connection" in section and "`COMPUTER_TOOL_CALL`s" in section
    assert "The 30 seconds the platform waits for the reply free none of them" in section


def test_the_section_says_what_the_person_does() -> None:
    section = _section()

    assert "restart the shim" in section
    assert f"`launchctl kickstart -k gui/$(id -u)/{install.LAUNCHD_LABEL}`" in section
    unit = install.SYSTEMD_UNIT_NAME.removesuffix(".service")
    assert f"`systemctl --user restart {unit}`" in section
    task = install.TASKSCHED_TASK_NAME
    assert f"`schtasks /End /TN {task}`" in section and f"`schtasks /Run /TN {task}`" in section
    assert "has to be killed" in section


def test_the_bounded_looks_no_longer_read_as_covering_a_chats_requests() -> None:
    protocol = " ".join((ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8").split())

    assert "Activations and restores of other chats, whose folders are other folders" in protocol
    # The folder picker's own calls take turns, which the section says too.
    assert "The folder picker takes one call at a time" in protocol
    assert re.search(r"A chat's own file requests are another matter", protocol)
    # The Concurrency section points at it from where the slots are described.
    concurrency = protocol[protocol.index("## Concurrency") :]
    assert (
        "as many such requests as there are slots (four by default) hold every one"
        in (concurrency[:1500])
    )
