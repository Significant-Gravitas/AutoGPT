"""A call's recording: the spool, its uploads, and deleting them."""

import asyncio
import os
import time
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.voice_relay import locks, tokens
from backend.voice_relay.conftest import FakeSocket, MemoryRecordings, speech
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.locks import CallStore
from backend.voice_relay.metering import CallMeter, Row
from backend.voice_relay.record import (
    CallRecording,
    pcm_of,
    recording_filename,
    remove_stale_spools,
    spool_writable,
    wav_of,
)
from backend.voice_relay.session import Session
from backend.voice_relay.session_test import CAPS, FAST

USER = "user-1"


def _recording(redis, tmp_path, store: MemoryRecordings) -> CallRecording:
    call_id = tokens.new_call_id()
    return CallRecording(
        call_id=call_id,
        user_id=USER,
        spool_dir=tmp_path,
        store=store,
        call_store=CallStore(redis, call_id=call_id, user_id=USER),
    )


async def _named(redis, recording: CallRecording) -> str | None:
    record = await locks.read_call(redis, recording.call_id)
    return record.audio_path if record else None


async def test_every_frame_the_call_takes_is_spooled_in_order(redis, tmp_path):
    recording = _recording(redis, tmp_path, MemoryRecordings())
    frames = speech(0.2)

    for frame in frames:
        recording.append(frame)

    assert recording.samples == sum(len(frame) for frame in frames) // 2
    assert recording.spool.read_bytes() == b"".join(frames)


async def test_a_checkpoint_uploads_the_recording_so_far_and_names_it(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    frames = speech(0.2)
    for frame in frames:
        recording.append(frame)

    assert await recording.checkpoint()

    [path] = store.objects
    assert pcm_of(store.objects[path]) == b"".join(frames)
    assert path.endswith("/" + recording_filename(recording.call_id))
    assert await _named(redis, recording) == path


async def test_each_checkpoint_replaces_the_last_upload(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    recording.append(speech(0.04)[0])
    await recording.checkpoint()
    first = store.stored[0]
    recording.append(speech(0.04)[0])

    await recording.checkpoint()

    assert list(store.objects) == [store.stored[1]]
    assert store.deleted == [first]
    assert await _named(redis, recording) == store.stored[1]
    assert len(pcm_of(store.objects[store.stored[1]])) == 2 * len(speech(0.04)[0])


async def test_a_failed_upload_leaves_the_last_one_named(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    recording.append(speech(0.04)[0])
    await recording.checkpoint()
    store.fail_stores = 1
    recording.append(speech(0.04)[0])

    assert not await recording.checkpoint()

    assert list(store.objects) == store.stored[:1]
    assert await _named(redis, recording) == store.stored[0]


async def test_an_upload_the_record_cannot_name_keeps_the_last_one(
    redis, tmp_path, mocker: MockerFixture
):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    recording.append(speech(0.04)[0])
    await recording.checkpoint()
    mocker.patch.object(
        recording.call_store, "record", new=AsyncMock(return_value=False)
    )
    recording.append(speech(0.04)[0])

    assert not await recording.checkpoint()

    # Both stay: the record names the first, the second waits to be named.
    assert set(store.objects) == set(store.stored)
    assert store.deleted == []


async def test_closing_uploads_once_and_then_deletes_the_spool(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    frames = speech(0.2)
    for frame in frames:
        recording.append(frame)

    pcm = await recording.close()

    assert pcm == b"".join(frames)
    assert len(store.stored) == 1
    assert pcm_of(store.objects[store.stored[0]]) == pcm
    assert not recording.spool.exists()
    # Nothing more is taken once the call's recording is closed.
    recording.append(frames[0])
    assert recording.samples * 2 == len(pcm)


async def test_a_spool_whose_last_upload_failed_is_kept_until_discarded(
    redis, tmp_path
):
    store = MemoryRecordings(fail_stores=1)
    recording = _recording(redis, tmp_path, store)
    recording.append(speech(0.04)[0])

    assert await recording.close()
    assert recording.spool.exists()

    await recording.discard_spool()
    assert not recording.spool.exists()


async def test_discarding_deletes_every_upload_and_the_spool(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)
    recording.append(speech(0.04)[0])
    await recording.checkpoint()
    await recording.close()

    await recording.discard()

    assert store.objects == {}
    assert not recording.spool.exists()


async def test_a_call_with_nothing_recorded_uploads_nothing(redis, tmp_path):
    store = MemoryRecordings()
    recording = _recording(redis, tmp_path, store)

    assert not await recording.checkpoint()
    assert await recording.close() == b""
    assert store.stored == []


def test_a_recording_reads_back_as_the_audio_it_holds():
    pcm = b"".join(speech(0.12))
    assert pcm_of(wav_of(pcm)) == pcm
    with pytest.raises(ValueError):
        pcm_of(b"RIFF not a wave file")


async def test_a_call_checkpoints_its_recording_while_it_runs(redis, tmp_path):
    """Every ``checkpoint_s`` the call uploads what it has, so a pod that dies
    loses only the stretch since."""
    store = MemoryRecordings()
    ws = FakeSocket()
    call_id = tokens.new_call_id()
    call_store = CallStore(redis, call_id=call_id, user_id=USER)
    rows: list[Row] = []

    async def sink(row: Row) -> None:
        rows.append(row)

    session = Session(
        call_id=call_id,
        user_id=USER,
        leg="websocket",
        ws=ws,
        vendor=FakeVendor([Reply(text="Hi", audio_s=0.1)], end_of_speech_s=60.0),
        store=call_store,
        meter=CallMeter(
            user_id=USER,
            call_id=call_id,
            leg="websocket",
            model="gemini-3.8-live",
            redis=redis,
            sink=sink,
        ),
        classifier=None,
        caps=CAPS,
        timing=FAST.model_copy(update={"checkpoint_s": 0.15}),
        recording=CallRecording(
            call_id=call_id,
            user_id=USER,
            spool_dir=tmp_path,
            store=store,
            call_store=call_store,
        ),
    )
    task = asyncio.create_task(session.run())
    await ws.wait_for("ready")
    for _ in range(4):
        ws.send_audio(speech(0.08))
        await asyncio.sleep(0.12)
    while_live = len(store.stored)
    ws.send_json({"type": "hangup"})
    await asyncio.wait_for(task, timeout=5)
    assert session.recording is not None
    pcm = await session.recording.close()

    # Uploads while it ran, one more at its end, and only the last one kept.
    assert while_live >= 2
    assert list(store.objects) == store.stored[-1:]
    assert pcm_of(store.objects[store.stored[-1]]) == pcm
    assert await _named(redis, session.recording) == store.stored[-1]


def test_spools_no_call_still_writes_are_deleted_and_the_rest_kept(tmp_path):
    left = tmp_path / f"{'a' * 32}.pcm"
    left.write_bytes(bytes(640))
    an_hour_ago = time.time() - 3600
    os.utime(left, (an_hour_ago, an_hour_ago))
    writing = tmp_path / f"{'b' * 32}.pcm"
    writing.write_bytes(bytes(640))
    other = tmp_path / "notes.txt"
    other.write_text("not a spool")
    os.utime(other, (an_hour_ago, an_hour_ago))

    assert remove_stale_spools(tmp_path, older_than_s=600) == 1

    assert not left.exists()
    assert writing.exists() and other.exists()


def test_a_spool_directory_that_takes_no_writes_is_reported(tmp_path):
    assert spool_writable(tmp_path / "calls")
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("")
    assert not spool_writable(not_a_directory / "calls")
    # The probe leaves nothing behind.
    assert list((tmp_path / "calls").iterdir()) == []
