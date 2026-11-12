"""A call's start: the model connect, the room join and the name read run side
by side; the call is ready only once the person's microphone is in its room
and their browser has taken Otto's track; then it is counted, goes live, the
browser hears ``ready``, and the model is sent the call's opening. A call
that cannot get there ends before ``ready``, unpaid for."""

import asyncio
import gc
import logging

import pytest
from prometheus_client import REGISTRY

from backend.voice_relay import locks, rooms
from backend.voice_relay.conftest import (
    FakeMedia,
    FakeSocket,
    counted_calls,
    livekit_edge,
    speech,
    until,
)
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.session import Session
from backend.voice_relay.session_test import FAST, USER, _session

KICKOFF = "(The user just opened the voice onboarding call. Their first name is Dana.)"


class Starting:
    """A call's session running from its start message, with what it runs on."""

    def __init__(self, session: Session, ws: FakeSocket, vendor: FakeVendor, media):
        self.session = session
        self.ws = ws
        self.vendor = vendor
        self.media = media
        self.task: asyncio.Task | None = None

    async def run(self) -> None:
        assert (
            await locks.take_user_lock(
                self.session.store.redis, USER, self.session.call_id
            )
            is None
        )
        self.task = asyncio.create_task(self.session.run())

    async def ended(self) -> str | None:
        assert self.task is not None
        await asyncio.wait_for(asyncio.shield(self.task), timeout=10)
        return self.session.end_reason

    @property
    def room(self):
        return self.media.rooms[0]

    def person_arrives(self) -> None:
        """The person's browser publishes their microphone in the room."""
        person = rooms.person_identity(USER, self.session.call_id)
        self.room.person = person
        self.room.add_microphone(person)


def _starting(
    redis,
    *,
    vendor: FakeVendor | None = None,
    media: FakeMedia | None = None,
    timing=FAST,
    read_first_name=None,
) -> Starting:
    vendor = vendor or FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    media = media or FakeMedia()
    ws = FakeSocket()
    session = _session(
        redis,
        vendor,
        ws,
        [],
        edge=livekit_edge(media),
        timing=timing,
        read_first_name=read_first_name,
    )
    return Starting(session, ws, vendor, media)


async def _waiting(call: Starting) -> None:
    """Wait until the call has opened and waits to be ready."""
    await until(lambda: bool(call.vendor.sessions) and bool(call.media.rooms))
    await until(lambda: call.session.audio is not None)


def _observed(histogram: str) -> float:
    return REGISTRY.get_sample_value(f"{histogram}_count") or 0.0


async def test_the_model_is_sent_nothing_until_the_call_is_ready(redis, counted):
    """Not while the browser still asks for the person's microphone, nor once
    it is in the room and they speak before their browser has taken Otto's
    track. Then, in order: the call counts against the person's allowance,
    goes live, the browser hears ready, and the model gets the opening."""
    media = FakeMedia(user=None)
    media.otto_heard_gate = asyncio.Event()
    call = _starting(redis, media=media, read_first_name=_named("Dana"))
    await call.run()
    await _waiting(call)
    live = call.vendor.sessions[0]
    await asyncio.sleep(0.1)
    assert _sent(live) == ([], 0, 0, [])

    call.person_arrives()
    await media.hearing()
    media.say(speech(0.5))
    await asyncio.sleep(0.1)

    # The microphone is in the room and the person speaks, but the browser
    # has not taken Otto's track: nothing is counted, sent or said.
    assert call.session.away_since is None
    assert call.ws.types() == []
    assert _sent(live) == ([], 0, 0, [])
    assert await counted_calls(redis) == []
    record = await locks.read_call(redis, call.session.call_id)
    assert record is None or record.connect_s is None

    media.otto_heard_gate.set()
    await call.ws.wait_for("ready")
    await until(lambda: bool(live.notes))

    assert await counted_calls(redis) == [1, 1]
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.connect_s is not None
    assert live.notes[0] == (KICKOFF, True)
    await call.ws.wait_for("turn_complete")
    call.ws.send_json({"type": "hangup"})
    assert await call.ended() == "user_hangup"


async def test_ready_waits_for_the_persons_microphone_in_the_room(redis):
    call = _starting(redis, media=FakeMedia(user=None))
    await call.run()
    await _waiting(call)
    await asyncio.sleep(0.1)
    assert call.ws.types() == []

    call.person_arrives()

    await call.ws.wait_for("ready")
    call.ws.send_json({"type": "hangup"})
    await call.ended()


@pytest.mark.parametrize(
    "ends_it,reason",
    [
        ("the-person-elsewhere", "superseded"),
        ("the-socket-closes", "client_gone"),
        ("the-room-is-lost", "room_error"),
        ("no-microphone-in-time", "client_gone"),
    ],
)
async def test_a_call_that_ends_before_it_is_ready_never_goes_live(
    redis, counted, ends_it, reason
):
    """It ends before ready, unpaid for, and with no connect_s on its record:
    a reload passes over it."""
    call = _starting(
        redis,
        media=FakeMedia(user=None),
        timing=FAST.model_copy(
            update={
                "ready_within_s": 0.5 if ends_it == "no-microphone-in-time" else 30.0
            }
        ),
    )
    await call.run()
    await _waiting(call)
    if ends_it == "the-person-elsewhere":
        await locks.request_end(redis, call.session.call_id)
    elif ends_it == "the-socket-closes":
        call.ws.drop()
    elif ends_it == "the-room-is-lost":
        call.room.lost()

    assert await call.ended() == reason

    assert "ready" not in call.ws.types()
    assert call.vendor.sessions[0].notes == []
    assert await counted_calls(redis) == []
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.never_went_live


async def test_a_start_the_model_refuses_at_once_leaves_no_coroutine_unawaited(
    redis, recwarn
):
    """The socket's reader is started first; a refusal before it ever ran
    cancels it unstarted, and it is let go without a warning."""
    gc.collect()
    recwarn.clear()
    call = _starting(
        redis, vendor=FakeVendor(refuse=ConnectionError("the vendor is down"))
    )
    await call.run()

    assert await call.ended() == "vendor_error"
    del call
    gc.collect()
    assert [str(w.message) for w in recwarn if "never awaited" in str(w.message)] == []


async def test_an_opening_the_model_does_not_take_replaces_its_session(
    redis, monkeypatch
):
    """The microphone stays closed until Otto's first turn has played, so a
    session that never took the opening would never hear the person: it is
    replaced, and the replacement opens the call."""
    monkeypatch.setattr("backend.voice_relay.session.VENDOR_SEND_TIMEOUT_S", 0.2)
    vendor = FakeVendor(
        [Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)], stall_opening=True
    )
    call = _starting(redis, vendor=vendor, read_first_name=_named("Dana"))
    await call.run()

    await call.ws.wait_for("turn_complete")

    first, second = vendor.sessions
    assert first.closed
    assert second.notes[0] == (KICKOFF, True)
    assert _said(call) == "Hi, I'm Otto, an AI."
    assert call.session.vendor_reconnects == 1
    call.ws.send_json({"type": "hangup"})
    assert await call.ended() == "user_hangup"


async def test_a_session_lost_while_the_call_got_ready_is_replaced_once(redis):
    """Its stream has ended and a send to it fails: the stream's end and the
    opening's failure start one replacement between them."""
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    media = FakeMedia()
    media.otto_heard_gate = asyncio.Event()
    call = _starting(redis, vendor=vendor, media=media, read_first_name=_named("Dana"))
    await call.run()
    await _waiting(call)
    await vendor.sessions[0].close()

    media.otto_heard_gate.set()
    await call.ws.wait_for("turn_complete")

    assert len(vendor.sessions) == 2
    assert vendor.sessions[1].notes[0] == (KICKOFF, True)
    assert call.session.vendor_reconnects == 1
    call.ws.send_json({"type": "hangup"})
    assert await call.ended() == "user_hangup"


async def test_the_model_connect_the_room_join_and_the_name_read_run_side_by_side(
    redis,
):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    vendor.connect_gate = asyncio.Event()
    media = FakeMedia()
    media.connect_gate = asyncio.Event()
    reading = asyncio.Event()

    async def read_first_name() -> str | None:
        reading.set()
        return "Dana"

    call = _starting(redis, vendor=vendor, media=media, read_first_name=read_first_name)
    await call.run()

    # The name is read and the room is being joined while the model connects.
    await asyncio.wait_for(reading.wait(), timeout=5)
    await until(lambda: bool(media.rooms))
    assert vendor.sessions == []
    vendor.connect_gate.set()
    media.connect_gate.set()

    await call.ws.wait_for("ready")
    await until(lambda: bool(vendor.sessions[0].notes))
    assert vendor.sessions[0].notes[0] == (KICKOFF, True)
    call.ws.send_json({"type": "hangup"})
    await call.ended()


async def test_each_side_of_a_calls_start_is_timed(redis, caplog):
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    vendor_before = _observed("voice_interview_vendor_connect_seconds")
    room_before = _observed("voice_interview_room_join_seconds")
    call = _starting(redis)
    await call.run()

    await call.ws.wait_for("ready")

    assert _observed("voice_interview_vendor_connect_seconds") - vendor_before == 1
    assert _observed("voice_interview_room_join_seconds") - room_before == 1
    [line] = [
        r.getMessage()
        for r in caplog.records
        if "the model connected in" in r.getMessage()
    ]
    assert f"Voice call {call.session.call_id}:" in line
    assert "the relay joined the room in" in line
    call.ws.send_json({"type": "hangup"})
    await call.ended()


def _sent(live) -> tuple[list, int, int, list]:
    """Everything the model session was sent: notes, audio bytes, ends of the
    audio stream and tool results."""
    return live.notes, live.audio_in_bytes, live.stream_ends, live.tool_results


def _said(call: Starting) -> str:
    """What the browser was shown of Otto's words."""
    return "".join(m["text"] for m in call.ws.sent if m["type"] == "agent_text")


def _named(name: str):
    async def read() -> str | None:
        return name

    return read
