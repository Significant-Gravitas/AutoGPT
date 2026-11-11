"""Otto's voice coming back through the person's microphone: a call starts
closed, the probe finds a microphone that carries him, a half-duplex call
takes nothing of it while he is audible, a full-duplex call catches his words
in the person's captions, and the transcript of record drops what was his."""

import time

import pytest
from prometheus_client import REGISTRY

from backend.voice_relay import locks, vad
from backend.voice_relay.conftest import silence, speech, until
from backend.voice_relay.echo import (
    ECHO_CLIP_SHARE,
    ECHO_NOTE,
    HALF_DUPLEX_SHARE,
    Probe,
    echoed_clip,
    is_echo,
    words_of,
)
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.locks import DuplexStats, TurnMark
from backend.voice_relay.postcall import without_echo
from backend.voice_relay.session import SERVER_NOTE
from backend.voice_relay.session_test import FAST, Call, open_call
from backend.voice_relay.vendor import AgentAudio, AgentText, TurnComplete, UserText

GREETING = "Hi, I'm Otto, an AI. What do you do?"
ECHOED = "That's a unique bakery. What do you sell?"
# The probe counts from a tenth of a second after Otto's first audio until a
# fifth of a second after his first turn played out.
TIMING = FAST.model_copy(update={"echo_probe_skip_s": 0.1, "echo_tail_s": 0.2})
# A tenth of a second of Otto's speech, as the vendor sends it.
OTTO_AUDIO = bytes(4_800)


def _echo_total(by: str) -> float:
    return REGISTRY.get_sample_value("voice_interview_echo_total", {"by": by}) or 0.0


async def _greeting(redis, tmp_path) -> Call:
    """A call whose Otto is greeting the person, for two seconds of audio."""
    vendor = FakeVendor([Reply(text=GREETING, audio_s=2.0)])
    return await open_call(redis, vendor, timing=TIMING, spool_dir=tmp_path)


async def _in_the_probe(call: Call) -> None:
    """Wait until the probe counts the microphone."""
    session = call.session
    await until(
        lambda: session.probe_from is not None
        and time.monotonic() >= session.probe_from
    )


async def _decided(call: Call) -> None:
    await until(lambda: call.session.duplex != "probing")


async def _otto_speaks(call: Call, text: str, seconds: float) -> None:
    """Otto says *text*, for *seconds* of audio, in a turn still open."""
    call.vendor.sessions[0].emit(
        AgentText(text=text),
        *(AgentAudio(pcm=OTTO_AUDIO) for _ in range(round(seconds * 10))),
    )
    await until(lambda: call.session.cur.audio_s >= seconds - 0.05)


async def _full_duplex(redis, tmp_path) -> Call:
    """A call the probe found clean, past its greeting."""
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)
    call.say(silence(0.4))
    await _decided(call)
    assert call.session.duplex == "full"
    return call


async def _hung_up(call: Call) -> locks.CallRecord:
    call.ws.send_json({"type": "hangup"})
    await call.ended()
    record = await locks.read_call(call.session.store.redis, call.session.call_id)
    assert record is not None
    return record


async def _recorded(call: Call) -> bytes:
    assert call.session.recording is not None
    return await call.session.recording.close()


async def test_a_clean_microphone_makes_the_call_full_duplex(redis, tmp_path):
    before = _echo_total("probe")
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)

    call.say(silence(0.4))
    await _decided(call)

    assert call.session.duplex == "full"
    assert call.session.probe_share == 0.0
    assert "duplex" not in call.ws.types()
    assert _echo_total("probe") == before
    record = await _hung_up(call)
    assert record.duplex == DuplexStats(mode="full", by=None, probe_share=0.0)


async def test_a_microphone_that_carries_otto_while_he_greets_makes_the_call_half_duplex(
    redis, tmp_path
):
    before = _echo_total("probe")
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)

    # Otto's greeting comes back through the person's speakers.
    call.say(speech(0.5))
    await _decided(call)

    assert call.session.duplex == "half"
    assert call.session.probe_share is not None
    assert call.session.probe_share >= HALF_DUPLEX_SHARE
    duplex = [m for m in call.ws.sent if m["type"] == "duplex"]
    assert duplex == [{"type": "duplex", "mode": "half", "by": "probe"}]
    assert _echo_total("probe") - before == 1
    # The call starts closed: neither the model nor the recording got any of
    # it, nor did the call count it as the person speaking.
    assert call.vendor.sessions[0].speech_in_bytes == 0
    assert call.session.last_heard is None
    record = await _hung_up(call)
    recorded = await _recorded(call)
    assert recorded and not any(recorded)
    assert record.duplex is not None
    assert (record.duplex.mode, record.duplex.by) == ("half", "probe")
    assert record.duplex.probe_share == call.session.probe_share


async def test_what_a_working_canceller_leaks_as_it_settles_is_not_counted(
    redis, tmp_path
):
    """Speech in the probe's first moments is a canceller still settling."""
    vendor = FakeVendor([Reply(text=GREETING, audio_s=1.0)])
    call = await open_call(
        redis,
        vendor,
        timing=TIMING.model_copy(update={"echo_probe_skip_s": 5.0}),
        spool_dir=tmp_path,
    )
    await until(lambda: call.session.probe_from is not None)

    call.say(speech(0.5))
    await _decided(call)

    assert call.session.duplex == "full"
    assert call.session.probe.frames == 0
    await _hung_up(call)


async def test_the_probe_decides_once_ottos_first_turn_played_out_with_no_frame_then(
    redis, tmp_path
):
    call = await _greeting(redis, tmp_path)

    await _decided(call)

    assert call.session.duplex == "full"
    assert call.session.probe.frames == 0
    await _hung_up(call)


async def test_a_half_duplex_call_takes_nothing_while_otto_is_audible(redis, tmp_path):
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)
    call.say(speech(0.5))
    await _decided(call)
    assert call.session.duplex == "half"
    live = call.vendor.sessions[0]

    await _otto_speaks(call, ECHOED, 2.0)
    before = call.session.received
    call.say(speech(0.4))
    await until(
        lambda: call.session.received >= before + len(b"".join(speech(0.4))) // 2
    )

    # Nothing of it reached the model, the idle clock, the reply timer or the
    # end tool's count of an answer.
    assert live.speech_in_bytes == 0
    assert call.session.last_heard is None
    assert call.session.answer.speech_s == 0

    live.emit(TurnComplete())
    await until(lambda: call.session.mic_open(time.monotonic()))
    said = speech(0.4)
    call.say(said)
    await until(lambda: live.speech_in_bytes > 0)
    assert call.session.last_heard is not None
    assert call.session.answer.speech_s > 0
    await _hung_up(call)
    recorded = await _recorded(call)
    # The recording holds what the model was given: only the answer.
    assert vad.heard(recorded[-len(b"".join(said)) :])
    assert not any(recorded[: -len(b"".join(said))])


async def test_ottos_own_words_in_a_caption_make_a_full_duplex_call_half_duplex_once(
    redis, tmp_path
):
    before = _echo_total("captions")
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, ECHOED, 2.0)
    call.say(speech(0.4))
    await until(lambda: call.session.answer.speech_s > 0)

    live.emit(UserText(text="That's a"))
    await until(lambda: call.session.duplex == "half")

    assert call.session.echo_by == "captions"
    assert call.session.user_buf == ""
    # What the echo made count as an answer is dropped with it.
    assert (call.session.answer.speech_s, call.session.answer.caption_chars) == (
        0,
        0,
    )
    assert [m for m in call.ws.sent if m["type"] == "duplex"] == [
        {"type": "duplex", "mode": "half", "by": "captions"}
    ]
    assert "That's a" not in [m.get("text") for m in call.ws.sent]
    assert _echo_total("captions") - before == 1
    await until(lambda: bool(live.notes_after_opening))
    assert live.notes_after_opening == [(SERVER_NOTE + ECHO_NOTE, True)]

    # The call is half-duplex for good: another echo tells the model nothing.
    live.emit(UserText(text="What do you sell"))
    await until(lambda: "What do you sell" in [m.get("text") for m in call.ws.sent])
    assert live.notes_after_opening == [(SERVER_NOTE + ECHO_NOTE, True)]
    record = await _hung_up(call)
    assert record.duplex is not None
    assert (record.duplex.mode, record.duplex.by) == ("half", "captions")


async def test_a_caption_that_is_not_ottos_words_is_the_persons(redis, tmp_path):
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, ECHOED, 0.5)
    live.emit(TurnComplete())
    await call.ws.wait_for("turn_complete", count=2)

    # One word of Otto's, and two of his not in a run, are the person's.
    live.emit(UserText(text="Unique"), UserText(text=" sell bread"))
    await until(lambda: call.session.user_buf == "Unique sell bread")

    assert call.session.duplex == "full"
    assert live.notes_after_opening == []
    assert {"type": "user_text", "text": " sell bread"} in call.ws.sent
    await _hung_up(call)


# --- the rules on their own ---------------------------------------------------


@pytest.mark.parametrize(
    "caption,echo",
    [
        ("That's a", True),
        ("that's A unique!", True),
        ("what do you sell", True),
        ("Unique", False),
        ("unique sell", False),
        ("I sell bread", False),
        ("Hi I'm Otto", True),
    ],
    ids=[
        "two-words-in-a-run",
        "case-and-punctuation-aside",
        "in-the-current-turn",
        "one-word",
        "not-in-a-run",
        "the-persons-own",
        "in-the-previous-turn",
    ],
)
def test_a_caption_is_ottos_when_two_or_more_of_its_words_run_in_his_turn(
    caption, echo
):
    assert is_echo(caption, ECHOED, GREETING) is echo


@pytest.mark.parametrize(
    "text,echoed",
    [
        ("That's a unique bakery!", True),
        ("I run a bakery.", False),
        ("Yes, a unique bakery, what do you", True),
        ("", False),
    ],
    ids=["all-his", "mostly-the-persons", "four-fifths-his", "nothing-said"],
)
def test_a_clip_is_ottos_when_four_fifths_of_its_words_are_his(text, echoed):
    assert echoed_clip(text, GREETING, ECHOED) is echoed


def test_the_clip_rule_is_four_fifths_of_the_words():
    assert ECHO_CLIP_SHARE == 0.80
    assert len(words_of("Yes, a unique bakery, what do you")) == 7


def test_each_turn_is_judged_against_ottos_turns_either_side_of_it():
    marks = [
        TurnMark(index=0, at=0, text=GREETING),
        TurnMark(index=1, at=100, text=ECHOED),
        TurnMark(index=2, at=200, text="Lovely."),
    ]

    kept = without_echo(["I run a bakery.", "what do you sell", "hi i'm otto"], marks)

    # The last turn's words are the greeting's, two turns away: not this one.
    assert kept == ["I run a bakery.", "", "hi i'm otto"]


@pytest.mark.parametrize(
    "frames,heard,carries",
    [(10, 2, True), (10, 1, False), (0, 0, False)],
    ids=["a-fifth-heard", "a-tenth-heard", "no-frames"],
)
def test_the_probe_finds_otto_in_a_fifth_of_the_frames(frames, heard, carries):
    probe = Probe()
    for index in range(frames):
        probe.count(heard=index < heard)

    assert probe.carries_otto is carries
