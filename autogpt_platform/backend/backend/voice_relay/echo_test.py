"""Otto's voice coming back through the person's microphone: a call starts
closed, the probe finds a microphone that carries him, a half-duplex call
takes nothing of it while he is audible, and a full-duplex call keeps what it
overhears while he is audible until his next turn, then judges it once."""

import asyncio
import logging
import re
import time

import pytest
from prometheus_client import REGISTRY

from backend.voice_relay import locks, vad
from backend.voice_relay.config import IN_RATE
from backend.voice_relay.conftest import silence, speech, until
from backend.voice_relay.echo import (
    ECHO_NOTE,
    HALF_DUPLEX_SHARE,
    OVERHEARD_MIN_S,
    Overheard,
    Probe,
    is_echo,
    silenced,
)
from backend.voice_relay.end_tool import NOT_CONFIRMED, SAVED
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.locks import DuplexStats
from backend.voice_relay.session import SERVER_NOTE
from backend.voice_relay.session_test import FAST, Call, open_call
from backend.voice_relay.vendor import (
    AgentAudio,
    AgentText,
    FunctionCall,
    Interrupted,
    ToolCall,
    TurnComplete,
    UserText,
)

GREETING = "Hi, I'm Otto, an AI. What do you do?"
ASKED = "Is it more customer emails or invoices?"
ECHOED = "That's a unique bakery. What do you sell?"
CONFIRMING = "So, customer emails first. Does that sound right?"
# The probe counts from a tenth of a second after Otto's first audio until a
# fifth of a second after his first turn played out.
TIMING = FAST.model_copy(update={"echo_probe_skip_s": 0.1, "echo_tail_s": 0.2})
# A tenth of a second of Otto's speech, as the vendor sends it.
OTTO_AUDIO = bytes(4_800)
NOTE = (SERVER_NOTE + ECHO_NOTE, True)


def _echo_total(by: str) -> float:
    return REGISTRY.get_sample_value("voice_interview_echo_total", {"by": by}) or 0.0


async def _greeting(redis, tmp_path, timing=TIMING) -> Call:
    """A call whose Otto is greeting the person, for two seconds of audio."""
    vendor = FakeVendor([Reply(text=GREETING, audio_s=2.0)])
    return await open_call(redis, vendor, timing=timing, spool_dir=tmp_path)


async def _in_the_probe(call: Call) -> None:
    """Wait until the probe counts the microphone."""
    session = call.session
    await until(
        lambda: session.probe_from is not None
        and time.monotonic() >= session.probe_from
    )


async def _decided(call: Call) -> None:
    await until(lambda: call.session.duplex != "probing")


async def _half_duplex(redis, tmp_path) -> Call:
    """A call whose microphone carried Otto's greeting."""
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)
    call.say(speech(1.2))
    await _decided(call)
    assert call.session.duplex == "half"
    return call


async def _full_duplex(redis, tmp_path, timing=TIMING) -> Call:
    """A call the probe found clean, past its greeting."""
    call = await _greeting(redis, tmp_path, timing)
    await _in_the_probe(call)
    call.say(silence(1.2))
    await _decided(call)
    assert call.session.duplex == "full"
    return call


async def _otto_speaks(call: Call, text: str, seconds: float) -> None:
    """Otto says *text*, for *seconds* of audio, in a turn still open."""
    call.vendor.sessions[0].emit(
        AgentText(text=text),
        *(AgentAudio(pcm=OTTO_AUDIO) for _ in range(round(seconds * 10))),
    )
    await until(lambda: call.session.cur.audio_s >= seconds - 0.05)


async def _otto_ends_his_turn(call: Call) -> None:
    turns = len(call.session.turns)
    call.vendor.sessions[0].emit(TurnComplete())
    await until(lambda: len(call.session.turns) == turns + 1)


async def _otto_inaudible(call: Call) -> None:
    session = call.session
    await until(lambda: time.monotonic() >= session.playout_end + TIMING.echo_tail_s)


async def _heard(call: Call, frames: list[bytes]) -> None:
    """The person's microphone carries *frames*, all of them taken."""
    before = call.session.received
    call.say(frames)
    await until(lambda: call.session.received >= before + len(b"".join(frames)) // 2)


async def _room_for(call: Call, seconds: float) -> None:
    """Wait until the call can take *seconds* of the person's audio at once."""
    session = call.session
    await until(
        lambda: session.bucket + (time.monotonic() - session.bucket_at) * IN_RATE
        >= seconds * IN_RATE
    )


async def _captioned(call: Call, text: str) -> None:
    call.vendor.sessions[0].emit(UserText(text=text))
    await until(lambda: {"type": "user_text", "text": text} in call.ws.sent)


async def _next_turn_begins(call: Call, text: str = "Sorry about that.") -> None:
    call.vendor.sessions[0].emit(AgentText(text=text))
    await until(lambda: call.session.cur.began)


async def _overheard_echo(call: Call) -> None:
    """Otto's words come back through the speakers while he plays."""
    await _otto_speaks(call, ECHOED, 2.0)
    await _heard(call, speech(0.6))
    await _captioned(call, "That's a unique")
    await _otto_ends_his_turn(call)


async def _hung_up(call: Call) -> locks.CallRecord:
    call.ws.send_json({"type": "hangup"})
    await call.ended()
    record = await locks.read_call(call.session.store.redis, call.session.call_id)
    assert record is not None
    return record


async def _recorded(call: Call) -> bytes:
    assert call.session.recording is not None
    return await call.session.recording.close()


# --- the probe ----------------------------------------------------------------


async def test_a_clean_microphone_makes_the_call_full_duplex(redis, tmp_path):
    before = _echo_total("probe")
    call = await _full_duplex(redis, tmp_path)

    assert call.session.probe_share == 0.0
    assert "duplex" not in call.ws.types()
    assert _echo_total("probe") == before
    record = await _hung_up(call)
    assert record.duplex == DuplexStats(mode="full", by=None, probe_share=0.0)


async def test_a_microphone_that_carries_otto_while_he_greets_makes_the_call_half_duplex(
    redis, tmp_path
):
    before = _echo_total("probe")
    call = await _half_duplex(redis, tmp_path)

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


async def test_a_probe_with_under_1_s_of_mic_records_a_null_share(
    redis, tmp_path, caplog
):
    """Too little to judge by is no evidence of echo, nor of a clean
    microphone."""
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    call = await _greeting(redis, tmp_path)
    await _in_the_probe(call)

    # A third of a second of it, all heard.
    call.say(speech(0.3))
    await _decided(call)

    assert call.session.duplex == "full"
    assert call.session.probe_share is None
    assert any("too little to judge" in r.getMessage() for r in caplog.records)
    record = await _hung_up(call)
    assert record.duplex == DuplexStats(mode="full", by=None, probe_share=None)


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

    call.say(speech(1.2))
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


# --- half-duplex --------------------------------------------------------------


async def test_a_half_duplex_call_takes_nothing_while_otto_is_audible(redis, tmp_path):
    call = await _half_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]

    await _otto_speaks(call, ECHOED, 2.0)
    await _heard(call, speech(0.4))

    # Nothing of it reached the model, the idle clock, the reply timer or the
    # end tool's count of an answer.
    assert live.speech_in_bytes == 0
    assert call.session.last_heard is None
    assert call.session.answer.speech_s == 0

    await _otto_ends_his_turn(call)
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


# --- what a full-duplex call overhears ----------------------------------------


async def test_an_answer_after_otto_stops_that_repeats_his_phrase_stays_the_persons(
    redis, tmp_path
):
    """And the call stays full-duplex: nothing was heard over him."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ASKED, 0.5)
    await _otto_ends_his_turn(call)
    await _otto_inaudible(call)

    await _heard(call, speech(0.5))
    await _captioned(call, "customer emails")
    await _next_turn_begins(call, "Got it.")

    assert call.session.duplex == "full"
    assert call.session.user_buf == "customer emails"
    assert call.vendor.sessions[0].notes_after_opening == []
    await _hung_up(call)


async def test_an_answer_that_starts_in_the_tail_and_carries_on_stays_the_persons(
    redis, tmp_path
):
    """Its words are Otto's, but most of it was heard once he had stopped."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ASKED, 2.0)
    await _heard(call, speech(0.4))
    await _otto_ends_his_turn(call)
    await _otto_inaudible(call)
    await _heard(call, speech(0.8))
    await _captioned(call, "customer emails")

    await _next_turn_begins(call, "Got it.")

    assert call.session.duplex == "full"
    assert call.session.user_buf == "customer emails"
    assert call.vendor.sessions[0].notes_after_opening == []
    await _hung_up(call)


async def test_what_do_you_mean_over_otto_stays_the_persons_and_lands_in_user_buf(
    redis, tmp_path
):
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, "So what do you sell to them?", 2.0)
    await _heard(call, speech(0.6))
    await _captioned(call, "What do you mean?")
    # Kept aside until Otto's next turn, and shown as it came.
    assert call.session.user_buf == ""
    await _otto_ends_his_turn(call)

    await _next_turn_begins(call, "I mean, who buys from you?")

    assert call.session.duplex == "full"
    assert call.session.user_buf == "What do you mean?"
    assert call.vendor.sessions[0].notes_after_opening == []
    await _hung_up(call)


async def test_ottos_words_heard_while_he_plays_are_caught_once_at_his_next_turns_first_event(
    redis, tmp_path
):
    before = _echo_total("captions")
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _overheard_echo(call)
    # Nothing is judged until Otto's next turn begins.
    assert call.session.duplex == "full"
    assert live.notes_after_opening == []

    await _next_turn_begins(call)

    assert call.session.duplex == "half"
    assert call.session.echo_by == "captions"
    assert call.session.user_buf == ""
    assert (call.session.answer.speech_s, call.session.answer.caption_chars) == (0, 0)
    assert [m for m in call.ws.sent if m["type"] == "duplex"] == [
        {"type": "duplex", "mode": "half", "by": "captions"}
    ]
    assert _echo_total("captions") - before == 1
    assert live.notes_after_opening == [NOTE]
    # Half-duplex for good: what the person says next tells the model nothing.
    await _captioned(call, "What do you sell")
    assert live.notes_after_opening == [NOTE]
    record = await _hung_up(call)
    assert record.duplex is not None
    assert (record.duplex.mode, record.duplex.by) == ("half", "captions")
    [(start, end)] = record.duplex.echo_spans
    assert 0 < start < end <= call.session.received
    # Otto's words are not in what the call heard of the person.
    assert all(line["text"] != "That's a unique" for line in call.session.transcript())


async def test_a_caught_span_is_on_the_calls_record_before_the_call_ends(
    redis, tmp_path
):
    """A call whose pod dies is transcribed from its last upload and its
    record as it stood: the span is silenced there too."""
    call = await _full_duplex(redis, tmp_path)
    await _overheard_echo(call)

    await _next_turn_begins(call)
    await until(lambda: call.session.duplex == "half")

    async def recorded_spans() -> list[tuple[int, int]]:
        while True:
            record = await locks.read_call(redis, call.session.call_id)
            if record is not None and record.duplex is not None:
                return record.duplex.echo_spans
            await asyncio.sleep(0.02)

    spans = await asyncio.wait_for(recorded_spans(), timeout=5)
    assert spans == call.session.echo_spans and len(spans) == 1
    assert not call.session.ended.is_set()
    await _hung_up(call)


async def test_an_end_tool_call_that_opens_that_next_turn_is_refused(redis, tmp_path):
    """Otto's voice cut him off and ran on past his echo tail, and its
    caption landed after his cut-off turn ended, so what was heard past the
    tail counts as an answer to it; what the echo made count as one is
    not."""
    call = await _full_duplex(
        redis, tmp_path, TIMING.model_copy(update={"echo_tail_s": 1.0})
    )
    live = call.vendor.sessions[0]
    await _otto_speaks(call, ECHOED, 2.0)
    live.emit(Interrupted())
    await _otto_ends_his_turn(call)
    await _heard(call, speech(1.0))
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _captioned(call, "That's a unique")
    assert call.session.answer.answered

    live.emit(
        ToolCall(
            calls=(
                FunctionCall(
                    call_id="call-1",
                    name="finish_onboarding",
                    args={"first_task": "Answer customer emails"},
                ),
            )
        )
    )
    await until(lambda: bool(live.tool_results))

    assert call.session.duplex == "half"
    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    await _hung_up(call)


def _calls_end_tool(call_id: str = "call-1") -> ToolCall:
    return ToolCall(
        calls=(
            FunctionCall(
                call_id=call_id,
                name="finish_onboarding",
                args={"first_task": "Answer customer emails"},
            ),
        )
    )


async def test_ottos_echo_in_the_tail_after_his_question_is_no_answer(redis, tmp_path):
    """His last word comes back while his echo may still be heard, captioned
    as one word, which is never judged his: it does not answer him either."""
    call = await _full_duplex(
        redis, tmp_path, TIMING.model_copy(update={"echo_tail_s": 1.0})
    )
    live = call.vendor.sessions[0]
    await _otto_speaks(call, "So, customer emails first. Does that sound right?", 0.3)
    await _otto_ends_his_turn(call)
    await until(lambda: time.monotonic() >= call.session.playout_end)

    await _heard(call, speech(0.5))
    await _captioned(call, "Right.")
    live.emit(_calls_end_tool())
    await until(lambda: bool(live.tool_results))

    assert call.session.duplex == "full"
    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    await _hung_up(call)


async def test_a_late_caption_of_ottos_words_heard_over_him_is_caught_before_it_counts(
    redis, tmp_path
):
    """The stretch was judged at his next turn's first word, before its
    caption came: the caption is still checked against his turns, and what
    the stretch made count as an answer is dropped."""
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, ECHOED, 2.0)
    await _heard(call, speech(1.0))
    await _otto_ends_his_turn(call)
    await _otto_inaudible(call)
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _next_turn_begins(call, "Great.")
    assert call.session.duplex == "full"

    await _captioned(call, "That's a unique")
    live.emit(_calls_end_tool())
    await until(lambda: bool(live.tool_results))

    assert call.session.duplex == "half"
    assert call.session.echo_by == "captions"
    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    assert call.session.user_buf == ""
    assert live.notes_after_opening == [NOTE]
    record = await _hung_up(call)
    assert record.duplex is not None
    [(start, end)] = record.duplex.echo_spans
    assert 0 < start < end <= call.session.received
    assert all(line["text"] != "That's a unique" for line in call.session.transcript())


async def test_a_caption_after_ottos_next_turn_ended_is_not_judged_with_the_stretch(
    redis, tmp_path
):
    """What the person says once his next turn has ended answers that turn."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ECHOED, 2.0)
    await _heard(call, speech(1.0))
    await _otto_ends_his_turn(call)
    await _next_turn_begins(call, "What else do you sell?")
    await _otto_ends_his_turn(call)

    await _captioned(call, "That's a unique")

    assert call.session.duplex == "full"
    assert call.session.user_buf == "That's a unique"
    assert call.vendor.sessions[0].notes_after_opening == []
    await _hung_up(call)


async def test_a_caught_stretch_silences_only_what_was_heard_over_otto(redis, tmp_path):
    """Speech over Otto's question, then an answer in the clear in his
    words: the stretch is caught, and the answer stays in the recording the
    transcript of record is made from."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ASKED, 2.0)
    await _heard(call, speech(1.0))
    heard_over_him = call.session.received
    await _otto_ends_his_turn(call)
    await _otto_inaudible(call)
    await _heard(call, speech(0.4))
    await _captioned(call, "customer emails")

    await _next_turn_begins(call, "Got it.")

    record = await _hung_up(call)
    assert record.duplex is not None
    [(start, end)] = record.duplex.echo_spans
    assert 0 <= start < end <= heard_over_him
    kept = silenced(await _recorded(call), record.duplex.echo_spans)
    assert vad.heard(kept[heard_over_him * 2 :])


async def test_one_word_of_ottos_heard_mostly_over_him_is_no_answer(
    redis, tmp_path, caplog
):
    """Sound over his question, then one of his words past the tail: too
    short to be judged his, it still answers nothing, so the end tool is
    refused once. The call stays full-duplex and the word stays the
    person's; their next answer is one."""
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    before = _echo_total("captions")
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, CONFIRMING, 2.0)
    await _heard(call, speech(1.0))
    await _otto_ends_his_turn(call)
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _captioned(call, "Right.")

    live.emit(_calls_end_tool())
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    assert call.session.duplex == "full"
    assert call.session.user_buf == "Right."
    assert call.session.echo_spans == []
    assert live.notes_after_opening == []
    assert _echo_total("captions") == before
    [line] = [r.getMessage() for r in caplog.records if "one word" in r.getMessage()]
    assert re.search(r"\d+\.\d\ds over him, \d+\.\d\ds clear", line)
    assert "Right" not in line

    await _otto_speaks(call, "Take your time.", 0.3)
    await _otto_ends_his_turn(call)
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _captioned(call, "Yes.")
    live.emit(_calls_end_tool("call-2"))
    await until(lambda: len(live.tool_results) == 2)

    assert live.tool_results[1].response == SAVED
    assert call.session.finish_called
    await _hung_up(call)


async def test_a_late_one_word_caption_of_ottos_heard_mostly_over_him_is_no_answer(
    redis, tmp_path
):
    """The stretch was judged at his next turn's first word, before its
    caption came: one word of his arriving then answers nothing either, and
    stays the person's."""
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, CONFIRMING, 2.0)
    await _heard(call, speech(1.0))
    await _otto_ends_his_turn(call)
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _next_turn_begins(call, "Great.")

    await _captioned(call, "Right.")
    live.emit(_calls_end_tool())
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert call.session.duplex == "full"
    assert call.session.user_buf == "Right."
    assert call.session.echo_spans == []
    assert live.notes_after_opening == []
    await _hung_up(call)


async def test_one_word_in_the_clear_with_nothing_heard_over_otto_answers_him(
    redis, tmp_path
):
    """With nothing heard over him, a word of his said past the tail is an
    answer like any other."""
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _otto_speaks(call, CONFIRMING, 2.0)
    await _otto_ends_his_turn(call)
    await until(lambda: call.session.answer.is_open(time.monotonic()))
    await _heard(call, speech(0.4))
    await _captioned(call, "Right.")

    live.emit(_calls_end_tool())
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [SAVED]
    assert call.session.finish_called
    await _hung_up(call)


async def test_a_second_of_speech_clear_of_otto_is_the_persons_whatever_was_heard_over_him(
    redis, tmp_path
):
    """Twice as much heard over his question as after it, and a run of his
    words, but over a second of it in the clear: an echo is never clear of
    him that long, so it is the person's answer."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ASKED, 3.0)
    # The relay takes at most two seconds of audio at once: the 2.4 s heard
    # over him comes in two halves.
    await _heard(call, speech(1.2))
    await _room_for(call, 1.2)
    await _heard(call, speech(1.2))
    await _otto_ends_his_turn(call)
    await _otto_inaudible(call)
    await _heard(call, speech(1.2))
    await _captioned(call, "customer emails")

    await _next_turn_begins(call, "Got it.")

    assert call.session.duplex == "full"
    assert call.session.user_buf == "customer emails"
    assert call.vendor.sessions[0].notes_after_opening == []
    assert call.session.echo_spans == []
    await _hung_up(call)


async def test_a_catch_is_logged_with_the_seconds_heard_over_and_clear_of_otto_and_no_words(
    redis, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    call = await _full_duplex(redis, tmp_path)
    await _overheard_echo(call)

    await _next_turn_begins(call)

    [line] = [
        r.getMessage() for r in caplog.records if "his own words" in r.getMessage()
    ]
    assert re.search(r"\d+\.\d\ds over him, \d+\.\d\ds clear", line)
    assert "unique" not in line
    await _hung_up(call)


@pytest.mark.parametrize("saying_goodbye", ["wrapping", "finishing"])
async def test_no_note_while_wrapping_or_finishing(redis, tmp_path, saying_goodbye):
    """ "Carry on from your last question" must not restart a goodbye. The
    call still goes half-duplex, and forgets what the echo counted."""
    call = await _full_duplex(redis, tmp_path)
    live = call.vendor.sessions[0]
    await _overheard_echo(call)
    if saying_goodbye == "wrapping":
        await call.session.wrap_up("scope")
    else:
        call.session.finish_called = True
        call.session.finish_at = time.monotonic()
    goodbye_notes = list(live.notes_after_opening)

    await _next_turn_begins(call, "Goodbye for now.")

    assert call.session.duplex == "half"
    assert call.session.answer.caption_chars == 0
    assert live.notes_after_opening == goodbye_notes
    await _hung_up(call)


async def test_an_open_stretch_is_released_at_the_calls_end(redis, tmp_path):
    """Never judged, its caption is the person's."""
    call = await _full_duplex(redis, tmp_path)
    await _otto_speaks(call, ECHOED, 2.0)
    await _heard(call, speech(0.6))
    await _captioned(call, "That's a unique")

    record = await _hung_up(call)

    assert call.session.duplex == "full"
    assert record.duplex is not None and record.duplex.echo_spans == []
    assert {"who": "user", "text": "That's a unique"} in call.session.transcript()


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
        "in-the-latest-turn",
        "one-word",
        "not-in-a-run",
        "the-persons-own",
        "in-the-turn-before",
    ],
)
def test_a_caption_is_ottos_words_when_two_or_more_run_in_his_turn(caption, echo):
    assert is_echo(caption, ECHOED, GREETING) is echo


@pytest.mark.parametrize(
    "caption,echo",
    [("Right.", True), ("Yes.", False)],
    ids=["one-of-his-words", "not-one-of-his"],
)
def test_one_word_is_ottos_words_where_one_word_is_enough(caption, echo):
    assert is_echo(caption, CONFIRMING, min_words=1) is echo


@pytest.mark.parametrize(
    "over_s,clear_s,text,may_be",
    [
        (1.0, 0.4, "Right.", True),
        (0.6, 0.31, "Right.", False),
        (1.0, 0.4, "Yes.", False),
        (1.0, 0.4, "", False),
    ],
    ids=["one-of-his-words-over-him", "mostly-in-the-clear", "not-his", "no-caption"],
)
def test_one_word_of_ottos_may_be_his_only_when_mostly_heard_over_him(
    over_s, clear_s, text, may_be
):
    stretch = Overheard(start=0, end=0, over_s=over_s, clear_s=clear_s, text=text)

    assert stretch.may_be_otto(GREETING, CONFIRMING) is may_be


@pytest.mark.parametrize(
    "over_s,clear_s,text,otto",
    [
        (0.6, 0.0, "That's a unique", True),
        (OVERHEARD_MIN_S - 0.01, 0.0, "That's a unique", False),
        (0.6, 0.31, "That's a unique", False),
        (0.6, 0.3, "That's a unique", True),
        (0.6, 0.0, "What do you mean", False),
        (0.6, 0.0, "customer", False),
        (3.0, 0.99, "That's a unique", True),
        (3.0, 1.0, "That's a unique", False),
    ],
    ids=[
        "over-otto-and-his-words",
        "too-little-over-him",
        "mostly-in-the-clear",
        "twice-as-much-over-him",
        "not-a-run-of-his",
        "one-word",
        "under-a-second-clear",
        "a-second-clear",
    ],
)
def test_an_overheard_stretch_is_ottos_when_mostly_heard_over_him_and_a_run_of_his(
    over_s, clear_s, text, otto
):
    stretch = Overheard(start=0, end=0, over_s=over_s, clear_s=clear_s, text=text)

    assert stretch.is_otto(ECHOED, GREETING) is otto


@pytest.mark.parametrize(
    "frames,heard,seconds,carries",
    [
        (50, 10, 1.0, True),
        (50, 5, 1.0, False),
        (10, 10, 0.4, False),
        (0, 0, 0.0, False),
    ],
    ids=["a-fifth-heard", "a-tenth-heard", "too-little-to-judge", "no-frames"],
)
def test_the_probe_finds_otto_in_a_fifth_of_a_second_or_more_of_frames(
    frames, heard, seconds, carries
):
    probe = Probe()
    for index in range(frames):
        probe.count(heard=index < heard, seconds=seconds / frames)

    assert probe.carries_otto is carries


def test_a_caught_span_is_silenced_within_the_recording():
    pcm = bytes(range(1, 21))

    assert silenced(pcm, []) == pcm
    assert silenced(pcm, [(2, 4)]) == pcm[:4] + bytes(4) + pcm[8:]
    # Clamped to the recording's ten samples.
    assert silenced(pcm, [(8, 50)]) == pcm[:16] + bytes(4)
    assert silenced(pcm, [(30, 40), (-5, 1)]) == bytes(2) + pcm[2:]
