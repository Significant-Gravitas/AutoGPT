"""The end tool is a request the relay decides on: accepted only once the
person has answered Otto out loud, refused otherwise; and every tool call
logs counts and names, never anything said."""

import logging

import pytest
from prometheus_client import REGISTRY

from backend.voice_relay import locks
from backend.voice_relay.conftest import silence, speech, until
from backend.voice_relay.end_tool import (
    MIN_ANSWER_SPEECH_S,
    NOT_CONFIRMED,
    SAVED,
    Answer,
    arg_names,
    judge,
    safe_name,
)
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.locks import EndToolStats
from backend.voice_relay.session_test import Call, open_call
from backend.voice_relay.vendor import (
    AgentAudio,
    AgentText,
    FunctionCall,
    ToolCall,
    TurnComplete,
    UserText,
)

QUESTION = "So your team starts on customer emails. Does that sound right?"
# A tenth of a second of Otto's speech, as the vendor sends it.
OTTO_AUDIO = bytes(4_800)


def _end_tool_total(outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "voice_interview_end_tool_total", {"outcome": outcome}
        )
        or 0.0
    )


def _unprompted_total() -> float:
    return REGISTRY.get_sample_value("voice_interview_unprompted_turns_total") or 0.0


def _calls_end_tool(first_task: str, call_id: str = "call-1") -> ToolCall:
    return ToolCall(
        calls=(
            FunctionCall(
                call_id=call_id,
                name="finish_onboarding",
                args={"first_task": first_task},
            ),
        )
    )


def _otto_says(text: str) -> tuple[AgentText, AgentAudio]:
    return AgentText(text=text), AgentAudio(pcm=OTTO_AUDIO)


async def _asked(redis, *replies: Reply) -> Call:
    """A call whose Otto has asked whether the first task is right, and
    whose question has played out."""
    vendor = FakeVendor([Reply(text=QUESTION, audio_s=0.1), *replies])
    call = await open_call(redis, vendor)
    await call.ws.wait_for("turn_complete")
    return call


async def _hung_up(call: Call) -> dict:
    call.ws.send_json({"type": "hangup"})
    return await call.ended()


async def test_a_question_met_with_silence_cannot_end_the_call(redis):
    """Otto asks, nobody answers, and the model calls the end tool and says
    goodbye anyway: the tool is refused, and the goodbye ends nothing."""
    before = _end_tool_total("refused_unanswered")
    call = await _asked(redis)
    await call.answer(silence(0.5))
    live = call.vendor.sessions[0]

    live.emit(_calls_end_tool("Answer customer emails"))
    await until(lambda: bool(live.tool_results))
    live.emit(*_otto_says("That sounds perfect. Goodbye for now."), TurnComplete())
    await call.ws.wait_for("turn_complete", count=2)

    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert NOT_CONFIRMED["error"] == "not_confirmed"
    assert not call.session.finish_called
    assert _end_tool_total("refused_unanswered") - before == 1
    # The call goes on until something else ends it.
    assert (await _hung_up(call))["reason"] == "user_hangup"
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None
    assert record.end_tool == EndToolStats(calls=1, refused=1)
    # Nothing the refused call carried is kept as the person's first task.
    assert record.first_task is None


async def test_an_answer_out_loud_then_the_end_tool_ends_the_call_finished(redis):
    before = _end_tool_total("accepted")
    call = await _asked(
        redis,
        Reply(
            text="Great. Your team starts on that. Goodbye!",
            audio_s=0.1,
            heard="Yes, that's right",
            first_task="Answer customer emails",
        ),
    )

    await call.answer(speech(0.5))
    ended = await call.ended()

    assert ended["reason"] == "finished"
    assert [result.response for result in call.vendor.sessions[0].tool_results] == [
        SAVED
    ]
    assert _end_tool_total("accepted") - before == 1
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.first_task == "Answer customer emails"
    assert record.end_tool is not None
    assert (record.end_tool.calls, record.end_tool.refused) == (1, 0)
    assert record.end_tool.accepted_at_s is not None


async def test_a_refused_end_tool_is_accepted_once_the_person_answers(redis):
    call = await _asked(
        redis,
        Reply(
            text="Great. Goodbye!",
            audio_s=0.1,
            heard="Yes",
            first_task="Answer customer emails",
        ),
    )
    live = call.vendor.sessions[0]
    live.emit(_calls_end_tool("Answer customer emails", "call-early"))
    await until(lambda: bool(live.tool_results))
    live.emit(*_otto_says("Take your time."), TurnComplete())
    await call.ws.wait_for("turn_complete", count=2)

    await call.answer(speech(0.5))
    ended = await call.ended()

    assert ended["reason"] == "finished"
    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED, SAVED]
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.end_tool is not None
    assert (record.end_tool.calls, record.end_tool.refused) == (2, 1)


async def test_the_end_tool_called_while_otto_is_still_speaking_is_refused(redis):
    """A question and the end tool in one turn: nothing can have answered a
    turn that has not ended, whatever the person said before it."""
    vendor = FakeVendor(
        [Reply(text="Hi, I'm Otto, an AI. What do you do?", audio_s=0.1)]
    )
    call = await open_call(redis, vendor)
    await call.ws.wait_for("turn_complete")
    live = vendor.sessions[0]
    await call.answer(speech(0.5))
    live.emit(UserText(text="I run a bakery"))
    await until(lambda: call.session.answer.answered)

    live.emit(*_otto_says(QUESTION), _calls_end_tool("Take the orders"))
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    await _hung_up(call)


async def test_a_tool_call_logs_counts_and_names_but_nothing_said(redis, caplog):
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    call = await _asked(
        redis,
        Reply(
            text="Goodbye!",
            audio_s=0.1,
            heard="Yes, my private plan",
            first_task="Ship the private plan",
        ),
    )

    await call.answer(speech(0.5))
    await call.ended()

    [line] = [r.getMessage() for r in caplog.records if " tool " in r.getMessage()]
    assert f"Voice call {call.session.call_id}: tool finish_onboarding call 1" in line
    assert "accepted" in line
    assert "turn 1;" in line
    assert "a first task of 21 characters" in line
    assert "answer sent" in line
    # Neither the first task nor the person's words.
    assert "private" not in line


async def test_an_unknown_tool_logs_its_argument_names_and_never_their_values(
    redis, caplog
):
    caplog.set_level(logging.INFO, logger="backend.voice_relay.session")
    before = _end_tool_total("unknown_tool")
    call = await _asked(redis)
    live = call.vendor.sessions[0]

    live.emit(
        ToolCall(
            calls=(
                FunctionCall(
                    call_id="call-2",
                    name="open<url>!",
                    args={"url": "https://private.example/a", "why": "secret"},
                ),
            )
        )
    )
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [
        {"error": "unknown tool"}
    ]
    assert _end_tool_total("unknown_tool") - before == 1
    [line] = [r.getMessage() for r in caplog.records if " tool " in r.getMessage()]
    assert "tool openurl call 1 unknown_tool" in line
    assert "arguments: url, why" in line
    assert "private" not in line and "secret" not in line
    await _hung_up(call)


async def test_a_turn_otto_begins_with_nothing_heard_or_sent_is_counted_unprompted(
    redis,
):
    before = _unprompted_total()
    vendor = FakeVendor(
        [Reply(text="Hi, I'm Otto, an AI. What do you do?", audio_s=0.1)]
    )
    call = await open_call(redis, vendor)
    await call.ws.wait_for("turn_complete")
    live = vendor.sessions[0]
    # The greeting answers the call's opening.
    assert _unprompted_total() - before == 0

    await call.answer(silence(0.3))
    live.emit(*_otto_says("Are you still there?"), TurnComplete())
    await call.ws.wait_for("turn_complete", count=2)
    assert _unprompted_total() - before == 1

    # A turn after the person spoke is theirs to have prompted.
    await call.answer(speech(0.5))
    live.emit(UserText(text="I run a bakery"))
    await until(lambda: call.session.answer.answered)
    live.emit(*_otto_says("Lovely."), TurnComplete())
    await call.ws.wait_for("turn_complete", count=3)
    assert _unprompted_total() - before == 1
    await _hung_up(call)


# --- the rules on their own ---------------------------------------------------


def test_an_answer_counts_only_what_comes_once_otto_has_played_out():
    answer = Answer()
    answer.hear(1.0, now=5.0)
    answer.caption("early", now=5.0)
    assert answer.silent

    answer.reopen(at=10.0)
    answer.hear(1.0, now=9.9)
    answer.caption("still playing", now=9.9)
    assert answer.silent

    answer.hear(MIN_ANSWER_SPEECH_S, now=10.0)
    assert not answer.answered
    answer.caption("  ", now=10.5)
    assert not answer.answered
    answer.caption(" yes ", now=10.5)
    assert answer.answered
    assert answer.caption_chars == 3

    answer.reopen(at=20.0)
    assert answer.silent and not answer.answered


@pytest.mark.parametrize(
    "speech_s,caption,answered",
    [
        (MIN_ANSWER_SPEECH_S, "yes", True),
        (MIN_ANSWER_SPEECH_S - 0.01, "yes", False),
        (2.0, "", False),
        (0.0, "yes", False),
    ],
    ids=["speech-and-a-caption", "too-little-speech", "no-caption", "no-speech"],
)
def test_the_end_tool_needs_both_speech_and_a_caption(speech_s, caption, answered):
    answer = Answer(opened_at=0.0, speech_s=speech_s)
    answer.caption(caption, now=1.0)

    outcome = judge(
        "finish_onboarding", finishing=False, otto_speaking=False, answer=answer
    )

    assert (outcome == "accepted") is answered


def test_what_a_tool_call_comes_to():
    answered = Answer(opened_at=0.0, speech_s=1.0, caption_chars=3)

    assert (
        judge("finish_onboarding", finishing=True, otto_speaking=True, answer=Answer())
        == "repeat"
    )
    assert (
        judge("finish_onboarding", finishing=False, otto_speaking=True, answer=answered)
        == "refused_unanswered"
    )
    assert (
        judge("open_url", finishing=False, otto_speaking=False, answer=answered)
        == "unknown_tool"
    )


def test_names_are_logged_as_letters_digits_and_underscores_cut_short():
    assert safe_name("finish_onboarding") == "finish_onboarding"
    assert safe_name("rm -rf /; drop<table>") == "rmrfdroptable"
    assert safe_name("x" * 100) == "x" * 64
    assert arg_names({"b": 1, "a b": 2, "c\n": 3}) == ("ab", "b", "c")
