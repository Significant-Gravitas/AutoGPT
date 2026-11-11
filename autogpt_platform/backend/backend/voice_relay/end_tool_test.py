"""The end tool is a request the relay decides on: accepted only once the
person has answered Otto out loud, refused otherwise; and every tool call
logs counts and names, never anything said."""

import logging

import pytest
from prometheus_client import REGISTRY

from backend.voice_relay import locks
from backend.voice_relay.conftest import silence, speech, until
from backend.voice_relay.end_tool import (
    MAX_ACK_AUDIO_S,
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
from backend.voice_relay.session_test import FAST, Call, open_call
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


def _otto_speaks_for(seconds: float) -> list[AgentAudio]:
    """Otto's audio for *seconds*, a tenth of a second at a time."""
    return [AgentAudio(pcm=OTTO_AUDIO) for _ in range(round(seconds * 10))]


async def _answered(call: Call) -> None:
    """The person answers Otto's question out loud, and is captioned."""
    await call.answer(speech(0.5))
    call.vendor.sessions[0].emit(UserText(text="Yes, that's right"))
    await until(lambda: call.session.answer.answered)


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


async def test_a_tool_call_that_starts_a_turn_with_no_answer_is_refused(redis):
    """Otto asks, nobody answers, and the model begins a turn with the end
    tool and says goodbye anyway: the tool is refused, and the goodbye ends
    nothing."""
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


async def test_a_question_and_the_tool_in_one_turn_is_refused_despite_an_earlier_answer(
    redis,
):
    """The person's answer was to the question before: nothing has been heard
    since the one in this turn."""
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


async def test_an_acknowledgement_then_the_tool_after_a_real_answer_is_accepted(
    redis,
):
    """ "Great!", the tool, then the goodbye, all in one turn: the call ends
    as that turn does, not on the grace past the tool."""
    vendor = FakeVendor([Reply(text=QUESTION, audio_s=0.1)])
    call = await open_call(
        redis, vendor, timing=FAST.model_copy(update={"finish_grace_s": 60.0})
    )
    await call.ws.wait_for("turn_complete")
    live = vendor.sessions[0]
    await _answered(call)

    live.emit(*_otto_says("Great!"), _calls_end_tool("Answer customer emails"))
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [SAVED]
    assert call.session.finish_called
    live.emit(*_otto_says(" Goodbye for now."), TurnComplete())
    assert (await call.ended(timeout=10))["reason"] == "finished"


async def test_more_than_four_seconds_of_ottos_audio_before_the_tool_is_refused(
    redis,
):
    """That much said in the turn is a new proposal, not a "Great!", and
    nobody has answered it, though its question has not been captioned."""
    call = await _asked(redis)
    live = call.vendor.sessions[0]
    await _answered(call)

    live.emit(
        AgentText(text="Then your team will start on the orders"),
        *_otto_speaks_for(MAX_ACK_AUDIO_S + 0.2),
        _calls_end_tool("Take the orders"),
    )
    await until(lambda: bool(live.tool_results))

    assert [result.response for result in live.tool_results] == [NOT_CONFIRMED]
    assert not call.session.finish_called
    await _hung_up(call)


async def test_a_second_call_after_a_refusal_in_the_same_turn_is_refused(redis):
    """Refused, then "take your time", then the tool again before the turn
    ends: still refused, even with an answer heard meanwhile."""
    call = await _asked(redis)
    live = call.vendor.sessions[0]
    live.emit(_calls_end_tool("Answer customer emails", "call-early"))
    await until(lambda: len(live.tool_results) == 1)
    live.emit(*_otto_says("Take your time."))
    await _answered(call)

    live.emit(_calls_end_tool("Answer customer emails", "call-again"))
    await until(lambda: len(live.tool_results) == 2)

    assert [result.response for result in live.tool_results] == [
        NOT_CONFIRMED,
        NOT_CONFIRMED,
    ]
    assert not call.session.finish_called
    await _hung_up(call)


async def test_two_end_tool_calls_in_one_event_are_judged_in_order(redis):
    call = await _asked(redis)
    live = call.vendor.sessions[0]
    await _answered(call)

    live.emit(
        ToolCall(
            calls=(
                *_calls_end_tool("Answer customer emails", "call-a").calls,
                *_calls_end_tool("Answer customer emails", "call-b").calls,
            )
        )
    )
    await until(lambda: len(live.tool_results) == 2)

    # The first is accepted; the second only repeats it.
    assert [result.response for result in live.tool_results] == [SAVED, SAVED]
    assert call.session.end_tool.calls == 2 and call.session.end_tool.refused == 0
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
    assert "turn 1, 0.0s of his audio in it;" in line
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


async def test_an_unprompted_turn_is_still_counted_with_under_0_3_s_of_noise_and_no_caption(
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

    # A quiet room's noise, too short to be an answer, and no caption.
    await call.answer(speech(0.1))
    await until(lambda: call.session.answer.speech_s > 0)
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

    assert (_judged(answer=answer) == "accepted") is answered


def _judged(
    name: str = "finish_onboarding",
    *,
    finishing: bool = False,
    answer: Answer | None = None,
    turn_audio_s: float = 0.0,
    turn_text: str = "",
    refused_this_turn: bool = False,
) -> str:
    return judge(
        name,
        finishing=finishing,
        answer=answer or Answer(opened_at=0.0, speech_s=1.0, caption_chars=3),
        turn_audio_s=turn_audio_s,
        turn_text=turn_text,
        refused_this_turn=refused_this_turn,
    )


@pytest.mark.parametrize(
    "case,outcome",
    [
        ({}, "accepted"),
        ({"turn_text": "Great!", "turn_audio_s": 0.6}, "accepted"),
        ({"turn_audio_s": MAX_ACK_AUDIO_S}, "accepted"),
        ({"turn_audio_s": MAX_ACK_AUDIO_S + 0.1}, "refused_unanswered"),
        ({"turn_text": "Does that sound right?"}, "refused_unanswered"),
        ({"refused_this_turn": True}, "refused_unanswered"),
        ({"answer": Answer()}, "refused_unanswered"),
        ({"finishing": True, "answer": Answer()}, "repeat"),
        ({"name": "open_url"}, "unknown_tool"),
    ],
    ids=[
        "answered-at-the-turns-start",
        "answered-then-acknowledged",
        "answered-at-the-acknowledgements-limit",
        "answered-then-said-too-much",
        "answered-then-asked-again",
        "refused-earlier-in-the-turn",
        "unanswered",
        "accepted-already",
        "another-tool",
    ],
)
def test_what_a_tool_call_comes_to(case, outcome):
    assert _judged(**case) == outcome


def test_names_are_logged_as_letters_digits_and_underscores_cut_short():
    assert safe_name("finish_onboarding") == "finish_onboarding"
    assert safe_name("rm -rf /; drop<table>") == "rmrfdroptable"
    assert safe_name("x" * 100) == "x" * 64
    assert arg_names({"b": 1, "a b": 2, "c\n": 3}) == ("ab", "b", "c")
