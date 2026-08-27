"""GitHub streams deltas. Microsoft streams snapshots. Do not swap them.

Both providers are called Copilot and both stream a chat answer, which is
exactly why the two accumulators are separate named modules. Applying either
strategy to the other producer fails silently: appending Microsoft's
cumulative snapshots repeats the answer once per event, and diffing GitHub's
incremental deltas emits almost nothing.

The first test states GitHub's direction outright so a reader arriving from
the M365 parser does not assume symmetry.
"""

from backend.integrations.github_copilot.stream_events import (
    CopilotEventAccumulator,
)


def event(event_type: str, **data) -> dict:
    return {"id": "e1", "type": event_type, "data": data}


def delta(text: str, message_id: str = "m1") -> dict:
    return event("assistant.message_delta", messageId=message_id, deltaContent=text)


class TestDeltasAreAppended:
    def test_chunks_accumulate_into_the_answer(self) -> None:
        """Incremental, not cumulative -- the opposite of M365."""
        acc = CopilotEventAccumulator()

        assert acc.feed(delta("Hello")) == "Hello"
        assert acc.feed(delta(" there")) == " there"
        assert acc.feed(delta(" world")) == " world"
        assert acc.turn.text == "Hello there world"

    def test_an_empty_delta_adds_nothing(self) -> None:
        acc = CopilotEventAccumulator()
        acc.feed(delta("Hi"))

        assert acc.feed(delta("")) == ""
        assert acc.turn.text == "Hi"


class TestTheFinalMessageWins:
    def test_it_supersedes_the_deltas_without_repeating_them(self) -> None:
        """The complete text is authoritative, but the user has already seen
        the deltas -- emitting all of it again would duplicate the answer on
        screen."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Hello"))
        acc.feed(delta(" there"))

        added = acc.feed(
            event("assistant.message", messageId="m1", content="Hello there world")
        )

        assert added == " world"
        assert acc.turn.text == "Hello there world"

    def test_it_repairs_a_dropped_delta(self) -> None:
        """This is why the authoritative message is worth handling at all: a
        lost chunk leaves a hole nobody notices, and the final message is
        what closes it."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Hel"))

        acc.feed(event("assistant.message", messageId="m1", content="Hello there"))

        assert acc.turn.text == "Hello there"

    def test_a_final_message_that_diverges_replaces_rather_than_appends(self) -> None:
        """A filtered or rewritten answer is not an extension of the draft."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Here is how to"))

        acc.feed(
            event("assistant.message", messageId="m1", content="I cannot help there.")
        )

        assert acc.turn.text == "I cannot help there."

    def test_a_message_with_no_preceding_deltas_is_taken_whole(self) -> None:
        """Streaming can be off, or the call short enough to arrive at once."""
        acc = CopilotEventAccumulator()

        assert (
            acc.feed(event("assistant.message", messageId="m9", content="All at once"))
            == "All at once"
        )
        assert acc.turn.text == "All at once"


class TestAnAgenticTurnSpansSeveralCalls:
    def test_messages_are_concatenated_in_arrival_order(self) -> None:
        """A turn with a tool call in the middle produces more than one
        message. Keeping only the latest would drop everything said before
        the tool ran."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Looking that up. ", "m1"))
        acc.feed(
            event("assistant.message", messageId="m1", content="Looking that up. ")
        )
        acc.feed(delta("The answer is 42.", "m2"))

        assert acc.turn.text == "Looking that up. The answer is 42."

    def test_the_turn_ends_on_idle_not_on_a_turn_boundary(self) -> None:
        """`assistant.turn_end` fires per LLM call. Ending there truncates an
        agentic answer mid-run; `session.idle` is the real signal."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Working"))

        acc.feed(event("assistant.turn_end", turnId="t1"))
        assert acc.turn.is_complete is False

        acc.feed(event("session.idle"))
        assert acc.turn.is_complete is True
        assert acc.turn.was_aborted is False

    def test_an_aborted_turn_says_so(self) -> None:
        acc = CopilotEventAccumulator()
        acc.feed(event("session.idle", aborted=True))

        assert acc.turn.is_complete is True
        assert acc.turn.was_aborted is True


class TestThingsThatAreNotAnswerText:
    def test_the_progress_counter_never_reaches_the_screen(self) -> None:
        """`assistant.streaming_delta` is cumulative *and* is not text -- it
        carries a byte count. Wiring it to a renderer puts numbers in the
        answer."""
        acc = CopilotEventAccumulator()
        acc.feed(delta("Real text"))

        assert (
            acc.feed(event("assistant.streaming_delta", totalResponseSizeBytes=4096))
            == ""
        )
        assert acc.turn.text == "Real text"
        assert "4096" not in acc.turn.text

    def test_reasoning_is_kept_apart_from_the_answer(self) -> None:
        acc = CopilotEventAccumulator()

        assert (
            acc.feed(event("assistant.reasoning_delta", deltaContent="thinking..."))
            == ""
        )
        assert acc.turn.reasoning == "thinking..."
        assert acc.turn.text == ""

    def test_an_unknown_event_is_ignored_rather_than_raising(self) -> None:
        """The runtime's event vocabulary will grow. A new type must not take
        down a turn that is otherwise fine."""
        acc = CopilotEventAccumulator()

        assert acc.feed(event("some.future.event", whatever=True)) == ""
        assert acc.feed({"type": "assistant.message_delta"}) == ""
        assert acc.feed({}) == ""


class TestFailures:
    def test_an_error_carries_what_it_takes_to_say_something_useful(self) -> None:
        """ "Your Copilot allowance is used up" and "your sign-in expired"
        need different words and different buttons, so the discriminator is
        kept rather than flattened into a message."""
        acc = CopilotEventAccumulator()

        acc.feed(
            event(
                "session.error",
                errorType="quota",
                message="Premium request allowance exhausted.",
                statusCode=402,
                providerCallId="req-123",
            )
        )

        assert acc.turn.error is not None
        assert acc.turn.error.error_type == "quota"
        assert acc.turn.error.status_code == 402
        assert acc.turn.error.provider_call_id == "req-123"

    def test_an_error_ends_the_turn(self) -> None:
        """Without this a caller waiting for `session.idle` waits for a
        message that is never coming."""
        acc = CopilotEventAccumulator()
        acc.feed(event("session.error", errorType="authentication", message="nope"))

        assert acc.turn.is_complete is True

    def test_an_error_with_no_detail_still_produces_something(self) -> None:
        acc = CopilotEventAccumulator()
        acc.feed(event("session.error"))

        assert acc.turn.error is not None
        assert acc.turn.error.error_type == "unknown"
        assert acc.turn.error.message
