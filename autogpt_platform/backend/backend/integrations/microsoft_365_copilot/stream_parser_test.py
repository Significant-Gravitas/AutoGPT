"""Cumulative snapshots are the trap this parser exists for.

Every other provider here streams deltas. Microsoft streams the whole
answer-so-far on every event, so the obvious implementation -- append what
arrives -- produces the answer repeated once per event, growing
quadratically. A reader who has seen the other transports will write that
by reflex, so the first test states it directly.
"""

import json

from backend.integrations.microsoft_365_copilot.stream_parser import (
    CopilotStreamParser,
)


def snapshot(*messages: dict) -> str:
    return json.dumps({"copilotConversation": {"messages": list(messages)}})


def assistant(text: str, message_id: str = "m1") -> dict:
    return {"id": message_id, "role": "assistant", "text": text}


class TestCumulativeSnapshots:
    def test_emits_only_what_is_new(self) -> None:
        """Appending the snapshots verbatim would give
        "Hello" + "Hello there" + "Hello there world"."""
        parser = CopilotStreamParser()

        assert parser.feed(snapshot(assistant("Hello"))) == "Hello"
        assert parser.feed(snapshot(assistant("Hello there"))) == " there"
        assert parser.feed(snapshot(assistant("Hello there world"))) == " world"
        assert parser.text == "Hello there world"

    def test_a_repeated_snapshot_adds_nothing(self) -> None:
        parser = CopilotStreamParser()

        parser.feed(snapshot(assistant("Hello")))

        assert parser.feed(snapshot(assistant("Hello"))) == ""
        assert parser.text == "Hello"

    def test_a_replaced_answer_is_not_diffed_against_the_old_one(self) -> None:
        """A regenerated or filtered answer replaces rather than continues.
        Diffing it against a longer previous string would emit nothing from
        here on -- the stream would look like it had stalled."""
        parser = CopilotStreamParser()

        parser.feed(snapshot(assistant("A long first draft")))
        replaced = parser.feed(snapshot(assistant("Sorry, I cannot help")))

        assert replaced == "Sorry, I cannot help"
        assert parser.text == "Sorry, I cannot help"


class TestWarmingUp:
    def test_a_snapshot_with_no_assistant_message_emits_nothing(self) -> None:
        """The stream warming up, not the model saying nothing."""
        parser = CopilotStreamParser()

        assert parser.feed(snapshot()) == ""
        assert parser.feed(snapshot({"id": "u1", "role": "user", "text": "hi"})) == ""
        assert parser.text == ""

    def test_an_empty_assistant_message_emits_nothing(self) -> None:
        parser = CopilotStreamParser()

        assert parser.feed(snapshot(assistant(""))) == ""
        assert parser.feed(snapshot(assistant("now something"))) == "now something"

    def test_keep_alives_and_the_done_marker_are_not_errors(self) -> None:
        """A healthy stream carries these. Treating them as malformed logs
        noise on every successful turn."""
        parser = CopilotStreamParser()

        assert parser.feed("") == ""
        assert parser.feed("   ") == ""
        assert parser.feed("[DONE]") == ""

    def test_a_malformed_event_is_dropped_rather_than_raising(self) -> None:
        """One bad frame must not take down a turn that is otherwise fine."""
        parser = CopilotStreamParser()

        assert parser.feed("{not json") == ""
        assert parser.feed(snapshot(assistant("still works"))) == "still works"


class TestPickingTheRightMessage:
    def test_locks_onto_the_assistant_answer_and_ignores_other_rows(self) -> None:
        """A conversation carries the user's turn too, and Graph sends the
        whole thing each time. Without locking on, a later row would be read
        as a continuation of the answer."""
        parser = CopilotStreamParser()
        user = {"id": "u1", "role": "user", "text": "what is the launch date"}

        assert parser.feed(snapshot(user, assistant("March"))) == "March"
        assert parser.feed(snapshot(user, assistant("March 4th"))) == " 4th"
        assert parser.text == "March 4th"

    def test_a_second_assistant_row_does_not_hijack_the_answer(self) -> None:
        parser = CopilotStreamParser()

        parser.feed(snapshot(assistant("First answer", "m1")))
        added = parser.feed(
            snapshot(
                assistant("First answer, extended", "m1"),
                assistant("An unrelated later row", "m2"),
            )
        )

        assert added == ", extended"
        assert parser.text == "First answer, extended"

    def test_reads_the_conversation_when_it_is_not_wrapped(self) -> None:
        """Graph has not been stable about the envelope. Reading only the
        wrapped shape would drop the entire answer silently."""
        parser = CopilotStreamParser()
        payload = json.dumps({"messages": [assistant("unwrapped")]})

        assert parser.feed(payload) == "unwrapped"

    def test_reads_the_body_content_shape_too(self) -> None:
        """``body.content`` is how Graph carries message text elsewhere, and
        the Copilot preview has sent both."""
        parser = CopilotStreamParser()
        payload = snapshot(
            {"id": "m1", "role": "assistant", "body": {"content": "from body"}}
        )

        assert parser.feed(payload) == "from body"


class TestUnicode:
    def test_multibyte_text_survives_the_diff(self) -> None:
        """The diff is over characters, not bytes -- an emoji or an accented
        character split mid-sequence would otherwise corrupt the suffix."""
        parser = CopilotStreamParser()

        assert parser.feed(snapshot(assistant("Café"))) == "Café"
        assert parser.feed(snapshot(assistant("Café ☕"))) == " ☕"
        assert parser.feed(snapshot(assistant("Café ☕ 🎉"))) == " 🎉"
        assert parser.text == "Café ☕ 🎉"
