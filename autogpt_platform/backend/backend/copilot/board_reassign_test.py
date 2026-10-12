"""The brief a reassigned task's new assignee reads first."""

from backend.copilot.board_reassign import (
    REASSIGN_BRIEF_KIND,
    TRANSCRIPT_CHAR_BUDGET,
    build_reassign_brief,
    reassign_brief_metadata,
)
from backend.copilot.model import ChatMessage

OMITTED = "[earlier messages omitted]"


def _transcript(brief: str) -> str:
    start = brief.index("<previous_conversation>\n") + len("<previous_conversation>\n")
    end = brief.index("\n</previous_conversation>")
    return brief[start:end]


def _brief(messages: list[ChatMessage], files: list[tuple[str, str]] | None = None):
    return build_reassign_brief(
        from_name="Otto", to_name="Priya", messages=messages, files=files or []
    )


def test_the_brief_carries_the_instruction_and_the_conversation() -> None:
    brief = _brief(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
    )

    assert brief.startswith(
        "The user reassigned this task from Otto to you. Below is the "
        "conversation so far, for context only.\n"
    )
    assert 'Open with "You re-assigned this task to me."' in brief
    assert _transcript(brief) == "User: hi\nOtto: hello"


def test_hidden_tool_and_empty_rows_are_left_out() -> None:
    brief = _brief(
        [
            ChatMessage(
                role="user",
                content="kickoff prompt",
                metadata={"hidden": True, "kind": "expert_kickoff"},
            ),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="tool", content="tool output", tool_call_id="t1"),
            ChatMessage(role="assistant", content=""),
            ChatMessage(role="assistant", content="hello"),
        ]
    )

    assert _transcript(brief) == "User: hi\nOtto: hello"


def test_injected_context_is_stripped_from_user_rows() -> None:
    brief = _brief(
        [
            ChatMessage(
                role="user", content="<user_context>plan: pro</user_context>\n\nhi"
            )
        ]
    )

    assert _transcript(brief) == "User: hi"


def test_an_oversize_transcript_drops_the_oldest_turns() -> None:
    turns = [
        ChatMessage(role="user", content=f"turn {i} " + "x" * 3_000) for i in range(20)
    ]

    transcript = _transcript(_brief(turns))

    assert transcript.startswith(OMITTED + "\n")
    assert len(transcript) <= TRANSCRIPT_CHAR_BUDGET
    assert "turn 0 " not in transcript
    assert transcript.endswith("turn 19 " + "x" * 3_000)
    kept = transcript.split("\n")[1:]
    assert all(line.startswith("User: turn ") for line in kept)


def test_a_single_turn_over_the_budget_is_cut_from_its_start() -> None:
    long_reply = "start " + "y" * TRANSCRIPT_CHAR_BUDGET + " end"

    transcript = _transcript(
        _brief(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content=long_reply),
            ]
        )
    )

    assert transcript.startswith(OMITTED + "\n")
    assert len(transcript) <= TRANSCRIPT_CHAR_BUDGET
    assert transcript.endswith(" end")
    assert "start " not in transcript


def test_files_are_listed_with_their_ids() -> None:
    brief = _brief(
        [ChatMessage(role="user", content="hi")],
        files=[("plan.md", "f-1"), ("data.csv", "f-2")],
    )

    assert brief.endswith(
        "</previous_conversation>\n"
        "\n<files>\n"
        "- plan.md (file_id: f-1)\n"
        "- data.csv (file_id: f-2)\n"
        "</files>\n"
        "These files are readable from this conversation."
    )


def test_no_files_means_no_files_block() -> None:
    brief = _brief([ChatMessage(role="user", content="hi")])

    assert "<files>" not in brief
    assert brief.endswith("</previous_conversation>\n")


def test_the_brief_row_is_hidden_and_marked() -> None:
    assert reassign_brief_metadata("e1") == {
        "hidden": True,
        "kind": REASSIGN_BRIEF_KIND,
        "expert_id": "e1",
    }
    assert REASSIGN_BRIEF_KIND == "reassign_brief"
