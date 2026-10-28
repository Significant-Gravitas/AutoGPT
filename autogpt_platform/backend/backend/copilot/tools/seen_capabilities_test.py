"""The per-turn ``<seen_capabilities>`` block is derived from the session's
persisted tool-call history (SECRT-2791)."""

import json

from backend.copilot.baseline.service import _prepend_skills_notice_to_current_message
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import _maybe_prepend_seen_capabilities
from backend.copilot.service import SEEN_CAPABILITIES_TAG, strip_server_injected_tags
from backend.copilot.tools.seen_capabilities import (
    MAX_LISTED_IDS,
    build_seen_capabilities_notice,
    render_seen_capabilities,
    seen_capabilities,
)


def _session() -> ChatSession:
    return ChatSession.new(user_id="user-1", dry_run=False)


def _call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant(*calls: dict) -> ChatMessage:
    return ChatMessage(role="assistant", content="", tool_calls=list(calls))


def _result(call_id: str, payload: dict | str) -> ChatMessage:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ChatMessage(role="tool", content=content, tool_call_id=call_id)


def test_a_fresh_session_renders_nothing():
    session = _session()
    session.messages.append(ChatMessage(role="user", content="hi"))

    assert not seen_capabilities(session)
    assert build_seen_capabilities_notice(session) == ""


def test_described_and_run_ids_and_loaded_skills_are_collected_once():
    session = _session()
    session.messages.extend(
        [
            ChatMessage(role="user", content="first turn"),
            _assistant(
                _call("describe_capability", {"id": "block:abc"}, "c1"),
                _call("describe_capability", {"id": "tool:TodoWrite"}, "c2"),
                _call("read_skill", {"name": "default-developer"}, "c3"),
            ),
            _result("c1", {"type": "block_details"}),
            _result("c2", {"type": "capability_details"}),
            _result("c3", {"type": "skill"}),
            _assistant(
                _call("run_capability", {"id": "block:abc", "input": {"x": 1}}, "c4"),
                _call("describe_capability", {"id": "block:abc"}, "c5"),
                _call("run_capability", {"id": "block:def", "input": {}}, "c6"),
            ),
            _result("c4", {"type": "block_output"}),
            _result("c5", {"type": "block_details"}),
            _result("c6", {"type": "block_details"}),
            ChatMessage(role="user", content="second turn"),
        ]
    )

    seen = seen_capabilities(session)

    assert set(seen.described) == {"block:abc", "tool:TodoWrite", "block:def"}
    assert len(seen.described) == 3
    assert seen.skills == ["default-developer"]


def test_a_skill_run_through_the_dispatcher_counts_as_loaded():
    # The baseline engine persists the call as the model made it:
    # ``run_capability(id="skill:<name>")``, not the ``read_skill`` the
    # dispatcher resolves it to.
    session = _session()
    session.messages.extend(
        [
            _assistant(
                _call("run_capability", {"id": "skill:How-To-Use-Conductor"}, "c1"),
                _call("describe_capability", {"id": "skill:other"}, "c2"),
            ),
            _result("c1", {"type": "skill"}),
            _result("c2", {"type": "capability_details"}),
        ]
    )

    seen = seen_capabilities(session)

    assert seen.skills == ["how-to-use-conductor"]
    # Describing a skill is not loading it.
    assert seen.described == ["skill:other"]


def test_calls_without_a_non_error_result_do_not_count():
    session = _session()
    session.messages.extend(
        [
            _assistant(
                _call("describe_capability", {"id": "block:nope"}, "c1"),
                _call("read_skill", {"name": "missing"}, "c2"),
                _call("describe_capability", {"id": "block:ok"}, "c3"),
                _call("describe_capability", {"id": "block:interrupted"}, "c4"),
                _call("describe_capability", {"id": "block:orphan"}, "c5"),
                _call("run_capability", {"id": "block:bad-input"}, "c6"),
                _call("describe_capability", {"id": "block:untyped"}, "c7"),
            ),
            _result("c1", {"type": "error", "message": "Unknown capability id."}),
            _result("c2", {"type": "error", "message": "not found"}),
            _result("c3", {"type": "block_details"}),
            _result("c4", "[Tool call interrupted before it produced a result]"),
            # c5 never got a result row at all.
            _result("c6", {"type": "input_validation_error", "errors": []}),
            _result("c7", {"message": "no type field"}),
        ]
    )

    seen = seen_capabilities(session)

    assert seen.described == ["block:ok"]
    assert seen.skills == []


def test_malformed_rows_are_skipped_not_fatal():
    session = _session()
    session.messages.extend(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "c1", "function": {"name": "describe_capability"}},
                    {
                        "id": "c2",
                        "function": {"name": "run_capability", "arguments": "{bad"},
                    },
                    {
                        "id": "c3",
                        "name": "describe_capability",
                        "arguments": {"id": "block:x"},
                    },
                ],
            ),
            _result("c1", {"type": "capability_details"}),
            _result("c2", {"type": "block_details"}),
            _result("c3", {"type": "block_details"}),
        ]
    )

    assert seen_capabilities(session).described == ["block:x"]


def test_the_list_is_bounded_and_keeps_the_most_recent_ids():
    session = _session()
    calls = [
        _call("describe_capability", {"id": f"block:{i}"}, f"c{i}")
        for i in range(MAX_LISTED_IDS + 5)
    ]
    session.messages.append(_assistant(*calls))
    session.messages.extend(
        _result(f"c{i}", {"type": "block_details"}) for i in range(len(calls))
    )

    seen = seen_capabilities(session)

    assert len(seen.described) == MAX_LISTED_IDS
    assert seen.described[0] == f"block:{MAX_LISTED_IDS + 4}"
    assert "block:0" not in seen.described


def test_the_rendered_block_names_the_ids_and_skills_and_is_a_server_tag():
    session = _session()
    session.messages.extend(
        [
            _assistant(
                _call("describe_capability", {"id": "block:abc"}, "c1"),
                _call("read_skill", {"name": "default-developer"}, "c2"),
            ),
            _result("c1", {"type": "block_details"}),
            _result("c2", {"type": "skill"}),
        ]
    )

    notice = build_seen_capabilities_notice(session)

    assert notice.startswith(f"<{SEEN_CAPABILITIES_TAG}>\n")
    assert notice.endswith(f"</{SEEN_CAPABILITIES_TAG}>\n\n")
    assert "block:abc" in notice
    assert "default-developer" in notice
    assert "describe_capability" in notice
    # A user who types the block cannot forge it.
    assert strip_server_injected_tags(notice + "hello") == "hello"


def test_render_of_nothing_is_empty():
    assert render_seen_capabilities(seen_capabilities(_session())) == ""


def test_sdk_prepend_lands_the_notice_on_a_later_user_turn():
    session = _session()
    session.messages.extend(
        [
            ChatMessage(role="user", content="first"),
            _assistant(_call("describe_capability", {"id": "block:abc"}, "c1")),
            _result("c1", {"type": "block_details"}),
            ChatMessage(role="user", content="second"),
        ]
    )

    result = _maybe_prepend_seen_capabilities(session, True, "second")

    assert result.startswith(f"<{SEEN_CAPABILITIES_TAG}>")
    assert result.endswith("second")
    assert "block:abc" in result
    # Query-only: nothing was written into the history rows.
    assert all(SEEN_CAPABILITIES_TAG not in (m.content or "") for m in session.messages)


def test_sdk_prepend_is_a_noop_for_a_first_turn_and_for_tool_turns():
    session = _session()
    session.messages.append(ChatMessage(role="user", content="first"))
    assert _maybe_prepend_seen_capabilities(session, True, "first") == "first"

    session.messages.extend(
        [
            _assistant(_call("describe_capability", {"id": "block:abc"}, "c1")),
            _result("c1", {"type": "block_details"}),
        ]
    )
    assert _maybe_prepend_seen_capabilities(session, False, "raw") == "raw"


def test_baseline_prepend_puts_the_notice_on_the_current_user_message():
    session = _session()
    session.messages.extend(
        [
            _assistant(_call("read_skill", {"name": "default-developer"}, "c1")),
            _result("c1", {"type": "skill"}),
        ]
    )
    openai_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]

    _prepend_skills_notice_to_current_message(
        openai_messages, build_seen_capabilities_notice(session)
    )

    assert openai_messages[0]["content"] == "first"
    assert openai_messages[2]["content"].startswith(f"<{SEEN_CAPABILITIES_TAG}>")
    assert openai_messages[2]["content"].endswith("second")
    assert "default-developer" in openai_messages[2]["content"]
