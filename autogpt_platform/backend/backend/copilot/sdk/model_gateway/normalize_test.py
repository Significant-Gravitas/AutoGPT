import pytest

from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.neutral import (
    DocumentPart,
    ImagePart,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
)
from backend.copilot.sdk.model_gateway.normalize import (
    normalize_request,
    reminder,
    strip_billing_header,
    system_text,
)
from backend.copilot.sdk.model_gateway.testing import cli_requests


def test_the_billing_line_is_dropped_and_the_rest_of_the_system_joined():
    request = normalize_request(cli_requests.first_request())
    assert request.system == f"{cli_requests.SDK_LINE}\n\n{cli_requests.SYSTEM_PROMPT}"
    assert "billing" not in request.system


def test_the_billing_line_is_dropped_in_either_system_shape():
    assert strip_billing_header(cli_requests.BILLING_LINE) == ""
    assert strip_billing_header("plain prompt") == "plain prompt"
    blocks = [
        {"type": "text", "text": "  " + cli_requests.BILLING_LINE},
        {"type": "text", "text": "kept"},
    ]
    assert strip_billing_header(blocks) == [{"type": "text", "text": "kept"}]
    assert system_text(None) == ""


def test_the_cli_system_notes_become_reminders_in_the_user_turn_they_follow():
    request = normalize_request(cli_requests.first_request())
    (turn,) = request.messages
    assert turn.role == "user"
    assert turn.parts == [
        TextPart(text=cli_requests.ASK),
        TextPart(text=reminder(cli_requests.ENVIRONMENT)),
    ]


def test_the_compaction_request_keeps_results_instruction_and_hook_note_in_order():
    request = normalize_request(cli_requests.compaction_request())
    user, assistant, last = request.messages
    assert user.role == last.role == "user" and assistant.role == "assistant"
    results = [p for p in last.parts if isinstance(p, ToolResultPart)]
    assert [r.call_id for r in results] == ["toolu_A1", "toolu_B2"]
    assert results[0].text == "Memory: the team chose HubSpot."
    texts = [p.text for p in last.parts if isinstance(p, TextPart)]
    assert texts == [cli_requests.SUMMARY_INSTRUCTION, reminder(cli_requests.HOOK_NOTE)]


def test_the_assistant_turn_keeps_reasoning_text_and_calls_in_order():
    request = normalize_request(cli_requests.continuation_request("sig-x"))
    assistant = request.messages[1]
    assert assistant.parts == [
        ReasoningPart(text="I should look.", signature="sig-x"),
        TextPart(text="Searching first.\n\n"),
        ToolCallPart(
            id="toolu_A1",
            name="mcp__copilot__memory_search",
            arguments={"query": "CRM"},
        ),
        ToolCallPart(
            id="toolu_B2", name="mcp__copilot__read_doc", arguments={"page": 1}
        ),
    ]


def test_tools_choice_limits_and_session_are_read():
    body = cli_requests.first_request() | {
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        "temperature": 0.2,
        "stop_sequences": ["END"],
    }
    request = normalize_request(body)
    assert [t.name for t in request.tools] == [
        "mcp__copilot__memory_search",
        "mcp__copilot__read_doc",
    ]
    assert request.tools[1].parameters["required"] == ["page"]
    assert request.tool_choice == ToolChoice(mode="any", parallel=False)
    assert request.max_tokens == 32000
    assert (request.temperature, request.stop) == (0.2, ["END"])
    assert request.session_id == cli_requests.SESSION_ID


def test_server_tools_are_left_out():
    body = cli_requests.first_request()
    body["tools"].append({"type": "web_search_20250305", "name": "web_search"})
    assert len(normalize_request(body).tools) == 2


def test_images_and_documents_in_user_turns_and_tool_results():
    image = {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x/y.png"},
                    },
                    {"type": "image", "source": {"type": "file", "file_id": "f_1"}},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "UERG",
                        },
                    },
                    {
                        "type": "document",
                        "source": {"type": "text", "data": "plain doc"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "is_error": True,
                        "content": [
                            {"type": "image", "source": image},
                            {"type": "text", "text": "shot"},
                        ],
                    },
                ],
            }
        ]
    }
    (turn,) = normalize_request(body).messages
    assert turn.parts[:5] == [
        ImagePart(url="data:image/jpeg;base64,QUJD"),
        TextPart(text="[image omitted: its source cannot be forwarded]"),
        TextPart(text="[image omitted: its source cannot be forwarded]"),
        DocumentPart(media_type="application/pdf", data="UERG"),
        TextPart(text="plain doc"),
    ]
    result = turn.parts[5]
    assert isinstance(result, ToolResultPart)
    assert result.is_error and result.text == "shot"
    assert result.images == [ImagePart(url="data:image/jpeg;base64,QUJD")]


def test_a_string_tool_result_and_an_empty_turn():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "ok"}
                ],
            },
            {"role": "user", "content": []},
            {"role": "system", "content": ""},
        ]
    }
    (turn,) = normalize_request(body).messages
    assert turn.parts == [ToolResultPart(call_id="t", content=[TextPart(text="ok")])]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": "not a list"},
        {"messages": ["not an object"]},
        {"messages": [{"role": "tool", "content": "x"}]},
    ],
)
def test_a_malformed_body_is_a_400(body):
    with pytest.raises(GatewayError) as caught:
        normalize_request(body)
    assert caught.value.status == 400


def test_a_session_id_is_only_taken_from_well_formed_metadata():
    for metadata in (None, {"user_id": "not json"}, {"user_id": '{"session_id": 3}'}):
        body = {"messages": [], "metadata": metadata}
        assert normalize_request(body).session_id is None
