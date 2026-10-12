from typing import Any

from backend.copilot.sdk.model_gateway.adapters.openai_request import chat_payload
from backend.copilot.sdk.model_gateway.normalize import normalize_request, reminder
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.testing import cli_requests as cli
from backend.copilot.sdk.model_gateway.tool_names import upstream_tool_name


def _route(**overrides: Any) -> GatewayRoute:
    return GatewayRoute(
        adapter="openai_chat",
        model="qwen-local",
        base_url="http://127.0.0.1:8080/v1",
        **overrides,
    )


def _payload(body: dict[str, Any], **route: Any) -> dict[str, Any]:
    payload, _ = chat_payload(normalize_request(body), _route(**route))
    return payload


def _roles(payload: dict[str, Any]) -> list[str]:
    return [m["role"] for m in payload["messages"]]


def test_the_first_request_is_system_then_one_user_turn_with_its_reminder():
    payload = _payload(cli.first_request())

    assert payload["model"] == "qwen-local"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["messages"] == [
        {"role": "system", "content": f"{cli.SDK_LINE}\n\n{cli.SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{cli.ASK}\n\n{reminder(cli.ENVIRONMENT)}"},
    ]
    assert "billing" not in str(payload)


def test_the_compaction_request_lands_as_tool_messages_then_the_instruction():
    payload = _payload(cli.compaction_request())

    assert _roles(payload) == ["system", "user", "assistant", "tool", "tool", "user"]
    assistant = payload["messages"][2]
    assert assistant["content"] == "Searching first.\n\n"
    assert [c["id"] for c in assistant["tool_calls"]] == ["toolu_A1", "toolu_B2"]
    assert assistant["tool_calls"][1]["function"] == {
        "name": "mcp__copilot__read_doc",
        "arguments": '{"page": 1}',
    }
    tools = payload["messages"][3:5]
    assert tools == [
        {
            "role": "tool",
            "tool_call_id": "toolu_A1",
            "content": "Memory: the team chose HubSpot.",
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_B2",
            "content": "HANDBOOK PAGE 1: amber-7.",
        },
    ]
    last = payload["messages"][5]["content"]
    assert last == f"{cli.SUMMARY_INSTRUCTION}\n\n{reminder(cli.HOOK_NOTE)}"


def test_reasoning_is_dropped_on_the_way_back():
    payload = _payload(cli.continuation_request("any-signature"))
    assert "I should look." not in str(payload)
    assert "any-signature" not in str(payload)


def test_a_hook_note_after_results_follows_the_tool_messages():
    payload = _payload(cli.continuation_request())
    assert _roles(payload)[-3:] == ["tool", "tool", "user"]
    assert payload["messages"][-1]["content"] == reminder(cli.HOOK_NOTE)


def test_images_are_data_url_parts_and_a_tool_result_s_images_follow_it():
    image = {"type": "base64", "media_type": "image/png", "data": "iVBOR"}
    body = cli.envelope(
        [
            *cli.tool_round(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_A1",
                        "content": [{"type": "image", "source": image}],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_B2",
                        "content": "page",
                    },
                    {"type": "text", "text": "Look at this too:"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://img/x.png"},
                    },
                ],
            },
        ]
    )
    messages = _payload(body)["messages"]

    assert messages[-3] == {
        "role": "tool",
        "tool_call_id": "toolu_A1",
        "content": "[the tool returned an image; it follows in the next message]",
    }
    assert messages[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this too:"},
            {"type": "text", "text": "[image omitted: its source cannot be forwarded]"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
        ],
    }


def test_tools_choice_and_limits_are_mapped():
    body = cli.first_request() | {
        "tool_choice": {
            "type": "tool",
            "name": "mcp__copilot__read_doc",
            "disable_parallel_tool_use": True,
        },
        "temperature": 0.3,
        "top_p": 0.9,
        "stop_sequences": ["###"],
    }
    payload = _payload(body)

    assert payload["tools"][0] == {
        "type": "function",
        "function": {
            "name": "mcp__copilot__memory_search",
            "description": "Search what the user told AutoPilot in earlier chats.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "mcp__copilot__read_doc"},
    }
    assert payload["parallel_tool_calls"] is False
    assert (
        payload["max_tokens"],
        payload["temperature"],
        payload["top_p"],
        payload["stop"],
    ) == (32000, 0.3, 0.9, ["###"])


def test_the_effort_is_the_route_s_never_the_cli_default():
    assert "reasoning_effort" not in _payload(cli.first_request())
    assert _payload(cli.first_request(), effort="low")["reasoning_effort"] == "low"


def test_any_and_none_choices():
    for mode, expected in (("any", "required"), ("none", "none"), ("auto", "auto")):
        body = cli.first_request() | {"tool_choice": {"type": mode}}
        assert _payload(body)["tool_choice"] == expected


def test_names_the_upstream_cannot_take_are_replaced_and_mapped_back():
    long_name = "mcp__user_server__" + "x" * 70
    body = cli.first_request()
    body["tools"].append(
        {"name": long_name, "description": "d", "input_schema": {"type": "object"}}
    )
    payload, names = chat_payload(normalize_request(body), _route())

    stand_in = upstream_tool_name(long_name)
    assert payload["tools"][-1]["function"]["name"] == stand_in
    assert payload["tools"][-1]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }
    assert names[stand_in] == long_name


def test_a_document_is_named_rather_than_silently_lost():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "UERG",
                        },
                    }
                ],
            }
        ]
    }
    assert _payload(body)["messages"] == [
        {
            "role": "user",
            "content": "[document omitted: this model server cannot read files]",
        }
    ]
