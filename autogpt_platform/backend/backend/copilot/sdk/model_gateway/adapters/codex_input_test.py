from typing import Any

from backend.copilot.sdk.model_gateway.adapters.codex_input import responses_payload
from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    ReasoningIdentity,
    reasoning_tag,
    seal,
)
from backend.copilot.sdk.model_gateway.normalize import normalize_request, reminder
from backend.copilot.sdk.model_gateway.testing import cli_requests as cli
from backend.copilot.sdk.model_gateway.testing.codex_lease import codex_route

_TAG = reasoning_tag("acct-test", "gpt-test")
_ENCRYPTED = "gAAAAABo-encrypted-" + "r" * 3000


def _payload(body: dict[str, Any], tag: ReasoningIdentity = _TAG) -> dict[str, Any]:
    payload, _ = responses_payload(
        normalize_request(body), codex_route("gpt-test"), tag
    )
    return payload


def test_every_call_is_stateless_and_asks_for_encrypted_reasoning():
    payload = _payload(cli.first_request())

    assert payload["model"] == "gpt-test"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"summary": "auto", "effort": "medium"}
    assert payload["instructions"] == f"{cli.SDK_LINE}\n\n{cli.SYSTEM_PROMPT}"
    assert payload["prompt_cache_key"] == cli.SESSION_ID
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": cli.ASK},
                {"type": "input_text", "text": reminder(cli.ENVIRONMENT)},
            ],
        }
    ]
    for unsupported in ("max_output_tokens", "max_tokens", "temperature", "stop"):
        assert unsupported not in payload


def test_the_whole_history_goes_back_with_real_roles_and_paired_calls():
    payload = _payload(cli.continuation_request(seal(_ENCRYPTED, _TAG)))
    kinds = [(item["type"], item.get("role")) for item in payload["input"]]

    assert kinds == [
        ("message", "user"),
        ("reasoning", None),
        ("message", "assistant"),
        ("function_call", None),
        ("function_call", None),
        ("function_call_output", None),
        ("function_call_output", None),
        ("message", "user"),
    ]
    reasoning, assistant, first_call = payload["input"][1:4]
    assert reasoning == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": _ENCRYPTED,
    }
    assert assistant["content"] == [
        {"type": "output_text", "text": "Searching first.\n\n"}
    ]
    assert first_call == {
        "type": "function_call",
        "call_id": "toolu_A1",
        "name": "mcp__copilot__memory_search",
        "arguments": '{"query": "CRM"}',
    }
    outputs = payload["input"][5:7]
    assert [o["call_id"] for o in outputs] == ["toolu_A1", "toolu_B2"]
    assert outputs[0]["output"] == "Memory: the team chose HubSpot."
    assert payload["input"][-1]["content"] == [
        {"type": "input_text", "text": reminder(cli.HOOK_NOTE)}
    ]


def test_the_compaction_request_reaches_codex_with_results_then_instruction():
    payload = _payload(cli.compaction_request(seal(_ENCRYPTED, _TAG)))
    *_, last_output, instruction = payload["input"]
    assert last_output["type"] == "function_call_output"
    assert instruction == {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": cli.SUMMARY_INSTRUCTION},
            {"type": "input_text", "text": reminder(cli.HOOK_NOTE)},
        ],
    }


def test_reasoning_from_another_source_is_left_out():
    for signature in (
        "",
        "EqQBCkYIBRgCIkDanthropic",
        seal(_ENCRYPTED, reasoning_tag("other-account", "gpt-test")),
    ):
        items = _payload(cli.continuation_request(signature))["input"]
        assert all(item["type"] != "reasoning" for item in items)


def test_images_go_as_input_images_in_turns_and_tool_outputs():
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
                        "content": [
                            {"type": "text", "text": "shot:"},
                            {"type": "image", "source": image},
                        ],
                    },
                    {"type": "tool_result", "tool_use_id": "toolu_B2", "content": "ok"},
                    {"type": "image", "source": image},
                ],
            },
        ]
    )
    items = _payload(body)["input"]
    outputs = [i for i in items if i["type"] == "function_call_output"]
    assert outputs[0]["output"] == [
        {"type": "input_text", "text": "shot:"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"},
    ]
    assert items[-1]["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}
    ]


def test_tools_are_functions_with_non_strict_schemas():
    payload = _payload(cli.first_request() | {"tool_choice": {"type": "any"}})
    assert payload["tools"][0] == {
        "type": "function",
        "name": "mcp__copilot__memory_search",
        "description": "Search what the user told AutoPilot in earlier chats.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "strict": False,
    }
    assert payload["tool_choice"] == "required"
    assert "parallel_tool_calls" not in payload


def test_a_named_tool_choice_and_no_parallel_calls():
    body = cli.first_request() | {
        "tool_choice": {
            "type": "tool",
            "name": "mcp__copilot__read_doc",
            "disable_parallel_tool_use": True,
        }
    }
    payload = _payload(body)
    assert payload["tool_choice"] == {
        "type": "function",
        "name": "mcp__copilot__read_doc",
    }
    assert payload["parallel_tool_calls"] is False
