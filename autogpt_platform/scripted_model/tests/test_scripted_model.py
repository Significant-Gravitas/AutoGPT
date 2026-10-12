"""The scripted model, spoken to through the clients the platform itself uses.

The Anthropic client parses a Messages stream as the Claude Agent SDK's CLI
does, and the OpenAI client is the baseline engine's own. A script that round
trips through both is one the copilot can follow.
"""

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import anthropic
import openai
import pytest

from scripted_model.scripts import parse_script

JOURNEY_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/playwright/journeys/model-scripts"
)
SONNET = "anthropic/claude-sonnet-5-5"

LOOKUP = {
    "title": "Haiku lookup",
    "turns": [
        {
            "text": "Let me check your library first.",
            "tool_calls": [{"name": "find_library_agent", "input": {"query": "haiku"}}],
            "usage": {
                "input_tokens": 3000,
                "output_tokens": 40,
                "cache_read_input_tokens": 20000,
                "cache_creation_input_tokens": 1000,
            },
        },
        {
            "expect_tool_results": [
                {"tool": "find_library_agent", "contains": "No agents"}
            ],
            "text": "You have none yet.",
            "usage": {"input_tokens": 3100, "output_tokens": 8},
        },
    ],
}
SDK_TOOLS = [
    {
        "name": "mcp__copilot__find_library_agent",
        "description": "Search the user's library.",
        "input_schema": {"type": "object"},
    }
]
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {"name": "find_library_agent", "parameters": {"type": "object"}},
    }
]


def test_messages_follow_the_script_through_a_tool_round(scripted_model):
    scripted_model.add_script("lookup", LOOKUP)
    client = _anthropic(scripted_model)
    # The CLI marks its cached prefix; an hour-long one prices writes higher.
    system = [
        {
            "type": "text",
            "text": "You are AutoPilot.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    messages = [
        {"role": "user", "content": "Any haiku agent? [[script:lookup#sdk-run]]"}
    ]

    first = _stream(client, messages, system)
    call = first.content[1]
    assert first.content[0].text == "Let me check your library first."
    assert (call.name, call.input) == (
        "mcp__copilot__find_library_agent",
        {"query": "haiku"},
    )
    assert first.stop_reason == "tool_use"
    assert (
        first.usage.input_tokens,
        first.usage.output_tokens,
        first.usage.cache_read_input_tokens,
        first.usage.cache_creation.ephemeral_1h_input_tokens,
    ) == (3000, 40, 20000, 1000)

    messages += [
        {"role": "assistant", "content": [b.model_dump() for b in first.content]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": "No agents match 'haiku'.",
                }
            ],
        },
    ]
    second = _stream(client, messages, system)
    assert [b.text for b in second.content] == ["You have none yet."]
    assert second.stop_reason == "end_turn"

    first_entry, second_entry = scripted_model.journal(nonce="sdk-run")
    assert first_entry["cost_usd"] == pytest.approx(
        (3000 * 2 + 40 * 10 + 20000 * 0.2 + 1000 * 4.0) / 1_000_000
    )
    assert second_entry["checks"] == [
        {"what": "find_library_agent result", "ok": True, "detail": "matched"}
    ]
    assert first_entry["completed"] and second_entry["completed"]


def test_chat_completions_follow_the_script_and_bill_like_openrouter(
    scripted_model,
):
    scripted_model.add_script("lookup", LOOKUP)
    client = _openai(scripted_model)
    messages: list[dict] = [
        {"role": "user", "content": "Any haiku agent? [[script:lookup#chat-run]]"}
    ]

    text, calls, usage = _stream_chat(client, messages)
    assert text == "Let me check your library first."
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [
        ("find_library_agent", {"query": "haiku"})
    ]
    assert (usage.prompt_tokens, usage.completion_tokens) == (24000, 40)
    # No cache marker asked for an hour, so the write is a five-minute one.
    assert usage.model_extra["cost"] == pytest.approx(
        (3000 * 2 + 40 * 10 + 20000 * 0.2 + 1000 * 2.5) / 1_000_000
    )

    messages += [
        {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": calls[0]["id"],
                    "type": "function",
                    "function": {
                        "name": calls[0]["name"],
                        "arguments": calls[0]["arguments"],
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": calls[0]["id"],
            "content": "No agents match 'haiku'.",
        },
    ]
    text, calls, _ = _stream_chat(client, messages)
    assert (text, calls) == ("You have none yet.", [])
    assert scripted_model.journal(nonce="chat-run")[-1]["checks"][0]["ok"]


def test_a_failed_expectation_is_recorded_with_what_the_tool_said(scripted_model):
    scripted_model.add_script("lookup", LOOKUP)
    messages = [
        {"role": "user", "content": "[[script:lookup#failed-check]]"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "x__find_library_agent"}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Search is down.",
                    "is_error": True,
                }
            ],
        },
    ]
    _stream(_anthropic(scripted_model), messages)
    (entry,) = scripted_model.journal(nonce="failed-check")
    assert entry["checks"] == [
        {
            "what": "find_library_agent result",
            "ok": False,
            "detail": "'find_library_agent' returned is_error=True: Search is down.",
        }
    ]


def test_a_follow_up_message_gets_the_next_turn(scripted_model):
    """A message the user queued while a turn ran arrives after that turn's
    answer, and the script's next turn answers it."""
    scripted_model.add_script(
        "follow-up",
        {
            "turns": [
                {
                    "text": "First answer.",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                {
                    "expect_user_text": "one more thing",
                    "text": "Second answer.",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ]
        },
    )
    client = _anthropic(scripted_model)
    for nonce, follow_up in (("asked", "And one more thing?"), ("forgot", "Thanks!")):
        messages = [
            {"role": "user", "content": f"[[script:follow-up#{nonce}]]"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": follow_up},
        ]
        assert [b.text for b in _stream(client, messages).content] == ["Second answer."]

    (asked,) = scripted_model.journal(nonce="asked")
    (forgot,) = scripted_model.journal(nonce="forgot")
    assert asked["checks"] == [
        {"what": "user message", "ok": True, "detail": "matched"}
    ]
    assert forgot["checks"] == [
        {"what": "user message", "ok": False, "detail": "the user said: 'Thanks!'"}
    ]


def test_a_request_the_script_cannot_answer_fails_with_the_reason(scripted_model):
    scripted_model.add_script("lookup", LOOKUP)
    client = _anthropic(scripted_model)
    one_round = [
        {"role": "user", "content": "[[script:lookup]]"},
        {"role": "assistant", "content": "Looking."},
        {"role": "user", "content": "And?"},
        {"role": "assistant", "content": "None."},
        {"role": "user", "content": "Once more?"},
    ]
    cases = [
        ("hello", SDK_TOOLS, "no [[script:<name>]] marker"),
        (
            "[[script:nope]]",
            SDK_TOOLS,
            "no script named 'nope'; known scripts: ['lookup']",
        ),
        (
            "[[script:lookup]]",
            [{**SDK_TOOLS[0], "name": "mcp__copilot__other"}],
            "did not offer",
        ),
        (one_round, SDK_TOOLS, "has 2 turn(s), but the platform asked for turn 3"),
    ]
    for content, tools, reason in cases:
        messages = (
            content
            if isinstance(content, list)
            else [{"role": "user", "content": content}]
        )
        with pytest.raises(anthropic.BadRequestError, match=re.escape(reason)):
            client.messages.create(
                model=SONNET, max_tokens=64, messages=messages, tools=tools
            )

    malformed = urllib.request.Request(
        f"{scripted_model.url}/api/v1/messages",
        data=json.dumps({"messages": ["not an object"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(malformed, timeout=5)
    assert refused.value.code == 400
    assert "malformed request" in refused.value.read().decode()

    with pytest.raises(openai.BadRequestError, match="no script named 'nope'"):
        _openai(scripted_model).chat.completions.create(
            model=SONNET,
            messages=[{"role": "user", "content": "[[script:nope]]"}],
            tools=CHAT_TOOLS,
        )
    assert [e["kind"] for e in scripted_model.journal()] == ["error"] * 5


def test_a_side_call_gets_the_script_title(scripted_model):
    scripted_model.add_script("lookup", LOOKUP)
    client = _openai(scripted_model)

    titled = client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Title: [[script:lookup#side]]"}],
    )
    untitled = client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Title this chat."}],
    )

    assert titled.choices[0].message.content == "Haiku lookup"
    assert untitled.choices[0].message.content == "Scripted reply"
    assert titled.usage.model_extra["cost"] == pytest.approx((200 * 1 + 10 * 5) / 1e6)
    assert scripted_model.journal(nonce="side")[0]["kind"] == "side_call"


def test_concurrent_runs_of_one_script_each_get_their_own_turns(scripted_model):
    """Every request carries its whole conversation, so runs sharing a script
    land on their own turns however their requests interleave."""
    scripted_model.add_script("lookup", LOOKUP)
    runs = 8
    # Every run's first turn happens before any run's second turn.
    between_turns = threading.Barrier(runs)
    failures: list[BaseException] = []

    def run(nonce: str) -> None:
        try:
            client = _anthropic(scripted_model)
            messages = [{"role": "user", "content": f"[[script:lookup#{nonce}]]"}]
            first = _stream(client, messages)
            between_turns.wait(timeout=10)
            call = first.content[1]
            messages += [
                {
                    "role": "assistant",
                    "content": [b.model_dump() for b in first.content],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": f"No agents for {nonce}.",
                        }
                    ],
                },
            ]
            _stream(client, messages)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run, args=(f"run-{i}",)) for i in range(runs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert failures == []
    for i in range(runs):
        entries = scripted_model.journal(nonce=f"run-{i}")
        assert [e["turn"] for e in entries] == [1, 2]
        assert entries[1]["tool_results"][0]["content"] == f"No agents for run-{i}."


@pytest.mark.parametrize(
    "path", sorted(JOURNEY_SCRIPTS.glob("*.json")), ids=lambda path: path.stem
)
def test_every_journey_script_is_valid(path: Path):
    parse_script(path.stem, json.loads(path.read_text(encoding="utf-8")))


def _anthropic(server) -> anthropic.Anthropic:
    # Like the CLI: CHAT_BASE_URL without its /v1.
    return anthropic.Anthropic(
        base_url=f"{server.url}/api", api_key="any", max_retries=0
    )


def _openai(server) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{server.url}/api/v1", api_key="any", max_retries=0)


def _stream(client: anthropic.Anthropic, messages: list[dict], system=None):
    extra = {"system": system} if system else {}
    with client.messages.stream(
        model=SONNET, max_tokens=1024, messages=messages, tools=SDK_TOOLS, **extra
    ) as stream:
        return stream.get_final_message()


def _stream_chat(client: openai.OpenAI, messages: list[dict]):
    stream = client.chat.completions.create(
        model=SONNET,
        messages=messages,
        tools=CHAT_TOOLS,
        stream=True,
        stream_options={"include_usage": True},
    )
    text, calls, usage = "", {}, None
    for chunk in stream:
        usage = chunk.usage or usage
        for choice in chunk.choices:
            text += choice.delta.content or ""
            for call in choice.delta.tool_calls or []:
                entry = calls.setdefault(
                    call.index, {"id": "", "name": "", "arguments": ""}
                )
                entry["id"] = call.id or entry["id"]
                if call.function:
                    entry["name"] += call.function.name or ""
                    entry["arguments"] += call.function.arguments or ""
    return text, [calls[i] for i in sorted(calls)], usage
