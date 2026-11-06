"""Request bodies in the exact shape CLI 2.1.284 sends for an unknown model.

Rebuilt from the requests captured on the wire during the shim research
(``shim-research.md`` section 3), with the content replaced: the system list
led by the billing line, the environment block and hook output as
``role: "system"`` messages, adaptive thinking with ``output_config``, and
the session id inside ``metadata.user_id``.

``compaction_request`` is the shape that broke the old Codex bridge: the
CLI compacts mid-turn by appending its summary instruction to the user
message that already holds the round's tool results, then sends the hook
note after it.
"""

import json
from typing import Any

SESSION_ID = "0e8aadc9-80e0-4f02-9267-805e2128e742"
BILLING_LINE = (
    "x-anthropic-billing-header: cc_version=2.1.284.d86; cc_entrypoint=sdk-py;"
)
SDK_LINE = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
SYSTEM_PROMPT = "You are AutoPilot, the assistant inside the AutoGPT platform."
ENVIRONMENT = "# Environment\nYou have been invoked in the following environment:"
HOOK_NOTE = (
    "PostToolUse:mcp__copilot__read_doc hook additional context: "
    "AUTOPILOT-HOOK-NOTE: budget is fine; keep going."
)
SUMMARY_INSTRUCTION = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n"
    "Your task is to create a detailed summary of the conversation so far."
)
ASK = "Find the CRM with memory_search, then read handbook pages 1 and 2."


def tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "mcp__copilot__memory_search",
            "description": "Search what the user told AutoPilot in earlier chats.",
            "input_schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "mcp__copilot__read_doc",
            "description": "Read one page of the handbook.",
            "input_schema": {
                "type": "object",
                "properties": {"page": {"type": "integer"}},
                "required": ["page"],
            },
        },
    ]


def envelope(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The fields the CLI sends around ``messages`` on every main-loop call."""
    return {
        "model": "local-model",
        "messages": messages,
        "system": [
            {"type": "text", "text": BILLING_LINE},
            {"type": "text", "text": SDK_LINE, "cache_control": {"type": "ephemeral"}},
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "tools": tools(),
        "metadata": {
            "user_id": json.dumps(
                {"device_id": "d" * 64, "account_uuid": "", "session_id": SESSION_ID}
            )
        },
        "max_tokens": 32000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "stream": True,
    }


def first_request() -> dict[str, Any]:
    return envelope(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ASK,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "system", "content": ENVIRONMENT},
        ]
    )


def tool_round(signature: str = "") -> list[dict[str, Any]]:
    """The model's first round: reasoning, a sentence and two tool calls."""
    return [
        {"role": "user", "content": ASK},
        {"role": "system", "content": ENVIRONMENT},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should look.",
                    "signature": signature,
                },
                {"type": "text", "text": "Searching first.\n\n"},
                {
                    "type": "tool_use",
                    "id": "toolu_A1",
                    "name": "mcp__copilot__memory_search",
                    "input": {"query": "CRM"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_B2",
                    "name": "mcp__copilot__read_doc",
                    "input": {"page": 1},
                },
            ],
        },
    ]


def results(*extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_A1",
                "content": [
                    {"type": "text", "text": "Memory: the team chose HubSpot."}
                ],
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_B2",
                "content": [{"type": "text", "text": "HANDBOOK PAGE 1: amber-7."}],
                "cache_control": {"type": "ephemeral"},
            },
            *extra,
        ],
    }


def continuation_request(signature: str = "") -> dict[str, Any]:
    """The round's tool results going back, then the hook note."""
    return envelope(
        [*tool_round(signature), results(), {"role": "system", "content": HOOK_NOTE}]
    )


def compaction_request(signature: str = "") -> dict[str, Any]:
    """The mid-turn compaction: tool results and the summary instruction in
    one user message, the hook note after it."""
    instruction = {"type": "text", "text": SUMMARY_INSTRUCTION}
    return envelope(
        [
            *tool_round(signature),
            results(instruction),
            {"role": "system", "content": HOOK_NOTE},
        ]
    )
