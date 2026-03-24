"""Tests for OrchestratorBlock compatibility with the OpenAI Responses API.

The OrchestratorBlock manages conversation history in the Chat Completions
format, but OpenAI models now use the Responses API which has a fundamentally
different conversation structure.  These tests document:

 - Every branch of the affected helper functions
 - Which branches PASS (existing Chat Completions / Anthropic paths)
 - Which branches FAIL (Responses API paths) — marked ``xfail``

Bug report:  When using gpt-5.2 (or any model via ``client.responses.create``),
the agent-mode loop fails on the second LLM call with::

    Error code: 400 - Invalid value: ''.
    Supported values are: 'assistant', 'system', 'developer', and 'user'.

Root cause:  ``raw_response`` is the entire ``Response`` object.  Serialising it
produces a dict with no ``role`` field.  Tool results are also formatted as
``{"role": "tool", …}`` (Chat Completions) instead of
``{"type": "function_call_output", …}`` (Responses API).
"""

import threading
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.orchestrator import (
    OrchestratorBlock,
    _combine_tool_responses,
    _convert_raw_response_to_dict,
    _create_tool_response,
    _get_tool_requests,
    _get_tool_responses,
    get_pending_tool_calls,
)
from backend.data.execution import ExecutionContext

# ───────────────────────────────────────────────────────────────────────────
# Mock objects that mirror the OpenAI Responses API SDK types
# ───────────────────────────────────────────────────────────────────────────


class _MockOutputText:
    """openai.types.responses.ResponseOutputText"""

    def __init__(self, text: str):
        self.type = "output_text"
        self.text = text
        self.annotations: list = []
        self.logprobs = None


class _MockOutputMessage:
    """openai.types.responses.ResponseOutputMessage"""

    def __init__(self, text: str, msg_id: str = "msg_abc123"):
        self.type = "message"
        self.id = msg_id
        self.role = "assistant"
        self.status = "completed"
        self.content = [_MockOutputText(text)]


class _MockFunctionCall:
    """openai.types.responses.ResponseFunctionToolCall"""

    def __init__(
        self,
        name: str,
        arguments: str,
        call_id: str = "call_abc123",
        fc_id: str = "fc_abc123",
    ):
        self.type = "function_call"
        self.id = fc_id
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.status = "completed"


class _MockUsage:
    def __init__(self, inp: int = 100, out: int = 50):
        self.input_tokens = inp
        self.output_tokens = out


class _MockResponse:
    """openai.types.responses.Response — top-level object returned by
    ``client.responses.create()``.  Stored as ``raw_response`` on the
    LLMResponse (llm.py line 848).
    """

    def __init__(self, output: list, resp_id: str = "resp_abc123"):
        self.id = resp_id
        self.object = "response"
        self.model = "gpt-5.2-2025-12-11"
        self.output = output
        self.usage = _MockUsage()
        self.user = None
        self.error = None
        self.status = "completed"
        self.store = False
        self.metadata: dict[str, Any] = {}
        self.output_text = self._text()

    def _text(self) -> str:
        for item in self.output:
            if getattr(item, "type", None) == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        return c.text
        return ""


# ───────────────────────────────────────────────────────────────────────────
# _convert_raw_response_to_dict  (lines 180-193)
# Branches:  str → dict wrapper  |  dict → pass-through  |  else → to_dict
# ───────────────────────────────────────────────────────────────────────────


class TestConvertRawResponseToDict:
    # -- passing branches --------------------------------------------------

    def test_string_input_wraps_as_assistant(self):
        """Branch 1 (str): Ollama returns a plain string."""
        result = _convert_raw_response_to_dict("hello")
        assert result == {"role": "assistant", "content": "hello"}

    def test_dict_input_passes_through(self):
        """Branch 2 (dict): Already a dict — returned as-is."""
        d = {"role": "assistant", "content": "hi", "extra": 42}
        result = _convert_raw_response_to_dict(d)
        assert result is d

    def test_chat_completion_message_object(self):
        """Branch 3 (else): A ChatCompletionMessage-like object with .role.

        Uses a Pydantic BaseModel to match how the OpenAI SDK structures
        ChatCompletionMessage objects (jsonable_encoder handles Pydantic).
        """
        from pydantic import BaseModel

        class _Msg(BaseModel):
            role: str = "assistant"
            content: str | None = "text"
            tool_calls: list | None = None

        result = _convert_raw_response_to_dict(_Msg())
        assert isinstance(result, dict)
        assert result.get("role") == "assistant"

    # -- failing branches (Responses API) -----------------------------------

    def test_responses_api_text_response_has_role(self):
        """Branch 3 (else): A Responses API Response with a text message.

        The serialised dict must have ``role`` so it can be used as a
        conversation input item.  Currently it does NOT.
        """
        resp = _MockResponse(output=[_MockOutputMessage("Hello!")])
        result = _convert_raw_response_to_dict(resp)

        # After the fix, the result should either:
        # - be a list of output items each with valid role/type, OR
        # - be a single dict with role="assistant"
        if isinstance(result, list):
            for item in result:
                assert item.get("role") or item.get("type")
        else:
            assert result.get("role") == "assistant"

    def test_responses_api_function_call_has_valid_type(self):
        """Branch 3 (else): A Responses API Response with a function_call.

        The serialised output must produce items with ``type: function_call``
        that the Responses API accepts as input.
        """
        resp = _MockResponse(
            output=[_MockFunctionCall("my_tool", '{"x": 1}', call_id="call_xyz")]
        )
        result = _convert_raw_response_to_dict(resp)

        if isinstance(result, list):
            fc_items = [i for i in result if i.get("type") == "function_call"]
            assert len(fc_items) == 1
            assert fc_items[0]["call_id"] == "call_xyz"
        else:
            # If it remains a single dict it must at least be a valid input item
            assert (
                result.get("type") == "function_call"
                or result.get("role") == "assistant"
            )

    def test_responses_api_mixed_output_items(self):
        """Branch 3 (else): Response with both a message and a function_call."""
        resp = _MockResponse(
            output=[
                _MockOutputMessage("Thinking…"),
                _MockFunctionCall("tool_a", "{}", call_id="call_111"),
            ]
        )
        result = _convert_raw_response_to_dict(resp)

        if isinstance(result, list):
            assert len(result) == 2
        else:
            # A single dict is wrong — there are two distinct items
            pytest.fail("Expected a list of output items, got a single dict")


# ───────────────────────────────────────────────────────────────────────────
# _get_tool_requests  (lines 61-86)
# Branches:
#   role != assistant → early return
#   tool_calls is list → iterate → id present / absent
#   content is list → iterate → type==tool_use / not → id present / absent
# ───────────────────────────────────────────────────────────────────────────


class TestGetToolRequests:
    # -- passing branches (Chat Completions) --------------------------------

    def test_openai_chat_completions_with_tool_calls(self):
        """role=assistant + tool_calls list → extracts IDs."""
        entry = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function"},
                {"id": "call_2", "type": "function"},
            ],
        }
        assert _get_tool_requests(entry) == ["call_1", "call_2"]

    def test_openai_chat_completions_tool_call_missing_id(self):
        """tool_call entry without 'id' → skipped."""
        entry = {
            "role": "assistant",
            "tool_calls": [{"type": "function"}],  # no id
        }
        assert _get_tool_requests(entry) == []

    def test_assistant_no_tool_calls_no_content(self):
        """role=assistant with neither tool_calls nor content list → empty."""
        entry = {"role": "assistant", "content": "Just text"}
        assert _get_tool_requests(entry) == []

    def test_tool_calls_not_a_list(self):
        """tool_calls is not a list (edge case) → skipped."""
        entry = {"role": "assistant", "tool_calls": "not a list"}
        assert _get_tool_requests(entry) == []

    # -- passing branches (Anthropic) ---------------------------------------

    def test_anthropic_tool_use_items(self):
        """role=assistant + content with tool_use items → extracts IDs."""
        entry = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_a"},
                {"type": "tool_use", "id": "toolu_b"},
            ],
        }
        assert _get_tool_requests(entry) == ["toolu_a", "toolu_b"]

    def test_anthropic_tool_use_missing_id(self):
        """tool_use item without 'id' → skipped."""
        entry = {
            "role": "assistant",
            "content": [{"type": "tool_use"}],
        }
        assert _get_tool_requests(entry) == []

    def test_content_list_with_non_tool_use_items(self):
        """Content list items that are not tool_use → skipped."""
        entry = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        }
        assert _get_tool_requests(entry) == []

    def test_content_not_a_list(self):
        """Content is a string (not a list) → skipped."""
        entry = {"role": "assistant", "content": "plain string"}
        assert _get_tool_requests(entry) == []

    def test_non_assistant_role_returns_empty(self):
        """role != assistant → immediate empty return."""
        entry = {"role": "user", "content": "hello"}
        assert _get_tool_requests(entry) == []

    def test_no_role_returns_empty(self):
        """No role key at all → immediate empty return."""
        entry = {"content": "orphan"}
        assert _get_tool_requests(entry) == []

    # -- failing branches (Responses API) -----------------------------------

    def test_responses_api_function_call_detected(self):
        """type=function_call with call_id → should return [call_id]."""
        entry = {
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_abc",
            "name": "my_tool",
            "arguments": '{"x": 1}',
            "status": "completed",
        }
        assert _get_tool_requests(entry) == ["call_abc"]

    def test_responses_api_function_call_missing_call_id(self):
        """type=function_call WITHOUT call_id → should return [].

        This currently returns [] because the item is ignored entirely
        (no role=assistant).  After the fix it should STILL return []
        because there is no call_id to extract.
        """
        entry = {
            "type": "function_call",
            "id": "fc_abc",
            "name": "my_tool",
            # no call_id
        }
        assert _get_tool_requests(entry) == []


# ───────────────────────────────────────────────────────────────────────────
# _get_tool_responses  (lines 89-111)
# Branches:
#   role=tool → tool_call_id present / absent
#   role=user → content is list → type==tool_result / not → tool_use_id present / absent
#   neither → empty
# ───────────────────────────────────────────────────────────────────────────


class TestGetToolResponses:
    # -- passing branches (Chat Completions) --------------------------------

    def test_openai_tool_response(self):
        """role=tool + tool_call_id → returns [id]."""
        entry = {"role": "tool", "tool_call_id": "call_abc", "content": "result"}
        assert _get_tool_responses(entry) == ["call_abc"]

    def test_openai_tool_response_missing_id(self):
        """role=tool but no tool_call_id → empty."""
        entry = {"role": "tool", "content": "result"}
        assert _get_tool_responses(entry) == []

    # -- passing branches (Anthropic) ---------------------------------------

    def test_anthropic_tool_result(self):
        """role=user + content with tool_result → returns [tool_use_id]."""
        entry = {
            "role": "user",
            "type": "message",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "ok"}
            ],
        }
        assert _get_tool_responses(entry) == ["toolu_abc"]

    def test_anthropic_tool_result_missing_use_id(self):
        """tool_result item without tool_use_id → skipped."""
        entry = {
            "role": "user",
            "content": [{"type": "tool_result", "content": "ok"}],
        }
        assert _get_tool_responses(entry) == []

    def test_anthropic_content_non_tool_result(self):
        """Content items that are not tool_result → skipped."""
        entry = {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
        }
        assert _get_tool_responses(entry) == []

    def test_user_role_content_not_list(self):
        """role=user but content is a string (not list) → empty."""
        entry = {"role": "user", "content": "just text"}
        assert _get_tool_responses(entry) == []

    def test_other_role_returns_empty(self):
        """role=assistant (neither tool nor user) → empty."""
        entry = {"role": "assistant", "content": "response"}
        assert _get_tool_responses(entry) == []

    def test_no_role_returns_empty(self):
        """No role at all → empty."""
        entry = {"content": "orphan"}
        assert _get_tool_responses(entry) == []

    # -- failing branches (Responses API) -----------------------------------

    def test_responses_api_function_call_output_detected(self):
        """type=function_call_output + call_id → should return [call_id]."""
        entry = {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": '{"result": "done"}',
        }
        assert _get_tool_responses(entry) == ["call_abc"]

    def test_responses_api_function_call_output_missing_call_id(self):
        """type=function_call_output WITHOUT call_id → should return [].

        Currently returns [] because the item is ignored entirely
        (no role=tool or role=user).  After the fix it should STILL
        return [] because there is no call_id to extract.
        """
        entry = {
            "type": "function_call_output",
            "output": "result",
        }
        assert _get_tool_responses(entry) == []


# ───────────────────────────────────────────────────────────────────────────
# _create_tool_response  (lines 114-133)
# Branches:  toolu_ prefix → Anthropic  |  else → Chat Completions
# ───────────────────────────────────────────────────────────────────────────


class TestCreateToolResponse:
    # -- passing branches ---------------------------------------------------

    def test_anthropic_format(self):
        """call_id starting with 'toolu_' → Anthropic format."""
        result = _create_tool_response("toolu_abc", "ok")
        assert result["role"] == "user"
        assert result["type"] == "message"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "toolu_abc"
        assert result["content"][0]["content"] == "ok"

    def test_openai_chat_completions_format(self):
        """call_id starting with 'call_' → Chat Completions format.

        This works correctly for Chat Completions models but is WRONG for
        the Responses API.  The test documents the current behaviour.
        """
        result = _create_tool_response("call_abc", "result text")
        assert result == {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "result text",
        }

    def test_non_string_output_is_json_dumped(self):
        """Non-string output is serialised to JSON (compact, via orjson)."""
        result = _create_tool_response("call_abc", {"key": "val"})
        assert result["content"] == '{"key":"val"}'

    def test_dict_output_anthropic(self):
        """Non-string output with Anthropic prefix is also JSON-dumped."""
        result = _create_tool_response("toolu_abc", [1, 2, 3])
        assert result["content"][0]["content"] == "[1,2,3]"

    # -- failing branches (Responses API) -----------------------------------

    def test_responses_api_format_with_flag(self):
        """With responses_api=True, produces Responses API format."""
        result = _create_tool_response("call_abc", "result", responses_api=True)
        assert result == {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "result",
        }

    def test_responses_api_flag_ignored_for_anthropic(self):
        """Anthropic prefix takes priority over responses_api flag."""
        result = _create_tool_response("toolu_abc", "result", responses_api=True)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"


# ───────────────────────────────────────────────────────────────────────────
# _combine_tool_responses  (lines 136-177)
# Branches:
#   len <= 1 → return as-is
#   > 1 Anthropic responses → combine
#   <= 1 Anthropic response → return original list
#   mixed Anthropic + non-Anthropic → combine Anthropic, keep others
# ───────────────────────────────────────────────────────────────────────────


class TestCombineToolResponses:
    def test_empty_list(self):
        assert _combine_tool_responses([]) == []

    def test_single_item(self):
        item = {"role": "tool", "tool_call_id": "call_1", "content": "r"}
        assert _combine_tool_responses([item]) == [item]

    def test_multiple_non_anthropic(self):
        """Multiple non-Anthropic items → returned unchanged."""
        items = [
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
        ]
        assert _combine_tool_responses(items) == items

    def test_single_anthropic_among_multiple(self):
        """Only 1 Anthropic response among >1 total → returned unchanged."""
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"}
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
        ]
        result = _combine_tool_responses(items)
        assert result == items  # No combining when only 1 Anthropic

    def test_multiple_anthropic_responses_combined(self):
        """Multiple Anthropic responses → combined into one user message."""
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"}
                ],
            },
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "b"}
                ],
            },
        ]
        result = _combine_tool_responses(items)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2

    def test_mixed_anthropic_and_non_anthropic(self):
        """Multiple Anthropic + non-Anthropic → Anthropic combined, others kept."""
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"}
                ],
            },
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "b"}
                ],
            },
            {"role": "tool", "tool_call_id": "call_3", "content": "c"},
        ]
        result = _combine_tool_responses(items)
        assert len(result) == 2  # 1 combined Anthropic + 1 non-Anthropic
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[1]["role"] == "tool"

    def test_anthropic_content_not_list_not_combined(self):
        """Anthropic-like item with non-list content → not detected as Anthropic."""
        items = [
            {"role": "user", "type": "message", "content": "not a list"},
            {"role": "user", "type": "message", "content": "also not a list"},
        ]
        assert _combine_tool_responses(items) == items

    def test_anthropic_content_no_tool_result_not_combined(self):
        """Anthropic-like item with list content but no tool_result → not combined."""
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "text", "text": "hi"}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "text", "text": "bye"}],
            },
        ]
        assert _combine_tool_responses(items) == items


# ───────────────────────────────────────────────────────────────────────────
# get_pending_tool_calls  (lines 196-214)
# Branches:  empty/None → {} | has history → count requests - responses
# ───────────────────────────────────────────────────────────────────────────


class TestGetPendingToolCalls:
    def test_none_history(self):
        assert get_pending_tool_calls(None) == {}

    def test_empty_history(self):
        assert get_pending_tool_calls([]) == {}

    def test_chat_completions_pending(self):
        """Chat Completions: request without response → pending."""
        history = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
        ]
        assert get_pending_tool_calls(history) == {"call_1": 1}

    def test_chat_completions_resolved(self):
        """Chat Completions: request + response → resolved (not pending)."""
        history = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
        assert get_pending_tool_calls(history) == {}

    def test_anthropic_pending(self):
        """Anthropic: request without response → pending."""
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1"}],
            },
        ]
        assert get_pending_tool_calls(history) == {"toolu_1": 1}

    def test_anthropic_resolved(self):
        """Anthropic: request + response → resolved."""
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ],
            },
        ]
        assert get_pending_tool_calls(history) == {}

    def test_responses_api_function_call_tracked_as_pending(self):
        """Responses API function_call → should be tracked as pending."""
        history = [
            {"role": "user", "content": "Do something."},
            {
                "type": "function_call",
                "id": "fc_abc",
                "call_id": "call_abc",
                "name": "tool",
                "arguments": "{}",
            },
        ]
        assert get_pending_tool_calls(history) == {"call_abc": 1}

    def test_responses_api_both_invisible_gives_false_resolved(self):
        """Both function_call and function_call_output are invisible to
        the current code, so get_pending_tool_calls returns {} — but for
        the WRONG reason (neither is tracked, not because they cancel out).

        This test documents that limitation: the result is {} but only
        because the function ignores both items entirely.
        """
        history = [
            {
                "type": "function_call",
                "id": "fc_abc",
                "call_id": "call_abc",
                "name": "tool",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "done",
            },
        ]
        # Returns {} — correct result but for the wrong reason
        assert get_pending_tool_calls(history) == {}

    def test_responses_api_unresolved_call_detected(self):
        """A function_call WITHOUT a matching function_call_output should
        show up as pending.  Currently it does NOT because function_call
        items are invisible.
        """
        history = [
            {"role": "user", "content": "Do something."},
            {
                "type": "function_call",
                "id": "fc_abc",
                "call_id": "call_abc",
                "name": "tool",
                "arguments": "{}",
            },
            # No function_call_output → should be pending
        ]
        pending = get_pending_tool_calls(history)
        assert "call_abc" in pending


# ───────────────────────────────────────────────────────────────────────────
# _update_conversation  (lines 753-772)
# Branches:
#   reasoning + no tool calls → append reasoning
#   reasoning + tool calls → skip reasoning
#   no reasoning → skip
#   tool_outputs → extend prompt
#   no tool_outputs → skip extend
# ───────────────────────────────────────────────────────────────────────────


class TestUpdateConversation:
    def _make_response(self, raw, reasoning=None):
        r = MagicMock()
        r.raw_response = raw
        r.reasoning = reasoning
        return r

    # -- passing branches ---------------------------------------------------

    def test_dict_raw_response_no_reasoning_no_tools(self):
        """Dict raw_response, no reasoning → appends assistant dict."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response({"role": "assistant", "content": "hi"})
        block._update_conversation(prompt, resp)
        assert prompt == [{"role": "assistant", "content": "hi"}]

    def test_dict_raw_response_with_reasoning_no_tool_calls(self):
        """Reasoning present, no tool calls → reasoning prepended."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response(
            {"role": "assistant", "content": "answer"},
            reasoning="Let me think…",
        )
        block._update_conversation(prompt, resp)
        assert len(prompt) == 2
        assert prompt[0] == {
            "role": "assistant",
            "content": "[Reasoning]: Let me think…",
        }
        assert prompt[1] == {"role": "assistant", "content": "answer"}

    def test_dict_raw_response_with_reasoning_and_anthropic_tool_calls(self):
        """Reasoning + Anthropic tool_use in content → reasoning skipped."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        raw = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "t", "input": {}}
            ],
        }
        resp = self._make_response(raw, reasoning="Thinking…")
        block._update_conversation(prompt, resp)
        assert len(prompt) == 1
        assert prompt[0] is raw

    def test_with_tool_outputs(self):
        """Tool outputs → extended onto prompt."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response({"role": "assistant", "content": None})
        outputs = [{"role": "tool", "tool_call_id": "call_1", "content": "r"}]
        block._update_conversation(prompt, resp, outputs)
        assert len(prompt) == 2
        assert prompt[1] == outputs[0]

    def test_without_tool_outputs(self):
        """No tool outputs → only assistant message appended."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response({"role": "assistant", "content": "done"})
        block._update_conversation(prompt, resp, None)
        assert len(prompt) == 1

    def test_string_raw_response(self):
        """Ollama string → wrapped as assistant dict."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response("hello from ollama")
        block._update_conversation(prompt, resp)
        assert prompt == [{"role": "assistant", "content": "hello from ollama"}]

    # -- failing branches (Responses API) -----------------------------------

    def test_responses_api_text_response_produces_valid_items(self):
        """Responses API text response → conversation items must have valid role."""
        block = OrchestratorBlock()
        prompt: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        resp = self._make_response(_MockResponse(output=[_MockOutputMessage("Answer")]))
        block._update_conversation(prompt, resp)

        # Every item in the prompt must be a valid API input
        for item in prompt:
            has_role = item.get("role") in ("assistant", "system", "user", "developer")
            has_type = item.get("type") in (
                "function_call",
                "function_call_output",
                "message",
            )
            assert has_role or has_type, f"Invalid conversation item: {item}"

    def test_responses_api_function_call_produces_valid_items(self):
        """Responses API function_call → conversation items must have valid type."""
        block = OrchestratorBlock()
        prompt: list[dict] = []
        resp = self._make_response(
            _MockResponse(output=[_MockFunctionCall("tool", "{}", call_id="call_1")])
        )
        tool_outputs = [
            {"type": "function_call_output", "call_id": "call_1", "output": "done"}
        ]
        block._update_conversation(prompt, resp, tool_outputs)

        for item in prompt:
            has_role = item.get("role") in ("assistant", "system", "user", "developer")
            has_type = item.get("type") in (
                "function_call",
                "function_call_output",
                "message",
            )
            assert has_role or has_type, f"Invalid conversation item: {item}"


# ───────────────────────────────────────────────────────────────────────────
# End-to-end: agent mode loop
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_mode_conversation_valid_for_responses_api():
    """Agent mode with a Responses API raw_response: the conversation passed
    to the second LLM call must contain only valid input items.

    Currently fails because:
    1. The full Response object is serialised as a dict without ``role``
    2. Tool results use ``role: "tool"`` instead of ``type: function_call_output``
    """
    import backend.blocks.llm as llm_module

    block = OrchestratorBlock()

    # First response: tool call
    mock_tc = MagicMock()
    mock_tc.id = "call_abc"
    mock_tc.function.name = "story_improver"
    mock_tc.function.arguments = (
        '{"prompt_values___story": "draft", "prompt_values___improvement": "polish"}'
    )

    resp1 = MagicMock()
    resp1.response = None
    resp1.tool_calls = [mock_tc]
    resp1.prompt_tokens = 100
    resp1.completion_tokens = 50
    resp1.reasoning = None
    resp1.raw_response = _MockResponse(
        output=[
            _MockFunctionCall(
                "story_improver",
                '{"prompt_values___story": "draft", "prompt_values___improvement": "polish"}',
                call_id="call_abc",
            )
        ]
    )

    # Second response: finished
    resp2 = MagicMock()
    resp2.response = "Done!"
    resp2.tool_calls = []
    resp2.prompt_tokens = 200
    resp2.completion_tokens = 10
    resp2.reasoning = None
    resp2.raw_response = _MockResponse(output=[_MockOutputMessage("Done!")])

    llm_mock = AsyncMock(side_effect=[resp1, resp2])

    tool_sigs = [
        {
            "type": "function",
            "function": {
                "name": "story_improver",
                "_sink_node_id": "sink-1",
                "_field_mapping": {},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt_values___story": {"type": "string"},
                        "prompt_values___improvement": {"type": "string"},
                    },
                    "required": [
                        "prompt_values___story",
                        "prompt_values___improvement",
                    ],
                },
            },
        }
    ]

    mock_db = AsyncMock()
    mock_db.get_node.return_value = MagicMock(block_id="bid")
    ner = MagicMock(node_exec_id="neid")
    mock_db.upsert_execution_input.return_value = (
        ner,
        {"prompt_values___story": "draft", "prompt_values___improvement": "polish"},
    )
    mock_db.get_execution_outputs_by_node_exec_id.return_value = {
        "response": "polished"
    }

    ep = AsyncMock()
    ep.running_node_execution = defaultdict(MagicMock)
    ep.execution_stats = MagicMock()
    ep.execution_stats_lock = threading.Lock()
    ns = MagicMock(error=None)
    ep.on_node_execution = AsyncMock(return_value=ns)

    with patch("backend.blocks.llm.llm_call", llm_mock), patch.object(
        block, "_create_tool_node_signatures", return_value=tool_sigs
    ), patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ), patch(
        "backend.executor.manager.async_update_node_execution_status",
        new_callable=AsyncMock,
    ), patch(
        "backend.integrations.creds_manager.IntegrationCredentialsManager"
    ):

        inp = OrchestratorBlock.Input(
            prompt="Improve this",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=5,
        )

        outputs = {}
        async for name, data in block.run(
            inp,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="gid",
            node_id="nid",
            graph_exec_id="geid",
            node_exec_id="neid",
            user_id="uid",
            graph_version=1,
            execution_context=ExecutionContext(human_in_the_loop_safe_mode=False),
            execution_processor=ep,
        ):
            outputs[name] = data

        # The second LLM call's prompt must only contain valid items
        assert llm_mock.call_count >= 2
        second_call = llm_mock.call_args_list[1]
        second_prompt = second_call.kwargs.get("prompt")
        if second_prompt is None:
            second_prompt = second_call[0][1] if len(second_call[0]) > 1 else None
        assert second_prompt is not None

        for i, item in enumerate(second_prompt):
            has_role = item.get("role") in ("assistant", "system", "user", "developer")
            has_type = item.get("type") in (
                "function_call",
                "function_call_output",
                "message",
            )
            assert (
                has_role or has_type
            ), f"input[{i}] has neither valid role nor type: {item!r}"


@pytest.mark.asyncio
async def test_traditional_mode_conversation_valid_for_responses_api():
    """Traditional mode: the yielded conversation must contain only valid items."""
    import backend.blocks.llm as llm_module

    block = OrchestratorBlock()

    mock_tc = MagicMock()
    mock_tc.function.name = "my_tool"
    mock_tc.function.arguments = '{"param": "val"}'

    resp = MagicMock()
    resp.response = None
    resp.tool_calls = [mock_tc]
    resp.prompt_tokens = 50
    resp.completion_tokens = 25
    resp.reasoning = None
    resp.raw_response = _MockResponse(
        output=[_MockFunctionCall("my_tool", '{"param": "val"}', call_id="call_t")]
    )

    tool_sigs = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "_sink_node_id": "sink-1",
                "_field_mapping": {},
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
            },
        }
    ]

    with patch(
        "backend.blocks.llm.llm_call", new_callable=AsyncMock, return_value=resp
    ), patch.object(block, "_create_tool_node_signatures", return_value=tool_sigs):

        inp = OrchestratorBlock.Input(
            prompt="Do it",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        outputs = {}
        async for name, data in block.run(
            inp,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="gid",
            node_id="nid",
            graph_exec_id="geid",
            node_exec_id="neid",
            user_id="uid",
            graph_version=1,
            execution_context=ExecutionContext(human_in_the_loop_safe_mode=False),
            execution_processor=MagicMock(),
        ):
            outputs[name] = data

        conversations = outputs["conversations"]
        for i, item in enumerate(conversations):
            has_role = item.get("role") in ("assistant", "system", "user", "developer")
            has_type = item.get("type") in (
                "function_call",
                "function_call_output",
                "message",
            )
            assert (
                has_role or has_type
            ), f"conversations[{i}] has neither valid role nor type: {item!r}"
