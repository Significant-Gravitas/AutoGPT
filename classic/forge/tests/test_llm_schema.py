"""Tests for LLM provider schema models."""

import pytest
from pydantic import ValidationError

from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderName,
    ModelProviderService,
    ModelProviderUsage,
    ToolResultMessage,
)
from forge.models.json_schema import JSONSchema


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------
class TestChatMessage:
    def test_user_factory(self):
        msg = ChatMessage.user("hello")
        assert msg.role == ChatMessage.Role.USER
        assert msg.content == "hello"

    def test_system_factory(self):
        msg = ChatMessage.system("you are an AI")
        assert msg.role == ChatMessage.Role.SYSTEM
        assert msg.content == "you are an AI"

    def test_roles_are_strings(self):
        assert ChatMessage.Role.USER == "user"
        assert ChatMessage.Role.SYSTEM == "system"
        assert ChatMessage.Role.ASSISTANT == "assistant"
        assert ChatMessage.Role.TOOL == "tool"
        assert ChatMessage.Role.FUNCTION == "function"

    def test_model_dump_includes_role_and_content(self):
        msg = ChatMessage.user("test")
        dumped = msg.model_dump()
        assert dumped["role"] == "user"
        assert dumped["content"] == "test"


# ---------------------------------------------------------------------------
# AssistantFunctionCall
# ---------------------------------------------------------------------------
class TestAssistantFunctionCall:
    def test_str_representation(self):
        fc = AssistantFunctionCall(name="search", arguments={"query": "test"})
        result = str(fc)
        assert "search" in result
        assert "query" in result

    def test_empty_arguments(self):
        fc = AssistantFunctionCall(name="noop", arguments={})
        result = str(fc)
        assert "noop" in result

    def test_multiple_arguments(self):
        fc = AssistantFunctionCall(
            name="write_file", arguments={"path": "/tmp/f.txt", "content": "data"}
        )
        result = str(fc)
        assert "write_file" in result
        assert "path" in result
        assert "content" in result


# ---------------------------------------------------------------------------
# AssistantToolCall
# ---------------------------------------------------------------------------
class TestAssistantToolCall:
    def test_construction(self):
        tc = AssistantToolCall(
            id="call_1",
            type="function",
            function=AssistantFunctionCall(name="test", arguments={"a": 1}),
        )
        assert tc.id == "call_1"
        assert tc.type == "function"
        assert tc.function.name == "test"
        assert tc.function.arguments == {"a": 1}

    def test_model_dump_roundtrip(self):
        tc = AssistantToolCall(
            id="call_2",
            type="function",
            function=AssistantFunctionCall(name="fn", arguments={"x": "y"}),
        )
        dumped = tc.model_dump()
        restored = AssistantToolCall.model_validate(dumped)
        assert restored.id == tc.id
        assert restored.function.name == tc.function.name
        assert restored.function.arguments == tc.function.arguments


# ---------------------------------------------------------------------------
# AssistantChatMessage
# ---------------------------------------------------------------------------
class TestAssistantChatMessage:
    def test_defaults(self):
        msg = AssistantChatMessage()
        assert msg.role == ChatMessage.Role.ASSISTANT
        assert msg.content == ""
        assert msg.tool_calls is None

    def test_with_content_only(self):
        msg = AssistantChatMessage(content="I will help you.")
        assert msg.content == "I will help you."
        assert msg.tool_calls is None

    def test_with_tool_calls(self):
        tc = AssistantToolCall(
            id="call_1",
            type="function",
            function=AssistantFunctionCall(name="search", arguments={"q": "test"}),
        )
        msg = AssistantChatMessage(content="Searching...", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "search"

    def test_empty_tool_calls_list_stored_as_none_via_dump(self):
        """When tool_calls is explicitly None, model_dump with exclude_none omits it."""
        msg = AssistantChatMessage(content="hi", tool_calls=None)
        dumped = msg.model_dump(exclude_none=True)
        assert "tool_calls" not in dumped

    def test_model_dump_preserves_tool_calls(self):
        tc = AssistantToolCall(
            id="call_1",
            type="function",
            function=AssistantFunctionCall(name="fn", arguments={"a": 1}),
        )
        msg = AssistantChatMessage(content="ok", tool_calls=[tc])
        dumped = msg.model_dump(exclude_none=True)
        assert "tool_calls" in dumped
        assert len(dumped["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# ToolResultMessage
# ---------------------------------------------------------------------------
class TestToolResultMessage:
    def test_construction(self):
        msg = ToolResultMessage(tool_call_id="call_1", content="Result here")
        assert msg.role == ChatMessage.Role.TOOL
        assert msg.tool_call_id == "call_1"
        assert msg.content == "Result here"
        assert msg.is_error is False

    def test_error_result(self):
        msg = ToolResultMessage(
            tool_call_id="call_2", content="Something failed", is_error=True
        )
        assert msg.is_error is True

    def test_model_dump_includes_tool_call_id(self):
        msg = ToolResultMessage(tool_call_id="call_1", content="ok")
        dumped = msg.model_dump(
            include={"role", "content", "tool_call_id"}, exclude_none=True
        )
        assert dumped["tool_call_id"] == "call_1"
        assert dumped["role"] == "tool"


# ---------------------------------------------------------------------------
# CompletionModelFunction
# ---------------------------------------------------------------------------
class TestCompletionModelFunction:
    @pytest.fixture
    def search_function(self):
        return CompletionModelFunction(
            name="web_search",
            description="Search the web",
            parameters={
                "query": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="Search query",
                    required=True,
                ),
                "limit": JSONSchema(
                    type=JSONSchema.Type.INTEGER,
                    description="Max results",
                    required=False,
                ),
            },
        )

    def test_fmt_line(self, search_function):
        line = search_function.fmt_line()
        assert "web_search" in line
        assert "Search the web" in line
        assert "query" in line
        assert "limit" in line

    def test_validate_call_valid(self, search_function):
        fc = AssistantFunctionCall(name="web_search", arguments={"query": "test"})
        is_valid, errors = search_function.validate_call(fc)
        assert is_valid
        assert errors == []

    def test_validate_call_wrong_function_name_raises(self, search_function):
        fc = AssistantFunctionCall(name="wrong_name", arguments={"query": "test"})
        with pytest.raises(ValueError, match="Can't validate wrong_name"):
            search_function.validate_call(fc)

    def test_validate_call_with_optional_param(self, search_function):
        fc = AssistantFunctionCall(
            name="web_search", arguments={"query": "test", "limit": 5}
        )
        is_valid, errors = search_function.validate_call(fc)
        assert is_valid

    def test_no_parameters(self):
        fn = CompletionModelFunction(
            name="get_time", description="Get current time", parameters={}
        )
        line = fn.fmt_line()
        assert "get_time" in line


# ---------------------------------------------------------------------------
# ModelProviderUsage
# ---------------------------------------------------------------------------
class TestModelProviderUsage:
    def test_initial_state(self):
        usage = ModelProviderUsage()
        assert usage.completion_tokens == 0
        assert usage.prompt_tokens == 0

    def test_update_usage_single_model(self):
        usage = ModelProviderUsage()
        usage.update_usage("gpt-4", input_tokens_used=100, output_tokens_used=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    def test_update_usage_multiple_models(self):
        usage = ModelProviderUsage()
        usage.update_usage("gpt-4", input_tokens_used=100, output_tokens_used=50)
        usage.update_usage("gpt-3.5", input_tokens_used=200, output_tokens_used=100)
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 150

    def test_update_usage_accumulates(self):
        usage = ModelProviderUsage()
        usage.update_usage("gpt-4", input_tokens_used=100)
        usage.update_usage("gpt-4", input_tokens_used=200)
        assert usage.prompt_tokens == 300

    def test_update_usage_output_defaults_to_zero(self):
        usage = ModelProviderUsage()
        usage.update_usage("gpt-4", input_tokens_used=100)
        assert usage.completion_tokens == 0


# ---------------------------------------------------------------------------
# ModelProviderBudget
# ---------------------------------------------------------------------------
class TestModelProviderBudget:
    @pytest.fixture
    def model_info(self):
        return ChatModelInfo(
            name="test-model",
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=10.0 / 1_000_000,
            completion_token_cost=30.0 / 1_000_000,
            max_tokens=4096,
        )

    def test_update_usage_and_cost(self, model_info):
        budget = ModelProviderBudget()
        cost = budget.update_usage_and_cost(
            model_info=model_info,
            input_tokens_used=1000,
            output_tokens_used=500,
        )
        expected = 1000 * (10.0 / 1e6) + 500 * (30.0 / 1e6)
        assert cost == pytest.approx(expected)
        assert budget.total_cost == pytest.approx(expected)

    def test_total_cost_increases(self, model_info):
        budget = ModelProviderBudget()
        assert budget.total_cost == 0.0
        budget.update_usage_and_cost(
            model_info=model_info, input_tokens_used=1000, output_tokens_used=500
        )
        assert budget.total_cost > 0.0

    def test_budget_accumulates_across_calls(self, model_info):
        budget = ModelProviderBudget()
        cost1 = budget.update_usage_and_cost(
            model_info=model_info, input_tokens_used=1000
        )
        cost2 = budget.update_usage_and_cost(
            model_info=model_info, input_tokens_used=2000
        )
        assert budget.total_cost == pytest.approx(cost1 + cost2)

    def test_usage_tracked_per_model(self, model_info):
        budget = ModelProviderBudget()
        budget.update_usage_and_cost(
            model_info=model_info, input_tokens_used=100, output_tokens_used=50
        )
        assert budget.usage.prompt_tokens == 100
        assert budget.usage.completion_tokens == 50


# ---------------------------------------------------------------------------
# ModelProviderConfiguration
# ---------------------------------------------------------------------------
class TestModelProviderConfiguration:
    def test_defaults(self):
        config = ModelProviderConfiguration()
        assert config.retries_per_request == 7
        assert config.fix_failed_parse_tries == 3
        assert config.extra_request_headers == {}
        assert config.thinking_budget_tokens is None
        assert config.reasoning_effort is None


# ---------------------------------------------------------------------------
# ChatModelInfo
# ---------------------------------------------------------------------------
class TestChatModelInfo:
    def test_service_is_chat(self):
        info = ChatModelInfo(
            name="test",
            provider_name=ModelProviderName.OPENAI,
            max_tokens=4096,
        )
        assert info.service == ModelProviderService.CHAT

    def test_defaults(self):
        info = ChatModelInfo(
            name="test",
            provider_name=ModelProviderName.OPENAI,
            max_tokens=4096,
        )
        assert info.has_function_call_api is False
        assert info.supports_extended_thinking is False
        assert info.supports_reasoning_effort is False
        assert info.prompt_token_cost == 0.0
        assert info.completion_token_cost == 0.0


# ---------------------------------------------------------------------------
# EmbeddingModelInfo
# ---------------------------------------------------------------------------
class TestEmbeddingModelInfo:
    def test_service_is_embedding(self):
        info = EmbeddingModelInfo(
            name="embed-test",
            provider_name=ModelProviderName.OPENAI,
            max_tokens=8191,
            embedding_dimensions=1536,
        )
        assert info.service == ModelProviderService.EMBEDDING


# ---------------------------------------------------------------------------
# ChatModelResponse
# ---------------------------------------------------------------------------
class TestChatModelResponse:
    def test_construction(self):
        resp = ChatModelResponse(
            response=AssistantChatMessage(content="hello"),
            parsed_result={"key": "value"},
            llm_info=ChatModelInfo(
                name="test",
                provider_name=ModelProviderName.OPENAI,
                max_tokens=4096,
            ),
            prompt_tokens_used=100,
            completion_tokens_used=50,
        )
        assert resp.parsed_result == {"key": "value"}
        assert resp.prompt_tokens_used == 100
        assert resp.completion_tokens_used == 50
        assert resp.response.content == "hello"


# ---------------------------------------------------------------------------
# EmbeddingModelResponse
# ---------------------------------------------------------------------------
class TestEmbeddingModelResponse:
    def test_completion_tokens_frozen_at_zero(self):
        resp = EmbeddingModelResponse(
            embedding=[0.1, 0.2, 0.3],
            llm_info=EmbeddingModelInfo(
                name="embed",
                provider_name=ModelProviderName.OPENAI,
                max_tokens=8191,
                embedding_dimensions=3,
            ),
            prompt_tokens_used=10,
        )
        assert resp.completion_tokens_used == 0
        with pytest.raises(ValidationError):
            resp.completion_tokens_used = 5  # type: ignore


# ---------------------------------------------------------------------------
# BaseModelProvider (get_incurred_cost / get_remaining_budget)
# These are tested indirectly via the concrete implementations,
# but we can verify the logic via ModelProviderBudget directly.
# ---------------------------------------------------------------------------
class TestBaseModelProviderBudgetAccess:
    def test_budget_total_cost_starts_at_zero(self):
        budget = ModelProviderBudget()
        assert budget.total_cost == 0.0
