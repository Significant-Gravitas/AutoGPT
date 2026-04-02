"""Tests for OpenAI message serialization, specifically tool_calls arguments."""

from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
)


class TestToolCallArgumentsSerialization:
    """Verify that tool_calls function arguments are serialized as JSON strings
    when preparing messages for the OpenAI API, not as dict objects."""

    def _make_assistant_message_with_tool_calls(
        self, arguments: dict
    ) -> AssistantChatMessage:
        return AssistantChatMessage(
            content="I will open the folder.",
            tool_calls=[
                AssistantToolCall(
                    id="call_abc123",
                    type="function",
                    function=AssistantFunctionCall(
                        name="open_folder",
                        arguments=arguments,
                    ),
                )
            ],
        )

    def test_tool_call_arguments_are_dict_in_model(self):
        """Arguments are stored as dict internally."""
        msg = self._make_assistant_message_with_tool_calls({"path": "."})
        assert msg.tool_calls is not None
        assert isinstance(msg.tool_calls[0].function.arguments, dict)

    def test_model_dump_serializes_arguments_as_dict_by_default(self):
        """model_dump() produces dict arguments.

        OpenAI API requires function.arguments to be a JSON string, but
        model_dump() outputs it as a dict. This test documents the problem.
        """
        msg = self._make_assistant_message_with_tool_calls({"path": "."})
        dumped = msg.model_dump(
            include={"role", "content", "tool_calls"},
            exclude_none=True,
        )
        raw_args = dumped["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw_args, dict), "model_dump should produce dict arguments"
