import re
from typing import Any

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.llm import (
    DEFAULT_LLM_MODEL,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIBlockBase,
    AICredentials,
    AICredentialsField,
    LlmModel,
    LLMResponse,
    llm_call,
)
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField

# Minimum max_output_tokens accepted by OpenAI-compatible APIs.
# A true/false answer fits comfortably within this budget.
MIN_LLM_OUTPUT_TOKENS = 16


def _parse_boolean_response(response_text: str) -> tuple[bool, str | None]:
    """Parse an LLM response into a boolean result.

    Returns a ``(result, error)`` tuple.  *error* is ``None`` when the
    response is unambiguous; otherwise it contains a diagnostic message
    and *result* defaults to ``False``.
    """
    text = response_text.strip().lower()
    if text == "true":
        return True, None
    if text == "false":
        return False, None

    # Fuzzy match – use word boundaries to avoid false positives like "untrue".
    tokens = set(re.findall(r"\b(true|false|yes|no|1|0)\b", text))
    if tokens == {"true"} or tokens == {"yes"} or tokens == {"1"}:
        return True, None
    if tokens == {"false"} or tokens == {"no"} or tokens == {"0"}:
        return False, None

    return False, f"Unclear AI response: '{response_text}'"


class AIConditionBlock(AIBlockBase):
    """
    An AI-powered condition block that uses natural language to evaluate conditions.

    This block allows users to define conditions in plain English (e.g., "the input is an email address",
    "the input is a city in the USA") and uses AI to determine if the input satisfies the condition.
    It provides the same yes/no data pass-through functionality as the standard ConditionBlock.
    """

    class Input(BlockSchemaInput):
        input_value: Any = SchemaField(
            description="The input value to evaluate with the AI condition",
            placeholder="Enter the value to be evaluated (text, number, or any data)",
        )
        condition: str = SchemaField(
            description="A plaintext English description of the condition to evaluate",
            placeholder="E.g., 'the input is the body of an email', 'the input is a City in the USA', 'the input is an error or a refusal'",
        )
        yes_value: Any = SchemaField(
            description="(Optional) Value to output if the condition is true. If not provided, input_value will be used.",
            placeholder="Leave empty to use input_value, or enter a specific value",
            default=None,
        )
        no_value: Any = SchemaField(
            description="(Optional) Value to output if the condition is false. If not provided, input_value will be used.",
            placeholder="Leave empty to use input_value, or enter a specific value",
            default=None,
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for evaluating the condition.",
            advanced=False,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchemaOutput):
        result: bool = SchemaField(
            description="The result of the AI condition evaluation (True or False)"
        )
        yes_output: Any = SchemaField(
            description="The output value if the condition is true"
        )
        no_output: Any = SchemaField(
            description="The output value if the condition is false"
        )
        error: str = SchemaField(
            description="Error message if the AI evaluation is uncertain or fails"
        )

    def __init__(self):
        super().__init__(
            id="553ec5b8-6c45-4299-8d75-b394d05f72ff",
            input_schema=AIConditionBlock.Input,
            output_schema=AIConditionBlock.Output,
            description="Uses AI to evaluate natural language conditions and provide conditional outputs",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            test_input={
                "input_value": "john@example.com",
                "condition": "the input is an email address",
                "yes_value": "Valid email",
                "no_value": "Not an email",
                "model": DEFAULT_LLM_MODEL,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", True),
                ("yes_output", "Valid email"),
            ],
            test_mock={
                "llm_call": lambda *args, **kwargs: LLMResponse(
                    raw_response="",
                    prompt=[],
                    response="true",
                    tool_calls=None,
                    prompt_tokens=50,
                    completion_tokens=10,
                    reasoning=None,
                )
            },
        )

    async def llm_call(
        self,
        credentials: APIKeyCredentials,
        llm_model: LlmModel,
        prompt: list,
        max_tokens: int,
    ) -> LLMResponse:
        """Wrapper method for llm_call to enable mocking in tests."""
        return await llm_call(
            credentials=credentials,
            llm_model=llm_model,
            prompt=prompt,
            force_json_output=False,
            max_tokens=max_tokens,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Evaluate the AI condition and return appropriate outputs.
        """
        # Prepare the yes and no values, using input_value as default
        yes_value = (
            input_data.yes_value
            if input_data.yes_value is not None
            else input_data.input_value
        )
        no_value = (
            input_data.no_value
            if input_data.no_value is not None
            else input_data.input_value
        )

        # Convert input_value to string for AI evaluation
        input_str = str(input_data.input_value)

        # Create the prompt for AI evaluation
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that evaluates conditions based on input data. "
                    "You must respond with only 'true' or 'false' (lowercase) to indicate whether "
                    "the given condition is met by the input value. Be accurate and consider the "
                    "context and meaning of both the input and the condition."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Input value: {input_str}\n"
                    f"Condition to evaluate: {input_data.condition}\n\n"
                    f"Does the input value satisfy the condition? Respond with only 'true' or 'false'."
                ),
            },
        ]

        # Call the LLM
        response = await self.llm_call(
            credentials=credentials,
            llm_model=input_data.model,
            prompt=prompt,
            max_tokens=MIN_LLM_OUTPUT_TOKENS,
        )

        # Extract the boolean result from the response
        result, error = _parse_boolean_response(response.response)
        if error:
            yield "error", error

        # Update internal stats
        self.merge_stats(
            NodeExecutionStats(
                input_token_count=response.prompt_tokens,
                output_token_count=response.completion_tokens,
                cache_read_token_count=response.cache_read_tokens,
                cache_creation_token_count=response.cache_creation_tokens,
                provider_cost=response.provider_cost,
            )
        )
        self.prompt = response.prompt

        # Yield results
        yield "result", result

        if result:
            yield "yes_output", yes_value
        else:
            yield "no_output", no_value
