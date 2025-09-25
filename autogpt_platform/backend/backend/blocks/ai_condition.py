from typing import Any

from backend.blocks.llm import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIBlockBase,
    AICredentials,
    AICredentialsField,
    LlmModel,
    LLMResponse,
    llm_call,
)
from backend.data.block import BlockCategory, BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField


class AIConditionBlock(AIBlockBase):
    """
    An AI-powered condition block that uses natural language to evaluate conditions.

    This block allows users to define conditions in plain English (e.g., "the input is an email address",
    "the input is a city in the USA") and uses AI to determine if the input satisfies the condition.
    It provides the same yes/no data pass-through functionality as the standard ConditionBlock.
    """

    class Input(BlockSchema):
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
            default=LlmModel.GPT4O,
            description="The language model to use for evaluating the condition.",
            advanced=False,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchema):
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
                "model": LlmModel.GPT4O,
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
        try:
            response = await self.llm_call(
                credentials=credentials,
                llm_model=input_data.model,
                prompt=prompt,
                max_tokens=10,  # We only expect a true/false response
            )

            # Extract the boolean result from the response
            response_text = response.response.strip().lower()
            if response_text == "true":
                result = True
            elif response_text == "false":
                result = False
            else:
                # If the response is not clear, try to interpret it using word boundaries
                import re

                # Use word boundaries to avoid false positives like 'untrue' or '10'
                tokens = set(re.findall(r"\b(true|false|yes|no|1|0)\b", response_text))

                if tokens == {"true"} or tokens == {"yes"} or tokens == {"1"}:
                    result = True
                elif tokens == {"false"} or tokens == {"no"} or tokens == {"0"}:
                    result = False
                else:
                    # Unclear or conflicting response - default to False and yield error
                    result = False
                    yield "error", f"Unclear AI response: '{response.response}'"

            # Update internal stats
            self.merge_stats(
                NodeExecutionStats(
                    input_token_count=response.prompt_tokens,
                    output_token_count=response.completion_tokens,
                )
            )
            self.prompt = response.prompt

        except Exception as e:
            # In case of any error, default to False to be safe
            result = False
            # Log the error but don't fail the block execution
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"AI condition evaluation failed: {str(e)}")
            yield "error", f"AI evaluation failed: {str(e)}"

        # Yield results
        yield "result", result

        if result:
            yield "yes_output", yes_value
        else:
            yield "no_output", no_value
