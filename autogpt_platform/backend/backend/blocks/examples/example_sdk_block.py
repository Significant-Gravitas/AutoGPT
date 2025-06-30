"""
Example Block using the new SDK

This demonstrates:
1. Import from backend.sdk module
2. Auto-registration decorators
3. No external configuration needed
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    Boolean,
    CredentialsMetaInput,
    Integer,
    SchemaField,
    String,
)

from ._config import (
    EXAMPLE_SERVICE_TEST_CREDENTIALS,
    EXAMPLE_SERVICE_TEST_CREDENTIALS_INPUT,
    example_service,
)


# Example of a simple service
class ExampleSDKBlock(Block):
    """
    Example block demonstrating the new SDK system.

    With the new SDK:
    - All imports come from 'backend.sdk'
    - Costs are registered via @cost_config decorator
    - Default credentials via @default_credentials decorator
    - Provider name via @provider decorator
    - No need to modify any files outside the blocks folder!
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = example_service.credentials_field(
            description="Credentials for Example Service API",
        )
        text: String = SchemaField(
            description="Text to process", default="Hello, World!"
        )
        max_length: Integer = SchemaField(
            description="Maximum length of output", default=100
        )

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed text result")
        length: Integer = SchemaField(description="Length of the result")
        api_key_used: Boolean = SchemaField(description="Whether API key was used")
        error: String = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="83815e8c-1273-418e-a8c2-4c454e042060",
            description="Example block showing SDK capabilities with auto-registration",
            categories={BlockCategory.TEXT, BlockCategory.BASIC},
            input_schema=ExampleSDKBlock.Input,
            output_schema=ExampleSDKBlock.Output,
            test_input={
                "credentials": EXAMPLE_SERVICE_TEST_CREDENTIALS_INPUT,
                "text": "Test input",
                "max_length": 50,
            },
            test_output=[
                ("result", "PROCESSED: Test input"),
                ("length", 21),  # Length of "PROCESSED: Test input"
                ("api_key_used", True),
            ],
            test_credentials=EXAMPLE_SERVICE_TEST_CREDENTIALS,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            # Get API key from credentials
            api_key = credentials.api_key.get_secret_value()

            # Simulate API processing
            processed_text = f"PROCESSED: {input_data.text}"

            # Truncate if needed
            if len(processed_text) > input_data.max_length:
                processed_text = processed_text[: input_data.max_length]

            yield "result", processed_text
            yield "length", len(processed_text)
            yield "api_key_used", bool(api_key)

        except Exception as e:
            yield "error", str(e)
            yield "result", ""
            yield "length", 0
            yield "api_key_used", False
