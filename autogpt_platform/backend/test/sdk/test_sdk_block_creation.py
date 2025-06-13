"""
Tests for creating blocks using the SDK.

This test suite verifies that blocks can be created using only SDK imports
and that they work correctly without decorators.
"""

import pytest

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    Boolean,
    CredentialsField,
    CredentialsMetaInput,
    Integer,
    Optional,
    ProviderBuilder,
    SchemaField,
    SecretStr,
    String,
)


class TestBasicBlockCreation:
    """Test creating basic blocks using the SDK."""

    def test_simple_block(self):
        """Test creating a simple block without any decorators."""

        class SimpleBlock(Block):
            """A simple test block."""

            class Input(BlockSchema):
                text: String = SchemaField(description="Input text")
                count: Integer = SchemaField(description="Repeat count", default=1)

            class Output(BlockSchema):
                result: String = SchemaField(description="Output result")

            def __init__(self):
                super().__init__(
                    id="simple-test-block",
                    description="A simple test block",
                    categories={BlockCategory.TEXT},
                    input_schema=SimpleBlock.Input,
                    output_schema=SimpleBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                result = input_data.text * input_data.count
                yield "result", result

        # Create and test the block
        block = SimpleBlock()
        assert block.id == "simple-test-block"
        assert BlockCategory.TEXT in block.categories

        # Test execution
        outputs = list(
            block.run(
                SimpleBlock.Input(text="Hello ", count=3),
            )
        )
        assert len(outputs) == 1
        assert outputs[0] == ("result", "Hello Hello Hello ")

    def test_block_with_credentials(self):
        """Test creating a block that requires credentials."""

        class APIBlock(Block):
            """A block that requires API credentials."""

            class Input(BlockSchema):
                credentials: CredentialsMetaInput = CredentialsField(
                    provider="test_api",
                    supported_credential_types={"api_key"},
                    description="API credentials",
                )
                query: String = SchemaField(description="API query")

            class Output(BlockSchema):
                response: String = SchemaField(description="API response")
                authenticated: Boolean = SchemaField(description="Was authenticated")

            def __init__(self):
                super().__init__(
                    id="api-test-block",
                    description="Test block with API credentials",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=APIBlock.Input,
                    output_schema=APIBlock.Output,
                )

            def run(
                self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
            ) -> BlockOutput:
                # Simulate API call
                api_key = credentials.api_key.get_secret_value()
                authenticated = bool(api_key)

                yield "response", f"API response for: {input_data.query}"
                yield "authenticated", authenticated

        # Create test credentials
        test_creds = APIKeyCredentials(
            id="test-creds",
            provider="test_api",
            api_key=SecretStr("test-api-key"),
            title="Test API Key",
        )

        # Create and test the block
        block = APIBlock()
        outputs = list(
            block.run(
                APIBlock.Input(
                    credentials={  # type: ignore
                        "provider": "test_api",
                        "id": "test-creds",
                        "type": "api_key",
                    },
                    query="test query",
                ),
                credentials=test_creds,
            )
        )

        assert len(outputs) == 2
        assert outputs[0] == ("response", "API response for: test query")
        assert outputs[1] == ("authenticated", True)

    def test_block_with_multiple_outputs(self):
        """Test block that yields multiple outputs."""

        class MultiOutputBlock(Block):
            """Block with multiple outputs."""

            class Input(BlockSchema):
                text: String = SchemaField(description="Input text")

            class Output(BlockSchema):
                uppercase: String = SchemaField(description="Uppercase version")
                lowercase: String = SchemaField(description="Lowercase version")
                length: Integer = SchemaField(description="Text length")
                is_empty: Boolean = SchemaField(description="Is text empty")

            def __init__(self):
                super().__init__(
                    id="multi-output-block",
                    description="Block with multiple outputs",
                    categories={BlockCategory.TEXT},
                    input_schema=MultiOutputBlock.Input,
                    output_schema=MultiOutputBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                text = input_data.text
                yield "uppercase", text.upper()
                yield "lowercase", text.lower()
                yield "length", len(text)
                yield "is_empty", len(text) == 0

        # Test the block
        block = MultiOutputBlock()
        outputs = list(block.run(MultiOutputBlock.Input(text="Hello World")))

        assert len(outputs) == 4
        assert ("uppercase", "HELLO WORLD") in outputs
        assert ("lowercase", "hello world") in outputs
        assert ("length", 11) in outputs
        assert ("is_empty", False) in outputs


class TestBlockWithProvider:
    """Test creating blocks associated with providers."""

    def setup_method(self):
        """Set up test provider."""
        # Create a provider using ProviderBuilder
        self.provider = (
            ProviderBuilder("test_service")
            .with_api_key("TEST_SERVICE_API_KEY", "Test Service API Key")
            .with_base_cost(10, BlockCostType.RUN)
            .build()
        )

    def test_block_using_provider(self):
        """Test block that uses a registered provider."""

        class TestServiceBlock(Block):
            """Block for test service."""

            class Input(BlockSchema):
                credentials: CredentialsMetaInput = CredentialsField(
                    provider="test_service",  # Matches our provider
                    supported_credential_types={"api_key"},
                    description="Test service credentials",
                )
                action: String = SchemaField(description="Action to perform")

            class Output(BlockSchema):
                result: String = SchemaField(description="Action result")
                provider_name: String = SchemaField(description="Provider used")

            def __init__(self):
                super().__init__(
                    id="test-service-block",
                    description="Block using test service provider",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=TestServiceBlock.Input,
                    output_schema=TestServiceBlock.Output,
                )

            def run(
                self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
            ) -> BlockOutput:
                # The provider name should match
                yield "result", f"Performed: {input_data.action}"
                yield "provider_name", credentials.provider

        # Create credentials for our provider
        creds = APIKeyCredentials(
            id="test-service-creds",
            provider="test_service",
            api_key=SecretStr("test-key"),
            title="Test Service Key",
        )

        # Test the block
        block = TestServiceBlock()
        outputs = dict(
            block.run(
                TestServiceBlock.Input(
                    credentials={  # type: ignore
                        "provider": "test_service",
                        "id": "test-service-creds",
                        "type": "api_key",
                    },
                    action="test action",
                ),
                credentials=creds,
            )
        )

        assert outputs["result"] == "Performed: test action"
        assert outputs["provider_name"] == "test_service"


class TestComplexBlockScenarios:
    """Test more complex block scenarios."""

    def test_block_with_optional_fields(self):
        """Test block with optional input fields."""
        # Optional is already imported at the module level

        class OptionalFieldBlock(Block):
            """Block with optional fields."""

            class Input(BlockSchema):
                required_field: String = SchemaField(description="Required field")
                optional_field: Optional[String] = SchemaField(
                    description="Optional field",
                    default=None,
                )
                optional_with_default: String = SchemaField(
                    description="Optional with default",
                    default="default value",
                )

            class Output(BlockSchema):
                has_optional: Boolean = SchemaField(description="Has optional value")
                optional_value: Optional[String] = SchemaField(
                    description="Optional value"
                )
                default_value: String = SchemaField(description="Default value")

            def __init__(self):
                super().__init__(
                    id="optional-field-block",
                    description="Block with optional fields",
                    categories={BlockCategory.TEXT},
                    input_schema=OptionalFieldBlock.Input,
                    output_schema=OptionalFieldBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                yield "has_optional", input_data.optional_field is not None
                yield "optional_value", input_data.optional_field
                yield "default_value", input_data.optional_with_default

        # Test with optional field provided
        block = OptionalFieldBlock()
        outputs = dict(
            block.run(
                OptionalFieldBlock.Input(
                    required_field="test",
                    optional_field="provided",
                )
            )
        )

        assert outputs["has_optional"] is True
        assert outputs["optional_value"] == "provided"
        assert outputs["default_value"] == "default value"

        # Test without optional field
        outputs = dict(
            block.run(
                OptionalFieldBlock.Input(
                    required_field="test",
                )
            )
        )

        assert outputs["has_optional"] is False
        assert outputs["optional_value"] is None
        assert outputs["default_value"] == "default value"

    def test_block_with_complex_types(self):
        """Test block with complex input/output types."""
        from backend.sdk import BaseModel, Dict, List

        class ItemModel(BaseModel):
            name: str
            value: int

        class ComplexBlock(Block):
            """Block with complex types."""

            class Input(BlockSchema):
                items: List[String] = SchemaField(description="List of items")
                mapping: Dict[String, Integer] = SchemaField(
                    description="String to int mapping"
                )

            class Output(BlockSchema):
                item_count: Integer = SchemaField(description="Number of items")
                total_value: Integer = SchemaField(description="Sum of mapping values")
                combined: List[String] = SchemaField(description="Combined results")

            def __init__(self):
                super().__init__(
                    id="complex-types-block",
                    description="Block with complex types",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=ComplexBlock.Input,
                    output_schema=ComplexBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                yield "item_count", len(input_data.items)
                yield "total_value", sum(input_data.mapping.values())

                # Combine items with their mapping values
                combined = []
                for item in input_data.items:
                    value = input_data.mapping.get(item, 0)
                    combined.append(f"{item}: {value}")

                yield "combined", combined

        # Test the block
        block = ComplexBlock()
        outputs = dict(
            block.run(
                ComplexBlock.Input(
                    items=["apple", "banana", "orange"],
                    mapping={"apple": 5, "banana": 3, "orange": 4},
                )
            )
        )

        assert outputs["item_count"] == 3
        assert outputs["total_value"] == 12
        assert outputs["combined"] == ["apple: 5", "banana: 3", "orange: 4"]

    def test_block_error_handling(self):
        """Test block error handling."""

        class ErrorHandlingBlock(Block):
            """Block that demonstrates error handling."""

            class Input(BlockSchema):
                value: Integer = SchemaField(description="Input value")
                should_error: Boolean = SchemaField(
                    description="Whether to trigger an error",
                    default=False,
                )

            class Output(BlockSchema):
                result: Integer = SchemaField(description="Result")
                error_message: Optional[String] = SchemaField(
                    description="Error if any", default=None
                )

            def __init__(self):
                super().__init__(
                    id="error-handling-block",
                    description="Block with error handling",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=ErrorHandlingBlock.Input,
                    output_schema=ErrorHandlingBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                if input_data.should_error:
                    raise ValueError("Intentional error triggered")

                if input_data.value < 0:
                    yield "error_message", "Value must be non-negative"
                    yield "result", 0
                else:
                    yield "result", input_data.value * 2
                    yield "error_message", None

        # Test normal operation
        block = ErrorHandlingBlock()
        outputs = dict(block.run(ErrorHandlingBlock.Input(value=5, should_error=False)))

        assert outputs["result"] == 10
        assert outputs["error_message"] is None

        # Test with negative value
        outputs = dict(
            block.run(ErrorHandlingBlock.Input(value=-5, should_error=False))
        )

        assert outputs["result"] == 0
        assert outputs["error_message"] == "Value must be non-negative"

        # Test with error
        with pytest.raises(ValueError, match="Intentional error triggered"):
            list(block.run(ErrorHandlingBlock.Input(value=5, should_error=True)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
