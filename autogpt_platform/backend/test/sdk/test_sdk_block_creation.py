"""
Tests for creating blocks using the SDK.

This test suite verifies that blocks can be created using only SDK imports
and that they work correctly without decorators.
"""

from typing import Any, Optional, Union

import pytest

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    OAuth2Credentials,
    ProviderBuilder,
    SchemaField,
    SecretStr,
)

from ._config import test_api, test_service


class TestBasicBlockCreation:
    """Test creating basic blocks using the SDK."""

    @pytest.mark.asyncio
    async def test_simple_block(self):
        """Test creating a simple block without any decorators."""

        class SimpleBlock(Block):
            """A simple test block."""

            class Input(BlockSchemaInput):
                text: str = SchemaField(description="Input text")
                count: int = SchemaField(description="Repeat count", default=1)

            class Output(BlockSchemaOutput):
                result: str = SchemaField(description="Output result")

            def __init__(self):
                super().__init__(
                    id="simple-test-block",
                    description="A simple test block",
                    categories={BlockCategory.TEXT},
                    input_schema=SimpleBlock.Input,
                    output_schema=SimpleBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                result = input_data.text * input_data.count
                yield "result", result

        # Create and test the block
        block = SimpleBlock()
        assert block.id == "simple-test-block"
        assert BlockCategory.TEXT in block.categories

        # Test execution
        outputs = []
        async for name, value in block.run(
            SimpleBlock.Input(text="Hello ", count=3),
        ):
            outputs.append((name, value))
        assert len(outputs) == 1
        assert outputs[0] == ("result", "Hello Hello Hello ")

    @pytest.mark.asyncio
    async def test_block_with_credentials(self):
        """Test creating a block that requires credentials."""

        class APIBlock(Block):
            """A block that requires API credentials."""

            class Input(BlockSchemaInput):
                credentials: CredentialsMetaInput = test_api.credentials_field(
                    description="API credentials for test service",
                )
                query: str = SchemaField(description="API query")

            class Output(BlockSchemaOutput):
                response: str = SchemaField(description="API response")
                authenticated: bool = SchemaField(description="Was authenticated")

            def __init__(self):
                super().__init__(
                    id="api-test-block",
                    description="Test block with API credentials",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=APIBlock.Input,
                    output_schema=APIBlock.Output,
                )

            async def run(
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
        outputs = []
        async for name, value in block.run(
            APIBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_api",
                    "id": "test-creds",
                    "type": "api_key",
                },
                query="test query",
            ),
            credentials=test_creds,
        ):
            outputs.append((name, value))

        assert len(outputs) == 2
        assert outputs[0] == ("response", "API response for: test query")
        assert outputs[1] == ("authenticated", True)

    @pytest.mark.asyncio
    async def test_block_with_multiple_outputs(self):
        """Test block that yields multiple outputs."""

        class MultiOutputBlock(Block):
            """Block with multiple outputs."""

            class Input(BlockSchemaInput):
                text: str = SchemaField(description="Input text")

            class Output(BlockSchemaOutput):
                uppercase: str = SchemaField(description="Uppercase version")
                lowercase: str = SchemaField(description="Lowercase version")
                length: int = SchemaField(description="Text length")
                is_empty: bool = SchemaField(description="Is text empty")

            def __init__(self):
                super().__init__(
                    id="multi-output-block",
                    description="Block with multiple outputs",
                    categories={BlockCategory.TEXT},
                    input_schema=MultiOutputBlock.Input,
                    output_schema=MultiOutputBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                text = input_data.text
                yield "uppercase", text.upper()
                yield "lowercase", text.lower()
                yield "length", len(text)
                yield "is_empty", len(text) == 0

        # Test the block
        block = MultiOutputBlock()
        outputs = []
        async for name, value in block.run(MultiOutputBlock.Input(text="Hello World")):
            outputs.append((name, value))

        assert len(outputs) == 4
        assert ("uppercase", "HELLO WORLD") in outputs
        assert ("lowercase", "hello world") in outputs
        assert ("length", 11) in outputs
        assert ("is_empty", False) in outputs


class TestBlockWithProvider:
    """Test creating blocks associated with providers."""

    @pytest.mark.asyncio
    async def test_block_using_provider(self):
        """Test block that uses a registered provider."""

        class TestServiceBlock(Block):
            """Block for test service."""

            class Input(BlockSchemaInput):
                credentials: CredentialsMetaInput = test_service.credentials_field(
                    description="Test service credentials",
                )
                action: str = SchemaField(description="Action to perform")

            class Output(BlockSchemaOutput):
                result: str = SchemaField(description="Action result")
                provider_name: str = SchemaField(description="Provider used")

            def __init__(self):
                super().__init__(
                    id="test-service-block",
                    description="Block using test service provider",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=TestServiceBlock.Input,
                    output_schema=TestServiceBlock.Output,
                )

            async def run(
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
        outputs = {}
        async for name, value in block.run(
            TestServiceBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_service",
                    "id": "test-service-creds",
                    "type": "api_key",
                },
                action="test action",
            ),
            credentials=creds,
        ):
            outputs[name] = value

        assert outputs["result"] == "Performed: test action"
        assert outputs["provider_name"] == "test_service"


class TestComplexBlockScenarios:
    """Test more complex block scenarios."""

    @pytest.mark.asyncio
    async def test_block_with_optional_fields(self):
        """Test block with optional input fields."""
        # Optional is already imported at the module level

        class OptionalFieldBlock(Block):
            """Block with optional fields."""

            class Input(BlockSchemaInput):
                required_field: str = SchemaField(description="Required field")
                optional_field: Optional[str] = SchemaField(
                    description="Optional field",
                    default=None,
                )
                optional_with_default: str = SchemaField(
                    description="Optional with default",
                    default="default value",
                )

            class Output(BlockSchemaOutput):
                has_optional: bool = SchemaField(description="Has optional value")
                optional_value: Optional[str] = SchemaField(
                    description="Optional value"
                )
                default_value: str = SchemaField(description="Default value")

            def __init__(self):
                super().__init__(
                    id="optional-field-block",
                    description="Block with optional fields",
                    categories={BlockCategory.TEXT},
                    input_schema=OptionalFieldBlock.Input,
                    output_schema=OptionalFieldBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                yield "has_optional", input_data.optional_field is not None
                yield "optional_value", input_data.optional_field
                yield "default_value", input_data.optional_with_default

        # Test with optional field provided
        block = OptionalFieldBlock()
        outputs = {}
        async for name, value in block.run(
            OptionalFieldBlock.Input(
                required_field="test",
                optional_field="provided",
            )
        ):
            outputs[name] = value

        assert outputs["has_optional"] is True
        assert outputs["optional_value"] == "provided"
        assert outputs["default_value"] == "default value"

        # Test without optional field
        outputs = {}
        async for name, value in block.run(
            OptionalFieldBlock.Input(
                required_field="test",
            )
        ):
            outputs[name] = value

        assert outputs["has_optional"] is False
        assert outputs["optional_value"] is None
        assert outputs["default_value"] == "default value"

    @pytest.mark.asyncio
    async def test_block_with_complex_types(self):
        """Test block with complex input/output types."""

        class ComplexBlock(Block):
            """Block with complex types."""

            class Input(BlockSchemaInput):
                items: list[str] = SchemaField(description="List of items")
                mapping: dict[str, int] = SchemaField(
                    description="String to int mapping"
                )

            class Output(BlockSchemaOutput):
                item_count: int = SchemaField(description="Number of items")
                total_value: int = SchemaField(description="Sum of mapping values")
                combined: list[str] = SchemaField(description="Combined results")

            def __init__(self):
                super().__init__(
                    id="complex-types-block",
                    description="Block with complex types",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=ComplexBlock.Input,
                    output_schema=ComplexBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
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
        outputs = {}
        async for name, value in block.run(
            ComplexBlock.Input(
                items=["apple", "banana", "orange"],
                mapping={"apple": 5, "banana": 3, "orange": 4},
            )
        ):
            outputs[name] = value

        assert outputs["item_count"] == 3
        assert outputs["total_value"] == 12
        assert outputs["combined"] == ["apple: 5", "banana: 3", "orange: 4"]

    @pytest.mark.asyncio
    async def test_block_error_handling(self):
        """Test block error handling."""

        class ErrorHandlingBlock(Block):
            """Block that demonstrates error handling."""

            class Input(BlockSchemaInput):
                value: int = SchemaField(description="Input value")
                should_error: bool = SchemaField(
                    description="Whether to trigger an error",
                    default=False,
                )

            class Output(BlockSchemaOutput):
                result: int = SchemaField(description="Result")
                error_message: Optional[str] = SchemaField(
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

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
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
        outputs = {}
        async for name, value in block.run(
            ErrorHandlingBlock.Input(value=5, should_error=False)
        ):
            outputs[name] = value

        assert outputs["result"] == 10
        assert outputs["error_message"] is None

        # Test with negative value
        outputs = {}
        async for name, value in block.run(
            ErrorHandlingBlock.Input(value=-5, should_error=False)
        ):
            outputs[name] = value

        assert outputs["result"] == 0
        assert outputs["error_message"] == "Value must be non-negative"

        # Test with error
        with pytest.raises(ValueError, match="Intentional error triggered"):
            async for _ in block.run(
                ErrorHandlingBlock.Input(value=5, should_error=True)
            ):
                pass

    @pytest.mark.asyncio
    async def test_block_error_field_override(self):
        """Test block that overrides the automatic error field from BlockSchemaOutput."""

        class ErrorFieldOverrideBlock(Block):
            """Block that defines its own error field with different type."""

            class Input(BlockSchemaInput):
                value: int = SchemaField(description="Input value")

            class Output(BlockSchemaOutput):
                result: int = SchemaField(description="Result")
                # Override the error field with different description/default but same type
                error: str = SchemaField(
                    description="Custom error field with specific validation codes",
                    default="NO_ERROR",
                )

            def __init__(self):
                super().__init__(
                    id="error-field-override-block",
                    description="Block that overrides the error field",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=ErrorFieldOverrideBlock.Input,
                    output_schema=ErrorFieldOverrideBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                if input_data.value < 0:
                    yield "error", "VALIDATION_ERROR:VALUE_NEGATIVE"
                    yield "result", 0
                else:
                    yield "result", input_data.value * 2
                    yield "error", "NO_ERROR"

        # Test alternative approach: Block that doesn't inherit from BlockSchemaOutput
        class FlexibleErrorBlock(Block):
            """Block that defines its own error structure by not inheriting BlockSchemaOutput."""

            class Input(BlockSchemaInput):
                value: int = SchemaField(description="Input value")

            # Use BlockSchemaInput as base to avoid automatic error field
            class Output(BlockSchema):  # Not BlockSchemaOutput!
                result: int = SchemaField(description="Result")
                error: Optional[dict[str, str]] = SchemaField(
                    description="Structured error information",
                    default=None,
                )

            def __init__(self):
                super().__init__(
                    id="flexible-error-block",
                    description="Block with flexible error structure",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=FlexibleErrorBlock.Input,
                    output_schema=FlexibleErrorBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                if input_data.value < 0:
                    yield "error", {
                        "type": "ValidationError",
                        "message": "Value must be non-negative",
                    }
                    yield "result", 0
                else:
                    yield "result", input_data.value * 2
                    yield "error", None

        # Test 1: String-based error override (constrained by BlockSchemaOutput)
        string_error_block = ErrorFieldOverrideBlock()
        outputs = {}
        async for name, value in string_error_block.run(
            ErrorFieldOverrideBlock.Input(value=5)
        ):
            outputs[name] = value

        assert outputs["result"] == 10
        assert outputs["error"] == "NO_ERROR"

        # Test string error with failure
        outputs = {}
        async for name, value in string_error_block.run(
            ErrorFieldOverrideBlock.Input(value=-3)
        ):
            outputs[name] = value

        assert outputs["result"] == 0
        assert outputs["error"] == "VALIDATION_ERROR:VALUE_NEGATIVE"

        # Test 2: Structured error (using BlockSchema base)
        flexible_block = FlexibleErrorBlock()
        outputs = {}
        async for name, value in flexible_block.run(FlexibleErrorBlock.Input(value=5)):
            outputs[name] = value

        assert outputs["result"] == 10
        assert outputs["error"] is None

        # Test structured error with failure
        outputs = {}
        async for name, value in flexible_block.run(FlexibleErrorBlock.Input(value=-3)):
            outputs[name] = value

        assert outputs["result"] == 0
        assert outputs["error"] == {
            "type": "ValidationError",
            "message": "Value must be non-negative",
        }

        # Verify schema differences
        string_schema = string_error_block.output_schema.jsonschema()
        flexible_schema = flexible_block.output_schema.jsonschema()

        # String error field
        string_error_field = string_schema["properties"]["error"]
        assert string_error_field.get("type") == "string"
        assert string_error_field.get("default") == "NO_ERROR"

        # Structured error field
        flexible_error_field = flexible_schema["properties"]["error"]
        # Should be object or anyOf with object/null for Optional[dict]
        assert (
            "anyOf" in flexible_error_field
            or flexible_error_field.get("type") == "object"
        )


class TestAuthenticationVariants:
    """Test complex authentication scenarios including OAuth, API keys, and scopes."""

    @pytest.mark.asyncio
    async def test_oauth_block_with_scopes(self):
        """Test creating a block that uses OAuth2 with scopes."""
        from backend.sdk import OAuth2Credentials, ProviderBuilder

        # Create a test OAuth provider with scopes
        # For testing, we don't need an actual OAuth handler
        # In real usage, you would provide a proper OAuth handler class
        oauth_provider = (
            ProviderBuilder("test_oauth_provider")
            .with_api_key("TEST_OAUTH_API", "Test OAuth API")
            .with_base_cost(5, BlockCostType.RUN)
            .build()
        )

        class OAuthScopedBlock(Block):
            """Block requiring OAuth2 with specific scopes."""

            class Input(BlockSchemaInput):
                credentials: CredentialsMetaInput = oauth_provider.credentials_field(
                    description="OAuth2 credentials with scopes",
                    scopes=["read:user", "write:data"],
                )
                resource: str = SchemaField(description="Resource to access")

            class Output(BlockSchemaOutput):
                data: str = SchemaField(description="Retrieved data")
                scopes_used: list[str] = SchemaField(
                    description="Scopes that were used"
                )
                token_info: dict[str, Any] = SchemaField(
                    description="Token information"
                )

            def __init__(self):
                super().__init__(
                    id="oauth-scoped-block",
                    description="Test OAuth2 with scopes",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=OAuthScopedBlock.Input,
                    output_schema=OAuthScopedBlock.Output,
                )

            async def run(
                self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
            ) -> BlockOutput:
                # Simulate OAuth API call with scopes
                token = credentials.access_token.get_secret_value()

                yield "data", f"OAuth data for {input_data.resource}"
                yield "scopes_used", credentials.scopes or []
                yield "token_info", {
                    "has_token": bool(token),
                    "has_refresh": credentials.refresh_token is not None,
                    "provider": credentials.provider,
                    "expires_at": credentials.access_token_expires_at,
                }

        # Create test OAuth credentials
        test_oauth_creds = OAuth2Credentials(
            id="test-oauth-creds",
            provider="test_oauth_provider",
            access_token=SecretStr("test-access-token"),
            refresh_token=SecretStr("test-refresh-token"),
            scopes=["read:user", "write:data"],
            title="Test OAuth Credentials",
        )

        # Test the block
        block = OAuthScopedBlock()
        outputs = {}
        async for name, value in block.run(
            OAuthScopedBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_oauth_provider",
                    "id": "test-oauth-creds",
                    "type": "oauth2",
                },
                resource="user/profile",
            ),
            credentials=test_oauth_creds,
        ):
            outputs[name] = value

        assert outputs["data"] == "OAuth data for user/profile"
        assert set(outputs["scopes_used"]) == {"read:user", "write:data"}
        assert outputs["token_info"]["has_token"] is True
        assert outputs["token_info"]["expires_at"] is None
        assert outputs["token_info"]["has_refresh"] is True

    @pytest.mark.asyncio
    async def test_mixed_auth_block(self):
        """Test block that supports both OAuth2 and API key authentication."""
        # No need to import these again, already imported at top

        # Create provider supporting both auth types
        # Create provider supporting API key auth
        # In real usage, you would add OAuth support with .with_oauth()
        mixed_provider = (
            ProviderBuilder("mixed_auth_provider")
            .with_api_key("MIXED_API_KEY", "Mixed Provider API Key")
            .with_base_cost(8, BlockCostType.RUN)
            .build()
        )

        class MixedAuthBlock(Block):
            """Block supporting multiple authentication methods."""

            class Input(BlockSchemaInput):
                credentials: CredentialsMetaInput = mixed_provider.credentials_field(
                    description="API key or OAuth2 credentials",
                    supported_credential_types=["api_key", "oauth2"],
                )
                operation: str = SchemaField(description="Operation to perform")

            class Output(BlockSchemaOutput):
                result: str = SchemaField(description="Operation result")
                auth_type: str = SchemaField(description="Authentication type used")
                auth_details: dict[str, Any] = SchemaField(description="Auth details")

            def __init__(self):
                super().__init__(
                    id="mixed-auth-block",
                    description="Block supporting OAuth2 and API key",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=MixedAuthBlock.Input,
                    output_schema=MixedAuthBlock.Output,
                )

            async def run(
                self,
                input_data: Input,
                *,
                credentials: Union[APIKeyCredentials, OAuth2Credentials],
                **kwargs,
            ) -> BlockOutput:
                # Handle different credential types
                if isinstance(credentials, APIKeyCredentials):
                    auth_type = "api_key"
                    auth_details = {
                        "has_key": bool(credentials.api_key.get_secret_value()),
                        "key_prefix": credentials.api_key.get_secret_value()[:5]
                        + "...",
                    }
                elif isinstance(credentials, OAuth2Credentials):
                    auth_type = "oauth2"
                    auth_details = {
                        "has_token": bool(credentials.access_token.get_secret_value()),
                        "scopes": credentials.scopes or [],
                    }
                else:
                    auth_type = "unknown"
                    auth_details = {}

                yield "result", f"Performed {input_data.operation} with {auth_type}"
                yield "auth_type", auth_type
                yield "auth_details", auth_details

        # Test with API key
        api_creds = APIKeyCredentials(
            id="mixed-api-creds",
            provider="mixed_auth_provider",
            api_key=SecretStr("sk-1234567890"),
            title="Mixed API Key",
        )

        block = MixedAuthBlock()
        outputs = {}
        async for name, value in block.run(
            MixedAuthBlock.Input(
                credentials={  # type: ignore
                    "provider": "mixed_auth_provider",
                    "id": "mixed-api-creds",
                    "type": "api_key",
                },
                operation="fetch_data",
            ),
            credentials=api_creds,
        ):
            outputs[name] = value

        assert outputs["auth_type"] == "api_key"
        assert outputs["result"] == "Performed fetch_data with api_key"
        assert outputs["auth_details"]["key_prefix"] == "sk-12..."

        # Test with OAuth2
        oauth_creds = OAuth2Credentials(
            id="mixed-oauth-creds",
            provider="mixed_auth_provider",
            access_token=SecretStr("oauth-token-123"),
            scopes=["full_access"],
            title="Mixed OAuth",
        )

        outputs = {}
        async for name, value in block.run(
            MixedAuthBlock.Input(
                credentials={  # type: ignore
                    "provider": "mixed_auth_provider",
                    "id": "mixed-oauth-creds",
                    "type": "oauth2",
                },
                operation="update_data",
            ),
            credentials=oauth_creds,
        ):
            outputs[name] = value

        assert outputs["auth_type"] == "oauth2"
        assert outputs["result"] == "Performed update_data with oauth2"
        assert outputs["auth_details"]["scopes"] == ["full_access"]

    @pytest.mark.asyncio
    async def test_multiple_credentials_block(self):
        """Test block requiring multiple different credentials."""
        from backend.sdk import ProviderBuilder

        # Create multiple providers
        primary_provider = (
            ProviderBuilder("primary_service")
            .with_api_key("PRIMARY_API_KEY", "Primary Service Key")
            .build()
        )

        # For testing purposes, using API key instead of OAuth handler
        secondary_provider = (
            ProviderBuilder("secondary_service")
            .with_api_key("SECONDARY_API_KEY", "Secondary Service Key")
            .build()
        )

        class MultiCredentialBlock(Block):
            """Block requiring credentials from multiple services."""

            class Input(BlockSchemaInput):
                primary_credentials: CredentialsMetaInput = (
                    primary_provider.credentials_field(
                        description="Primary service API key"
                    )
                )
                secondary_credentials: CredentialsMetaInput = (
                    secondary_provider.credentials_field(
                        description="Secondary service OAuth"
                    )
                )
                merge_data: bool = SchemaField(
                    description="Whether to merge data from both services",
                    default=True,
                )

            class Output(BlockSchemaOutput):
                primary_data: str = SchemaField(description="Data from primary service")
                secondary_data: str = SchemaField(
                    description="Data from secondary service"
                )
                merged_result: Optional[str] = SchemaField(
                    description="Merged data if requested"
                )

            def __init__(self):
                super().__init__(
                    id="multi-credential-block",
                    description="Block using multiple credentials",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=MultiCredentialBlock.Input,
                    output_schema=MultiCredentialBlock.Output,
                )

            async def run(
                self,
                input_data: Input,
                *,
                primary_credentials: APIKeyCredentials,
                secondary_credentials: OAuth2Credentials,
                **kwargs,
            ) -> BlockOutput:
                # Simulate fetching data with primary API key
                primary_data = f"Primary data using {primary_credentials.provider}"
                yield "primary_data", primary_data

                # Simulate fetching data with secondary OAuth
                secondary_data = f"Secondary data with {len(secondary_credentials.scopes or [])} scopes"
                yield "secondary_data", secondary_data

                # Merge if requested
                if input_data.merge_data:
                    merged = f"{primary_data} + {secondary_data}"
                    yield "merged_result", merged
                else:
                    yield "merged_result", None

        # Create test credentials
        primary_creds = APIKeyCredentials(
            id="primary-creds",
            provider="primary_service",
            api_key=SecretStr("primary-key-123"),
            title="Primary Key",
        )

        secondary_creds = OAuth2Credentials(
            id="secondary-creds",
            provider="secondary_service",
            access_token=SecretStr("secondary-token"),
            scopes=["read", "write"],
            title="Secondary OAuth",
        )

        # Test the block
        block = MultiCredentialBlock()
        outputs = {}

        # Note: In real usage, the framework would inject the correct credentials
        # based on the field names. Here we simulate that behavior.
        async for name, value in block.run(
            MultiCredentialBlock.Input(
                primary_credentials={  # type: ignore
                    "provider": "primary_service",
                    "id": "primary-creds",
                    "type": "api_key",
                },
                secondary_credentials={  # type: ignore
                    "provider": "secondary_service",
                    "id": "secondary-creds",
                    "type": "oauth2",
                },
                merge_data=True,
            ),
            primary_credentials=primary_creds,
            secondary_credentials=secondary_creds,
        ):
            outputs[name] = value

        assert outputs["primary_data"] == "Primary data using primary_service"
        assert outputs["secondary_data"] == "Secondary data with 2 scopes"
        assert "Primary data" in outputs["merged_result"]
        assert "Secondary data" in outputs["merged_result"]

    @pytest.mark.asyncio
    async def test_oauth_scope_validation(self):
        """Test OAuth scope validation and handling."""
        from backend.sdk import OAuth2Credentials, ProviderBuilder

        # Provider with specific required scopes
        # For testing OAuth scope validation
        scoped_provider = (
            ProviderBuilder("scoped_oauth_service")
            .with_api_key("SCOPED_OAUTH_KEY", "Scoped OAuth Service")
            .build()
        )

        class ScopeValidationBlock(Block):
            """Block that validates OAuth scopes."""

            class Input(BlockSchemaInput):
                credentials: CredentialsMetaInput = scoped_provider.credentials_field(
                    description="OAuth credentials with specific scopes",
                    scopes=["user:read", "user:write"],  # Required scopes
                )
                require_admin: bool = SchemaField(
                    description="Whether admin scopes are required",
                    default=False,
                )

            class Output(BlockSchemaOutput):
                allowed_operations: list[str] = SchemaField(
                    description="Operations allowed with current scopes"
                )
                missing_scopes: list[str] = SchemaField(
                    description="Scopes that are missing for full access"
                )
                has_required_scopes: bool = SchemaField(
                    description="Whether all required scopes are present"
                )

            def __init__(self):
                super().__init__(
                    id="scope-validation-block",
                    description="Block that validates OAuth scopes",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=ScopeValidationBlock.Input,
                    output_schema=ScopeValidationBlock.Output,
                )

            async def run(
                self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
            ) -> BlockOutput:
                current_scopes = set(credentials.scopes or [])
                required_scopes = {"user:read", "user:write"}

                if input_data.require_admin:
                    required_scopes.update({"admin:read", "admin:write"})

                # Determine allowed operations based on scopes
                allowed_ops = []
                if "user:read" in current_scopes:
                    allowed_ops.append("read_user_data")
                if "user:write" in current_scopes:
                    allowed_ops.append("update_user_data")
                if "admin:read" in current_scopes:
                    allowed_ops.append("read_admin_data")
                if "admin:write" in current_scopes:
                    allowed_ops.append("update_admin_data")

                missing = list(required_scopes - current_scopes)
                has_required = len(missing) == 0

                yield "allowed_operations", allowed_ops
                yield "missing_scopes", missing
                yield "has_required_scopes", has_required

        # Test with partial scopes
        partial_creds = OAuth2Credentials(
            id="partial-oauth",
            provider="scoped_oauth_service",
            access_token=SecretStr("partial-token"),
            scopes=["user:read"],  # Only one of the required scopes
            title="Partial OAuth",
        )

        block = ScopeValidationBlock()
        outputs = {}
        async for name, value in block.run(
            ScopeValidationBlock.Input(
                credentials={  # type: ignore
                    "provider": "scoped_oauth_service",
                    "id": "partial-oauth",
                    "type": "oauth2",
                },
                require_admin=False,
            ),
            credentials=partial_creds,
        ):
            outputs[name] = value

        assert outputs["allowed_operations"] == ["read_user_data"]
        assert "user:write" in outputs["missing_scopes"]
        assert outputs["has_required_scopes"] is False

        # Test with all required scopes
        full_creds = OAuth2Credentials(
            id="full-oauth",
            provider="scoped_oauth_service",
            access_token=SecretStr("full-token"),
            scopes=["user:read", "user:write", "admin:read"],
            title="Full OAuth",
        )

        outputs = {}
        async for name, value in block.run(
            ScopeValidationBlock.Input(
                credentials={  # type: ignore
                    "provider": "scoped_oauth_service",
                    "id": "full-oauth",
                    "type": "oauth2",
                },
                require_admin=False,
            ),
            credentials=full_creds,
        ):
            outputs[name] = value

        assert set(outputs["allowed_operations"]) == {
            "read_user_data",
            "update_user_data",
            "read_admin_data",
        }
        assert outputs["missing_scopes"] == []
        assert outputs["has_required_scopes"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
