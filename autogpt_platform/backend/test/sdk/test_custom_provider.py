# noqa: F405
"""
Test custom provider functionality in the SDK.

This test suite verifies that the SDK properly supports dynamic provider
registration and that custom providers work correctly with the system.
"""


from backend.integrations.providers import ProviderName
from backend.sdk import *  # noqa: F403
from backend.sdk.auto_registry import get_registry
from backend.util.test import execute_block_test

# Test credentials for custom providers
CUSTOM_TEST_CREDENTIALS = APIKeyCredentials(
    id="custom-provider-test-creds",
    provider="my-custom-service",
    api_key=SecretStr("test-api-key-12345"),
    title="Custom Service Test Credentials",
    expires_at=None,
)

CUSTOM_TEST_CREDENTIALS_INPUT = {
    "provider": CUSTOM_TEST_CREDENTIALS.provider,
    "id": CUSTOM_TEST_CREDENTIALS.id,
    "type": CUSTOM_TEST_CREDENTIALS.type,
    "title": CUSTOM_TEST_CREDENTIALS.title,
}


@provider("my-custom-service")
@cost_config(
    BlockCost(cost_amount=10, cost_type=BlockCostType.RUN),
    BlockCost(cost_amount=2, cost_type=BlockCostType.BYTE),
)
@default_credentials(
    APIKeyCredentials(
        id="my-custom-service-default",
        provider="my-custom-service",
        api_key=SecretStr("default-custom-api-key"),
        title="My Custom Service Default API Key",
        expires_at=None,
    )
)
class CustomProviderBlock(Block):
    """Test block with a completely custom provider."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="my-custom-service",
            supported_credential_types={"api_key"},
            description="Custom service credentials",
        )
        message: String = SchemaField(
            description="Message to process", default="Hello from custom provider!"
        )

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed message")
        provider_used: String = SchemaField(description="Provider name used")
        credentials_valid: Boolean = SchemaField(
            description="Whether credentials were valid"
        )

    def __init__(self):
        super().__init__(
            id="d1234567-89ab-cdef-0123-456789abcdef",
            description="Test block demonstrating custom provider support",
            categories={BlockCategory.TEXT},
            input_schema=CustomProviderBlock.Input,
            output_schema=CustomProviderBlock.Output,
            test_input={
                "credentials": CUSTOM_TEST_CREDENTIALS_INPUT,
                "message": "Test message",
            },
            test_output=[
                ("result", "CUSTOM: Test message"),
                ("provider_used", "my-custom-service"),
                ("credentials_valid", True),
            ],
            test_credentials=CUSTOM_TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Verify we got the right credentials
        api_key = credentials.api_key.get_secret_value()

        yield "result", f"CUSTOM: {input_data.message}"
        yield "provider_used", credentials.provider
        yield "credentials_valid", bool(api_key)


@provider("another-custom-provider")
@cost_config(
    BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
)
class AnotherCustomProviderBlock(Block):
    """Another test block to verify multiple custom providers work."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="another-custom-provider",
            supported_credential_types={"api_key"},
        )
        data: String = SchemaField(description="Input data")

    class Output(BlockSchema):
        processed: String = SchemaField(description="Processed data")

    def __init__(self):
        super().__init__(
            id="e2345678-9abc-def0-1234-567890abcdef",
            description="Another custom provider test",
            categories={BlockCategory.TEXT},
            input_schema=AnotherCustomProviderBlock.Input,
            output_schema=AnotherCustomProviderBlock.Output,
            test_input={
                "credentials": {
                    "provider": "another-custom-provider",
                    "id": "test-creds-2",
                    "type": "api_key",
                    "title": "Test Creds 2",
                },
                "data": "test data",
            },
            test_output=[("processed", "ANOTHER: test data")],
            test_credentials=APIKeyCredentials(
                id="test-creds-2",
                provider="another-custom-provider",
                api_key=SecretStr("another-test-key"),
                title="Another Test Key",
                expires_at=None,
            ),
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        yield "processed", f"ANOTHER: {input_data.data}"


class TestCustomProvider:
    """Test suite for custom provider functionality."""

    def test_custom_provider_enum_accepts_any_string(self):
        """Test that ProviderName enum accepts any string value."""
        # Test with a completely new provider name
        custom_provider = ProviderName("my-totally-new-provider")
        assert custom_provider.value == "my-totally-new-provider"

        # Test with existing provider
        existing_provider = ProviderName.OPENAI
        assert existing_provider.value == "openai"

        # Test comparison
        another_custom = ProviderName("my-totally-new-provider")
        assert custom_provider == another_custom

    def test_custom_provider_block_executes(self):
        """Test that blocks with custom providers can execute properly."""
        block = CustomProviderBlock()
        execute_block_test(block)

    def test_multiple_custom_providers(self):
        """Test that multiple custom providers can coexist."""
        block1 = CustomProviderBlock()
        block2 = AnotherCustomProviderBlock()

        # Both blocks should execute successfully
        execute_block_test(block1)
        execute_block_test(block2)

    def test_custom_provider_registration(self):
        """Test that custom providers are registered in the auto-registry."""
        registry = get_registry()

        # Check that our custom provider blocks have registered their costs
        block_costs = registry.get_block_costs_dict()
        assert CustomProviderBlock in block_costs
        assert AnotherCustomProviderBlock in block_costs

        # Check the costs are correct
        custom_costs = block_costs[CustomProviderBlock]
        assert len(custom_costs) == 2
        assert any(
            cost.cost_amount == 10 and cost.cost_type == BlockCostType.RUN
            for cost in custom_costs
        )
        assert any(
            cost.cost_amount == 2 and cost.cost_type == BlockCostType.BYTE
            for cost in custom_costs
        )

    def test_custom_provider_default_credentials(self):
        """Test that default credentials are registered for custom providers."""
        registry = get_registry()
        default_creds = registry.get_default_credentials_list()

        # Check that our custom provider's default credentials are registered
        custom_default_creds = [
            cred for cred in default_creds if cred.provider == "my-custom-service"
        ]
        assert len(custom_default_creds) >= 1
        assert custom_default_creds[0].id == "my-custom-service-default"

    def test_custom_provider_with_oauth(self):
        """Test that custom providers can use OAuth handlers."""
        # This is a placeholder for OAuth testing
        # In a real implementation, you would create a custom OAuth handler
        pass

    def test_custom_provider_with_webhooks(self):
        """Test that custom providers can use webhook managers."""
        # This is a placeholder for webhook testing
        # In a real implementation, you would create a custom webhook manager
        pass


# Test that runs as part of pytest
def test_custom_provider_functionality():
    """Run all custom provider tests."""
    test_instance = TestCustomProvider()

    # Run each test method
    test_instance.test_custom_provider_enum_accepts_any_string()
    test_instance.test_custom_provider_block_executes()
    test_instance.test_multiple_custom_providers()
    test_instance.test_custom_provider_registration()
    test_instance.test_custom_provider_default_credentials()
