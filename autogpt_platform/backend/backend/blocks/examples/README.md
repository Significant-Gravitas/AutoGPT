# Example Blocks Using the SDK Provider Pattern

This directory contains example blocks demonstrating how to use the AutoGPT SDK with the provider builder pattern.

## Provider Builder Pattern

The provider builder pattern is the recommended way to configure providers for your blocks. It provides a clean, declarative way to set up authentication, costs, rate limits, and other provider-specific settings.

### Basic Provider Configuration

Create a `_config.py` file in your block directory:

```python
from backend.sdk import BlockCostType, ProviderBuilder

# Configure your provider
my_provider = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "My Service API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

### Using the Provider in Your Block

Import the provider and use its `credentials_field()` method:

```python
from backend.sdk import Block, BlockSchema, CredentialsMetaInput
from ._config import my_provider

class MyBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = my_provider.credentials_field(
            description="Credentials for My Service"
        )
```

## Examples in This Directory

### 1. Simple Example Block (`simple_example_block.py`)
- Basic block without authentication
- Shows minimal SDK usage

### 2. Example SDK Block (`example_sdk_block.py`)
- Uses provider builder pattern for API key authentication
- Demonstrates credential handling

### 3. Webhook Example Block (`example_webhook_sdk_block.py`)
- Shows webhook configuration with provider pattern
- Includes webhook manager setup

### 4. Advanced Provider Example (`advanced_provider_example.py`)
- Multiple authentication types (API Key and OAuth)
- Custom API client integration
- Rate limiting configuration
- Advanced provider features

## Benefits of the Provider Pattern

1. **Centralized Configuration**: All provider settings in one place
2. **Type Safety**: Full type hints and IDE support
3. **Reusability**: Share provider config across multiple blocks
4. **Consistency**: Standardized pattern across the codebase
5. **Easy Testing**: Mock providers for unit tests

## Best Practices

1. Always create a `_config.py` file for your provider configurations
2. Use descriptive names for your providers
3. Include proper descriptions in `credentials_field()`
4. Set appropriate base costs for your blocks
5. Configure rate limits if your provider has them
6. Use OAuth when available for better security

## Testing

When testing blocks with providers:

```python
from backend.sdk import APIKeyCredentials, SecretStr

# Create test credentials
test_creds = APIKeyCredentials(
    id="test-creds",
    provider="my-service",
    api_key=SecretStr("test-api-key"),
    title="Test API Key",
)

# Use in your tests
block = MyBlock()
async for output_name, output_value in block.run(
    MyBlock.Input(
        credentials={
            "provider": "my-service",
            "id": "test-creds",
            "type": "api_key",
        }
    ),
    credentials=test_creds,
):
    # Process outputs
```