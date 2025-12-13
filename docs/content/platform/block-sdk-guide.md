# Block Creation with SDK

This guide explains how to create new blocks for the AutoGPT Platform using the SDK pattern with advanced features.

## Overview

Blocks are reusable components that perform specific tasks in AutoGPT workflows. They can integrate with external services, process data, or perform any programmatic operation.

## Basic Structure

### 1. Create Provider Configuration

First, create a `_config.py` file to configure your provider using the `ProviderBuilder`:

```python
from backend.sdk import BlockCostType, ProviderBuilder

# Simple API key provider
my_provider = (
    ProviderBuilder("my_provider")
    .with_api_key("MY_PROVIDER_API_KEY", "My Provider API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

For OAuth providers:

```python
from backend.sdk import BlockCostType, ProviderBuilder
from ._oauth import MyProviderOAuthHandler

my_provider = (
    ProviderBuilder("my_provider")
    .with_oauth(
        MyProviderOAuthHandler,
        scopes=["read", "write"],
        client_id_env_var="MY_PROVIDER_CLIENT_ID",
        client_secret_env_var="MY_PROVIDER_CLIENT_SECRET",
    )
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

### 2. Create the Block Class

Create your block file (e.g., `my_block.py`):

```python
import uuid
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from ._config import my_provider


class MyBlock(Block):
    class Input(BlockSchemaInput):
        # Define input fields
        credentials: CredentialsMetaInput = my_provider.credentials_field(
            description="API credentials for My Provider"
        )
        query: str = SchemaField(description="The query to process")
        limit: int = SchemaField(
            description="Number of results", 
            default=10,
            ge=1,  # Greater than or equal to 1
            le=100  # Less than or equal to 100
        )
        advanced_option: str = SchemaField(
            description="Advanced setting",
            default="",
            advanced=True  # Hidden by default in UI
        )

    class Output(BlockSchemaOutput):
        # Define output fields
        results: list = SchemaField(description="List of results")
        count: int = SchemaField(description="Total count")
        # error output pin is already defined on BlockSchemaOutput

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),  # Generate unique ID
            description="Brief description of what this block does",
            categories={BlockCategory.SEARCH},  # Choose appropriate categories
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, 
        input_data: Input, 
        *, 
        credentials: APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        try:
            # Your block logic here
            results = await self.process_data(
                input_data.query,
                input_data.limit,
                credentials
            )
            
            # Yield outputs
            yield "results", results
            yield "count", len(results)
            
        except Exception as e:
            yield "error", str(e)

    async def process_data(self, query, limit, credentials):
        # Implement your logic
        # Use credentials.api_key.get_secret_value() to access the API key
        pass
```

## Key Components Explained

### Provider Configuration

The `ProviderBuilder` allows you to:
- **`.with_api_key()`**: Add API key authentication
- **`.with_oauth()`**: Add OAuth authentication  
- **`.with_base_cost()`**: Set resource costs for the block
- **`.with_webhook_manager()`**: Add webhook support
- **`.with_user_password()`**: Add username/password auth

### Block Schema

- **Input/Output classes**: Define the data structure using `BlockSchema`
- **SchemaField**: Define individual fields with validation
- **CredentialsMetaInput**: Special field for handling credentials

### Block Implementation

1. **Unique ID**: Generate using `uuid.uuid4()`
2. **Categories**: Choose from `BlockCategory` enum (e.g., SEARCH, AI, PRODUCTIVITY)
3. **async run()**: Main execution method that yields outputs
4. **Error handling**: Error output pin is already defined on BlockSchemaOutput

## Advanced Features

### Testing

Add test configuration to your block:

```python
def __init__(self):
    super().__init__(
        # ... other config ...
        test_input={
            "query": "test query",
            "limit": 5,
            "credentials": {
                "provider": "my_provider",
                "id": str(uuid.uuid4()),
                "type": "api_key"
            }
        },
        test_output=[
            ("results", ["result1", "result2"]),
            ("count", 2)
        ],
        test_mock={
            "process_data": lambda *args, **kwargs: ["result1", "result2"]
        }
    )
```

### OAuth Support

Create an OAuth handler in `_oauth.py`:

```python
from backend.integrations.oauth.base import BaseOAuthHandler

class MyProviderOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = "my_provider"
    
    def _get_authorization_url(self, scopes: list[str], state: str) -> str:
        # Implementation
        pass
    
    def _exchange_code_for_token(self, code: str, scopes: list[str]) -> dict:
        # Implementation
        pass
```

### Webhook Support

Create a webhook manager in `_webhook.py`:

```python
from backend.integrations.webhooks._base import BaseWebhooksManager

class MyProviderWebhookManager(BaseWebhooksManager):
    PROVIDER_NAME = "my_provider"
    
    async def validate_event(self, event: dict) -> bool:
        # Implementation
        pass
```

## File Organization

```
backend/blocks/my_provider/
├── __init__.py          # Export your blocks
├── _config.py           # Provider configuration  
├── _oauth.py           # OAuth handler (optional)
├── _webhook.py         # Webhook manager (optional)
├── _api.py             # API client wrapper (optional)
├── models.py           # Data models (optional)
└── my_block.py         # Block implementations
```

## Best Practices

1. **Error Handling**: Error output pin is already defined on BlockSchemaOutput
2. **Credentials**: Use the provider's `credentials_field()` method
3. **Validation**: Use SchemaField constraints (ge, le, min_length, etc.)
4. **Categories**: Choose appropriate categories for discoverability
5. **Advanced Fields**: Mark complex options as `advanced=True`
6. **Async Operations**: Use `async`/`await` for I/O operations
7. **API Clients**: Use `Requests()` from SDK or external libraries
8. **Testing**: Include test inputs/outputs for validation

## Common Patterns

### Making API Requests

```python
from backend.sdk import Requests

async def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
    headers = {
        "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        "Content-Type": "application/json"
    }
    
    response = await Requests().post(
        "https://api.example.com/endpoint",
        headers=headers,
        json={"query": input_data.query}
    )
    
    data = response.json()
    yield "results", data.get("results", [])
```

### Multiple Auth Types

```python
async def run(
    self, 
    input_data: Input, 
    *, 
    credentials: OAuth2Credentials | APIKeyCredentials,
    **kwargs
):
    if isinstance(credentials, OAuth2Credentials):
        # Handle OAuth
        token = credentials.access_token.get_secret_value()
    else:
        # Handle API key
        token = credentials.api_key.get_secret_value()
```

## Testing Your Block

```bash
# Run all block tests
poetry run pytest backend/blocks/test/test_block.py -xvs

# Test specific block
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[MyBlock]' -xvs
```

## Integration Checklist

- [ ] Create provider configuration in `_config.py`
- [ ] Implement block class with Input/Output schemas
- [ ] Generate unique block ID with `uuid.uuid4()`
- [ ] Choose appropriate block categories
- [ ] Implement `async run()` method
- [ ] Handle errors gracefully
- [ ] Add test configuration
- [ ] Export block in `__init__.py`
- [ ] Test the block
- [ ] Document any special requirements

## Example Blocks for Reference

- **Simple API**: `/backend/blocks/firecrawl/` - Basic API key authentication
- **OAuth + API**: `/backend/blocks/linear/` - OAuth and API key support
- **Webhooks**: `/backend/blocks/exa/` - Includes webhook manager

Study these examples to understand different patterns and approaches for building blocks.