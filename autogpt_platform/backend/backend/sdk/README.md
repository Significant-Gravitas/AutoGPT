# AutoGPT Platform SDK

The AutoGPT Platform SDK simplifies block development by providing a unified interface for creating blocks with authentication, webhooks, cost tracking, and more.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Block Structure](#basic-block-structure)
- [Provider Configuration](#provider-configuration)
  - [API Key Authentication](#api-key-authentication)
  - [OAuth Authentication](#oauth-authentication)
  - [Multiple Authentication Methods](#multiple-authentication-methods)
- [Cost Management](#cost-management)
  - [Default Provider Costs](#default-provider-costs)
  - [Block-Specific Costs](#block-specific-costs)
  - [Cost Types](#cost-types)
  - [Tiered and Conditional Costs](#tiered-and-conditional-costs)
- [Webhooks](#webhooks)
  - [Auto-Managed Webhooks](#auto-managed-webhooks)
  - [Manual Webhooks](#manual-webhooks)
- [Advanced Features](#advanced-features)
  - [Custom API Clients](#custom-api-clients)
  - [Test Credentials](#test-credentials)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

## Quick Start

The SDK uses a single import pattern instead of multiple complex imports:

```python
from backend.sdk import (
    Block, 
    BlockCategory, 
    BlockOutput, 
    BlockSchema,
    SchemaField, 
)

# Additional standard library imports as needed
from typing import Optional, Literal  # For type hints
from enum import Enum  # For enumerations
```

## Basic Block Structure

Every block inherits from the `Block` base class and defines input/output schemas:

```python
from backend.sdk import Block, BlockCategory, BlockOutput, BlockSchema, SchemaField

class MyBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(description="Input text to process")
        count: int = SchemaField(description="Number of times to repeat", default=1)
    
    class Output(BlockSchema):
        result: str = SchemaField(description="Processed result")
    
    def __init__(self):
        super().__init__(
            id="unique-uuid-here",  # Generate with uuid.uuid4()
            description="My block description",
            categories={BlockCategory.TEXT},
            input_schema=MyBlock.Input,
            output_schema=MyBlock.Output,
        )
    
    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        result = input_data.text * input_data.count
        yield "result", result
```

## Provider Configuration

Providers manage authentication, costs, and API client configuration. Create a `_config.py` file in your block directory:

### API Key Authentication

```python
from backend.sdk import BlockCostType, ProviderBuilder

# API key from environment variable
my_service = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "My Service API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

Use in your block:

```python
from backend.sdk import APIKeyCredentials, Block, BlockSchema, CredentialsMetaInput, SchemaField
from ._config import my_service

class MyServiceBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = my_service.credentials_field(
            description="API credentials for My Service"
        )
        query: str = SchemaField(description="Query to process")
    
    async def run(
        self, 
        input_data: Input, 
        *, 
        credentials: APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        # Use api_key to make API calls
        yield "result", f"Processed {input_data.query}"
```

### OAuth Authentication

OAuth requires an OAuth handler class:

```python
from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, SecretStr
from typing import Optional
from urllib.parse import urlencode

class MyServiceOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName("my-service")
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://api.myservice.com/oauth/authorize"
        self.token_url = "https://api.myservice.com/oauth/token"
    
    def get_login_url(self, scopes: list[str], state: str, code_challenge: Optional[str]) -> str:
        # Build and return OAuth login URL
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        return f"{self.auth_base_url}?{urlencode(params)}"
    
    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        # Exchange authorization code for tokens
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }
        # Make request and return OAuth2Credentials
        response = await Requests().post(self.token_url, json=data)
        tokens = response.json()
        
        return OAuth2Credentials(
            provider="my-service",
            access_token=SecretStr(tokens["access_token"]),
            refresh_token=SecretStr(tokens.get("refresh_token")),
            expires_at=tokens.get("expires_at"),
            scopes=scopes,
            title="My Service OAuth"
        )
```

Configure the provider with OAuth:

```python
# In _config.py
import os
from backend.sdk import BlockCostType, ProviderBuilder
from ._oauth import MyServiceOAuthHandler

# Check if OAuth is configured
client_id = os.getenv("MY_SERVICE_CLIENT_ID")
client_secret = os.getenv("MY_SERVICE_CLIENT_SECRET")
OAUTH_IS_CONFIGURED = bool(client_id and client_secret)

# Build provider
builder = ProviderBuilder("my-service").with_base_cost(1, BlockCostType.RUN)

if OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        MyServiceOAuthHandler,
        scopes=["read", "write"]
    )

my_service = builder.build()
```

### Username/Password Authentication

For services that use HTTP Basic Auth or username/password authentication:

```python
# In _config.py
from backend.sdk import BlockCostType, ProviderBuilder

my_service = (
    ProviderBuilder("my-service")
    .with_user_password("MY_SERVICE_USERNAME", "MY_SERVICE_PASSWORD", "My Service Credentials")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

In your blocks, the credentials will be provided as `UserPasswordCredentials`:

```python
from backend.sdk import UserPasswordCredentials

async def run(
    self, 
    input_data: Input, 
    *, 
    credentials: UserPasswordCredentials,
    **kwargs
) -> BlockOutput:
    username = credentials.username.get_secret_value()
    password = credentials.password.get_secret_value()
    # Use for HTTP Basic Auth or other authentication
```

### Multiple Authentication Methods

Providers built with ProviderBuilder can support multiple authentication methods:

```python
my_service = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "My Service API Key")
    .with_oauth(MyServiceOAuthHandler, scopes=["read", "write"])
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

## Cost Management

### Default Provider Costs

Set a base cost that applies to all blocks using the provider:

```python
my_service = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "API Key")
    .with_base_cost(1, BlockCostType.RUN)  # 1 credit per run
    .build()
)
```

### Block-Specific Costs

Override provider costs using the `@cost` decorator:

```python
from backend.sdk import cost, BlockCost, BlockCostType

@cost(BlockCost(cost_type=BlockCostType.RUN, cost_amount=5))
class ExpensiveBlock(Block):
    # This block costs 5 credits per run, overriding provider default
    ...
```

### Cost Types

The SDK supports different cost calculation methods:

```python
# Fixed cost per run
@cost(BlockCost(cost_type=BlockCostType.RUN, cost_amount=10))

# Cost based on data size (per byte)
@cost(BlockCost(cost_type=BlockCostType.BYTE, cost_amount=0.001))

# Cost based on execution time (per second)
@cost(BlockCost(cost_type=BlockCostType.SECOND, cost_amount=0.1))
```

### Tiered and Conditional Costs

Define multiple costs with filters for tiered pricing:

```python
from backend.sdk import cost, BlockCost, BlockCostType
from enum import Enum

class ServiceTier(str, Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@cost(
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=1,
        cost_filter={"tier": "basic"}
    ),
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=5,
        cost_filter={"tier": "premium"}
    ),
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=20,
        cost_filter={"tier": "enterprise"}
    )
)
class TieredServiceBlock(Block):
    class Input(BlockSchema):
        tier: ServiceTier = SchemaField(
            description="Service tier",
            default=ServiceTier.BASIC
        )
        # ... other fields
```

## Webhooks

### Auto-Managed Webhooks

For services that require webhook registration/deregistration:

```python
from backend.sdk import BaseWebhooksManager, Webhook, ProviderName
from enum import Enum

class MyServiceWebhookManager(BaseWebhooksManager):
    PROVIDER_NAME = ProviderName("my-service")
    
    class WebhookType(str, Enum):
        DATA_UPDATE = "data_update"
        STATUS_CHANGE = "status_change"
    
    async def validate_payload(self, webhook: Webhook, request) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()
        event_type = request.headers.get("X-Event-Type", "unknown")
        return payload, event_type
    
    async def _register_webhook(
        self,
        credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        # Register webhook with external service
        # Return (webhook_id, metadata)
        api_key = credentials.api_key.get_secret_value()
        response = await Requests().post(
            "https://api.myservice.com/webhooks",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "url": ingress_url,
                "events": events,
                "secret": secret,
            }
        )
        data = response.json()
        return data["id"], {"events": events}
        
    async def _deregister_webhook(self, webhook: Webhook, credentials) -> None:
        # Deregister webhook from external service
        api_key = credentials.api_key.get_secret_value()
        await Requests().delete(
            f"https://api.myservice.com/webhooks/{webhook.provider_webhook_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
```

Configure the provider:

```python
my_service = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "API Key")
    .with_webhook_manager(MyServiceWebhookManager)
    .build()
)
```

Use in a webhook block:

```python
from backend.sdk import Block, BlockType, BlockWebhookConfig, ProviderName

class MyWebhookBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = my_service.credentials_field(
            description="Credentials for webhook service"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        event_filter: dict = SchemaField(
            description="Filter for specific events",
            default={}
        )
        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True
        )
    
    def __init__(self):
        super().__init__(
            id="unique-webhook-block-id",
            description="Receives webhooks from My Service",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("my-service"),
                webhook_type="data_update",
                event_filter_input="event_filter",
                resource_format="{resource_id}",
            ),
        )
```

### Manual Webhooks

For simple webhooks that don't require registration:

```python
from backend.sdk import BlockManualWebhookConfig, ManualWebhookManagerBase

class SimpleWebhookBlock(Block):
    def __init__(self):
        super().__init__(
            id="simple-webhook-id",
            description="Simple webhook receiver",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockManualWebhookConfig(
                provider=ProviderName("generic_webhook"),
                webhook_type="plain",
            ),
        )
```

## Advanced Features

### Custom API Clients

Provide a custom API client factory:

```python
from backend.sdk import Requests

class MyAPIClient:
    def __init__(self, credentials):
        self.api_key = credentials.api_key.get_secret_value()
        self.base_url = "https://api.myservice.com"
    
    async def request(self, method: str, endpoint: str, **kwargs):
        # Implement API request logic
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.api_key}"
        kwargs["headers"] = headers
        
        response = await Requests().request(
            method,
            f"{self.base_url}{endpoint}",
            **kwargs
        )
        return response.json()

my_service = (
    ProviderBuilder("my-service")
    .with_api_key("MY_SERVICE_API_KEY", "API Key")
    .with_api_client(lambda creds: MyAPIClient(creds))
    .build()
)
```

Use the API client in your block:

```python
async def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
    api_client = my_service.get_api(credentials)
    result = await api_client.request("GET", "/data")
    yield "result", result
```


### Test Credentials

Define test credentials for development and testing:

```python
# In _config.py
from backend.sdk import APIKeyCredentials, SecretStr

MY_SERVICE_TEST_CREDENTIALS = APIKeyCredentials(
    id="test-creds-id",
    provider="my-service",
    api_key=SecretStr("test-api-key"),
    title="Test API Key",
    expires_at=None,
)

MY_SERVICE_TEST_CREDENTIALS_INPUT = {
    "provider": MY_SERVICE_TEST_CREDENTIALS.provider,
    "id": MY_SERVICE_TEST_CREDENTIALS.id,
    "type": MY_SERVICE_TEST_CREDENTIALS.type,
    "title": MY_SERVICE_TEST_CREDENTIALS.title,
}
```

## Best Practices

1. **Always use `_config.py`**: Keep provider configuration separate from block logic
2. **Generate unique UUIDs**: Use `uuid.uuid4()` for block IDs
3. **Set appropriate costs**: Consider API pricing when setting block costs
4. **Handle errors gracefully**: Always wrap API calls in try-except blocks
5. **Document thoroughly**: Use clear descriptions for all fields
6. **Test with mock credentials**: Create test credentials for unit tests
7. **Follow naming conventions**: Use lowercase with underscores for provider names
8. **Check environment variables**: Verify OAuth credentials before configuring

## Complete Examples

### Simple API Key Block

```python
# _config.py
from backend.sdk import BlockCostType, ProviderBuilder

weather_api = (
    ProviderBuilder("weather_api")
    .with_api_key("WEATHER_API_KEY", "Weather API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

# weather_block.py
from backend.sdk import *
from ._config import weather_api

class WeatherBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = weather_api.credentials_field(
            description="Weather API credentials"
        )
        city: str = SchemaField(description="City name")
    
    class Output(BlockSchema):
        temperature: float = SchemaField(description="Temperature in Celsius")
        description: str = SchemaField(description="Weather description")
    
    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
            description="Get current weather for a city",
            categories={BlockCategory.SEARCH},
            input_schema=WeatherBlock.Input,
            output_schema=WeatherBlock.Output,
        )
    
    async def run(
        self, 
        input_data: Input, 
        *, 
        credentials: APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        # Make API call with api_key
        response = await Requests().get(
            f"https://api.weather.com/v1/current",
            params={"city": input_data.city, "api_key": api_key}
        )
        data = response.json()
        yield "temperature", data["temp"]
        yield "description", data["desc"]
```

### OAuth Block with Custom Costs

```python
# _oauth.py
from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, Requests, SecretStr
from typing import Optional
from urllib.parse import urlencode

class SocialOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName("social-api")
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    def get_login_url(self, scopes: list[str], state: str, code_challenge: Optional[str]) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        return f"https://social-api.com/oauth/authorize?{urlencode(params)}"
    
    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        response = await Requests().post(
            "https://social-api.com/oauth/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
            }
        )
        tokens = response.json()
        
        return OAuth2Credentials(
            provider="social-api",
            access_token=SecretStr(tokens["access_token"]),
            refresh_token=SecretStr(tokens.get("refresh_token")),
            expires_at=tokens.get("expires_at"),
            scopes=scopes,
            title="Social API OAuth"
        )

# _config.py
import os
from backend.sdk import BlockCostType, ProviderBuilder
from ._oauth import SocialOAuthHandler

social_api = (
    ProviderBuilder("social-api")
    .with_oauth(SocialOAuthHandler, scopes=["read", "write"])
    .with_base_cost(2, BlockCostType.RUN)
    .build()
)

# social_block.py
from backend.sdk import *
from typing import Literal
from ._config import social_api

@cost(
    BlockCost(cost_type=BlockCostType.RUN, cost_amount=5, cost_filter={"action": "post"}),
    BlockCost(cost_type=BlockCostType.RUN, cost_amount=1, cost_filter={"action": "read"})
)
class SocialBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = social_api.credentials_field(
            description="Social API OAuth credentials"
        )
        action: Literal["read", "post"] = SchemaField(
            description="Action to perform",
            default="read"
        )
        content: str = SchemaField(description="Content for the action")
    
    class Output(BlockSchema):
        result: str = SchemaField(description="Action result")
        cost_applied: int = SchemaField(description="Credits charged")
    
    def __init__(self):
        super().__init__(
            id="b2c3d4e5-6789-01ab-cdef-234567890abc",
            description="Interact with social media API",
            categories={BlockCategory.SOCIAL},
            input_schema=SocialBlock.Input,
            output_schema=SocialBlock.Output,
        )
    
    async def run(
        self, 
        input_data: Input, 
        *, 
        credentials: OAuth2Credentials,
        **kwargs
    ) -> BlockOutput:
        # Use OAuth2 credentials
        access_token = credentials.access_token.get_secret_value()
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        if input_data.action == "post":
            # Costs 5 credits
            response = await Requests().post(
                "https://social-api.com/v1/posts",
                headers=headers,
                json={"content": input_data.content}
            )
            yield "result", f"Posted: {input_data.content}"
            yield "cost_applied", 5
        else:
            # Costs 1 credit
            response = await Requests().get(
                "https://social-api.com/v1/timeline",
                headers=headers
            )
            yield "result", f"Read timeline: {len(response.json())} items"
            yield "cost_applied", 1
```

### Webhook Block with Event Filtering

```python
# _webhook.py
from backend.sdk import BaseWebhooksManager, Webhook, ProviderName, Requests
from enum import Enum

class DataServiceWebhookManager(BaseWebhooksManager):
    PROVIDER_NAME = ProviderName("data-service")
    
    class WebhookType(str, Enum):
        DATA_CHANGE = "data_change"
    
    async def validate_payload(self, webhook: Webhook, request) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()
        event_type = request.headers.get("X-Event-Type", "unknown")
        
        # Verify webhook signature if needed
        signature = request.headers.get("X-Signature")
        # ... signature validation logic
        
        return payload, event_type
    
    async def _register_webhook(
        self,
        credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        api_key = credentials.api_key.get_secret_value()
        response = await Requests().post(
            f"https://api.dataservice.com/webhooks",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "url": ingress_url,
                "resource": resource,
                "events": events,
                "secret": secret,
            }
        )
        webhook_data = response.json()
        return webhook_data["id"], {"resource": resource, "events": events}
    
    async def _deregister_webhook(self, webhook: Webhook, credentials) -> None:
        api_key = credentials.api_key.get_secret_value()
        await Requests().delete(
            f"https://api.dataservice.com/webhooks/{webhook.provider_webhook_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )

# _config.py
from backend.sdk import BlockCostType, ProviderBuilder
from ._webhook import DataServiceWebhookManager

data_service = (
    ProviderBuilder("data-service")
    .with_api_key("DATA_SERVICE_API_KEY", "Data Service API Key")
    .with_webhook_manager(DataServiceWebhookManager)
    .with_base_cost(0, BlockCostType.RUN)  # Webhooks typically free
    .build()
)

# data_webhook_block.py
from backend.sdk import *
from ._config import data_service

class DataWebhookBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = data_service.credentials_field(
            description="Data Service credentials"
        )
        webhook_url: str = SchemaField(
            description="Webhook URL (auto-generated)",
            default="",
            hidden=True,
        )
        resource_id: str = SchemaField(
            description="Resource ID to monitor",
            default=""
        )
        event_types: list[str] = SchemaField(
            description="Event types to listen for",
            default=["created", "updated", "deleted"]
        )
        payload: dict = SchemaField(
            description="Webhook payload",
            default={},
            hidden=True
        )
    
    class Output(BlockSchema):
        event_type: str = SchemaField(description="Type of event")
        resource_id: str = SchemaField(description="ID of affected resource")
        data: dict = SchemaField(description="Event data")
        timestamp: str = SchemaField(description="Event timestamp")
    
    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7890-12ab-cdef-345678901234",
            description="Receives data change events via webhook",
            categories={BlockCategory.INPUT},
            input_schema=DataWebhookBlock.Input,
            output_schema=DataWebhookBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("data-service"),
                webhook_type="data_change",
                event_filter_input="event_types",
                resource_format="{resource_id}",
            ),
        )
    
    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        
        # Extract event details
        event_type = payload.get("event_type", "unknown")
        resource_id = payload.get("resource_id", input_data.resource_id)
        
        # Filter events if needed
        if event_type not in input_data.event_types:
            return  # Skip unwanted events
        
        yield "event_type", event_type
        yield "resource_id", resource_id
        yield "data", payload.get("data", {})
        yield "timestamp", payload.get("timestamp", "")
```

### Real-World Example: Exa Webhook Integration

```python
# _webhook.py
from enum import Enum
import hashlib
import hmac

from backend.sdk import (
    BaseWebhooksManager,
    Webhook,
    ProviderName,
    Requests,
    APIKeyCredentials,
)
from backend.data.model import Credentials

class ExaWebhookManager(BaseWebhooksManager):
    """Webhook manager for Exa API."""
    
    PROVIDER_NAME = ProviderName("exa")
    
    class WebhookType(str, Enum):
        WEBSET = "webset"
    
    @classmethod
    async def validate_payload(cls, webhook: Webhook, request) -> tuple[dict, str]:
        """Validate incoming webhook payload and signature."""
        payload = await request.json()
        event_type = payload.get("eventType", "unknown")
        
        # Verify webhook signature if secret is available
        if webhook.secret:
            signature = request.headers.get("X-Exa-Signature")
            if signature:
                body = await request.body()
                expected_signature = hmac.new(
                    webhook.secret.encode(),
                    body,
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(signature, expected_signature):
                    raise ValueError("Invalid webhook signature")
        
        return payload, event_type
    
    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register webhook with Exa API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Exa webhooks require API key credentials")
        api_key = credentials.api_key.get_secret_value()
        
        response = await Requests().post(
            "https://api.exa.ai/v0/webhooks",
            headers={"x-api-key": api_key},
            json={
                "url": ingress_url,
                "events": events,
                "metadata": {"resource": resource}
            }
        )
        
        webhook_data = response.json()
        return webhook_data["id"], {
            "events": events,
            "exa_secret": webhook_data.get("secret"),
        }
    
    async def _deregister_webhook(self, webhook: Webhook, credentials: Credentials) -> None:
        """Deregister webhook from Exa API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Exa webhooks require API key credentials")
        api_key = credentials.api_key.get_secret_value()
        
        await Requests().delete(
            f"https://api.exa.ai/v0/webhooks/{webhook.provider_webhook_id}",
            headers={"x-api-key": api_key}
        )

# _config.py
from backend.sdk import BlockCostType, ProviderBuilder
from ._webhook import ExaWebhookManager

exa = (
    ProviderBuilder("exa")
    .with_api_key("EXA_API_KEY", "Exa API Key")
    .with_webhook_manager(ExaWebhookManager)
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

# webhook_blocks.py
from backend.sdk import *
from ._config import exa

class ExaWebsetWebhookBlock(Block):
    """Receives webhook notifications for Exa webset events."""
    
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="Exa API credentials"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        webset_id: str = SchemaField(
            description="The webset ID to monitor",
            default="",
        )
        event_filter: dict = SchemaField(
            description="Configure which events to receive",
            default={}
        )
        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True
        )
    
    class Output(BlockSchema):
        event_type: str = SchemaField(description="Type of event")
        webset_id: str = SchemaField(description="ID of the affected webset")
        data: dict = SchemaField(description="Event data")
    
    def __init__(self):
        super().__init__(
            id="d1e2f3a4-b5c6-7d8e-9f0a-1b2c3d4e5f6a",
            description="Receive notifications for Exa webset events",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("exa"),
                webhook_type="webset",
                event_filter_input="event_filter",
                resource_format="{webset_id}",
            ),
        )
    
    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "event_type", payload.get("eventType", "unknown")
        yield "webset_id", payload.get("websetId", input_data.webset_id)
        yield "data", payload.get("data", {})
```

For more examples, see the `/autogpt_platform/backend/backend/blocks/examples/` directory.