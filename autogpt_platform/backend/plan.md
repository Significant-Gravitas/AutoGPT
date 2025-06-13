# Block SDK Implementation Plan

## Overview

This plan outlines the implementation of a new Block SDK system that provides auto-registration and keeps all configuration self-contained within each block's folder. The goal is to eliminate the need to modify multiple files when adding new blocks or providers.

## Current System Analysis

### Pain Points
1. **Multiple File Modifications**: Adding a new provider requires changes to:
   - `credentials_store.py` for default credentials
   - `oauth/__init__.py` for OAuth handlers
   - `webhooks/__init__.py` for webhook managers
   - `settings.py` for API keys

2. **Hardcoded Lists**: Default credentials and registrations are maintained as hardcoded objects

3. **No Auto-Registration**: OAuth handlers and webhook managers need manual registration

4. **Scattered Configuration**: Provider-related config is spread across multiple modules

## Proposed Solution

### Core Components

#### 1. **Auto-Registry System** (`backend/sdk/registry.py`)
Central registry that collects all provider configurations automatically when blocks are imported.

```python
class AutoRegistry:
    """Central registry for all block-related configurations"""
    
    _providers: Dict[str, Provider] = {}
    _default_credentials: List[Credentials] = []
    _oauth_handlers: Dict[str, Type[BaseOAuthHandler]] = {}
    _webhook_managers: Dict[str, Type[BaseWebhookManager]] = {}
    _block_configurations: Dict[Type[Block], BlockConfiguration] = {}
    
    @classmethod
    def register_provider(cls, provider: Provider):
        """Auto-register provider and all its configurations"""
        cls._providers[provider.name] = provider
        
        # Register OAuth handler if provided
        if provider.oauth_handler:
            cls._oauth_handlers[provider.name] = provider.oauth_handler
        
        # Register webhook manager if provided
        if provider.webhook_manager:
            cls._webhook_managers[provider.name] = provider.webhook_manager
        
        # Register default credentials
        cls._default_credentials.extend(provider.default_credentials)
    
    @classmethod
    def get_all_credentials(cls) -> List[Credentials]:
        """Replace hardcoded get_all_creds() in credentials_store.py"""
        return cls._default_credentials
    
    @classmethod
    def get_oauth_handlers(cls) -> Dict[str, Type[BaseOAuthHandler]]:
        """Replace HANDLERS_BY_NAME in oauth/__init__.py"""
        return cls._oauth_handlers
    
    @classmethod
    def get_webhook_managers(cls) -> Dict[str, Type[BaseWebhookManager]]:
        """Replace load_webhook_managers() in webhooks/__init__.py"""
        return cls._webhook_managers
```

#### 2. **ProviderBuilder Class** (`backend/sdk/builder.py`)
Fluent API for building provider configurations that auto-register on `.build()`.

```python
class ProviderBuilder:
    """Builder for creating provider configurations."""
    
    def __init__(self, name: str):
        self.name = name
        self._oauth_handler = None
        self._webhook_manager = None
        self._default_credentials = []
        self._base_costs = []
        self._supported_auth_types = set()
        self._api_client_factory = None
        self._error_handler = None
    
    def with_oauth(self, handler_class: Type[BaseOAuthHandler], scopes: List[str] = None) -> "ProviderBuilder":
        """Add OAuth support."""
        self._oauth_handler = handler_class
        self._supported_auth_types.add("oauth2")
        if scopes:
            self._default_scopes = scopes
        return self
    
    def with_api_key(self, default_key: str, title: str) -> "ProviderBuilder":
        """Add API key support with default credentials."""
        self._supported_auth_types.add("api_key")
        self._default_credentials.append(
            APIKeyCredentials(
                id=f"{self.name}-default",
                provider=self.name,
                api_key=SecretStr(default_key),
                title=title
            )
        )
        return self
    
    def with_webhook_manager(self, manager_class: Type[BaseWebhooksManager]) -> "ProviderBuilder":
        """Register webhook manager for this provider."""
        self._webhook_manager = manager_class
        return self
    
    def with_base_cost(self, amount: float, cost_type: BlockCostType) -> "ProviderBuilder":
        """Set base cost for all blocks using this provider."""
        self._base_costs.append(BlockCost(cost_amount=amount, cost_type=cost_type))
        return self
    
    def with_api_client(self, factory: Callable) -> "ProviderBuilder":
        """Register API client factory."""
        self._api_client_factory = factory
        return self
    
    def build(self) -> "Provider":
        """Build and register the provider configuration."""
        provider = Provider(
            name=self.name,
            oauth_handler=self._oauth_handler,
            webhook_manager=self._webhook_manager,
            default_credentials=self._default_credentials,
            base_costs=self._base_costs,
            supported_auth_types=self._supported_auth_types,
            api_client_factory=self._api_client_factory,
            error_handler=self._error_handler
        )
        
        # Auto-registration happens here
        AutoRegistry.register_provider(provider)
        return provider
```

#### 3. **Provider Class** (`backend/sdk/provider.py`)
Configuration container that holds all provider-related settings and provides utilities.

```python
class Provider:
    """A configured provider that blocks can use."""
    
    def __init__(self, **config):
        self.name = config['name']
        self.oauth_handler = config.get('oauth_handler')
        self.webhook_manager = config.get('webhook_manager')
        self.default_credentials = config.get('default_credentials', [])
        self.base_costs = config.get('base_costs', [])
        self.supported_auth_types = config.get('supported_auth_types', set())
        self._api_client_factory = config.get('api_client_factory')
        self._error_handler = config.get('error_handler')
    
    def credentials_field(self, **kwargs) -> CredentialsField:
        """Return a CredentialsField configured for this provider."""
        return CredentialsField(
            provider=self.name,
            supported_credential_types=self.supported_auth_types,
            description=kwargs.get("description", f"{self.name.title()} credentials"),
            required_scopes=kwargs.get("required_scopes"),
            **kwargs
        )
    
    def get_api(self, credentials) -> Any:
        """Get API client instance for the given credentials."""
        if self._api_client_factory:
            return self._api_client_factory(credentials)
        raise NotImplementedError(f"No API client factory registered for {self.name}")
    
    def handle_error(self, error: Exception) -> str:
        """Handle provider-specific errors."""
        if self._error_handler:
            return self._error_handler(error)
        return str(error)
```

#### 4. **Enhanced Block Base Class**
Update the Block base class to support provider configuration.

```python
class Block(ABC):
    """Enhanced base block with provider configuration support."""
    
    def __init__(
        self,
        *,
        id: str,
        description: str,
        categories: Set[BlockCategory],
        input_schema: Type[BlockSchema],
        output_schema: Type[BlockSchema],
        block_config: Provider = None,  # New optional parameter
        block_type: BlockType = BlockType.STANDARD,
        **kwargs
    ):
        # Existing initialization
        super().__init__(
            id=id,
            description=description,
            categories=categories,
            input_schema=input_schema,
            output_schema=output_schema,
            block_type=block_type,
            **kwargs
        )
        
        # Store provider configuration
        self.block_config = block_config
        
        # Register this block with its configuration if provided
        if block_config:
            self._register_block_configuration()
    
    def _register_block_configuration(self):
        """Register block configuration with auto-registry."""
        config = BlockConfiguration(
            provider=self.block_config.name,
            costs=self.block_config.base_costs,
            default_credentials=self.block_config.default_credentials,
            webhook_manager=self.block_config.webhook_manager,
            oauth_handler=self.block_config.oauth_handler
        )
        AutoRegistry.register_block_configuration(self.__class__, config)
```

#### 5. **Integration Patches** (`backend/sdk/__init__.py`)
Patches existing system to use AutoRegistry instead of hardcoded lists.

```python
def _patch_integrations():
    """Patch existing integration points to use AutoRegistry"""
    
    # Patch credentials_store.get_all_creds()
    import backend.integrations.credentials_store as creds_store
    creds_store.get_all_creds = lambda: AutoRegistry.get_all_credentials()
    
    # Patch oauth handlers
    import backend.integrations.oauth as oauth
    oauth.HANDLERS_BY_NAME = AutoRegistry.get_oauth_handlers()
    
    # Patch webhook managers
    import backend.integrations.webhooks as webhooks
    original_load = webhooks.load_webhook_managers
    
    def patched_load():
        # Get original managers
        managers = original_load()
        # Add SDK-registered managers
        managers.update(AutoRegistry.get_webhook_managers())
        return managers
    
    webhooks.load_webhook_managers = patched_load

# Call on import
_patch_integrations()
```

## Migration Example: Before and After

### Before (Current System - Multiple Files)

**Step 1: Add API key to settings.py**
```python
# backend/util/settings.py
class Secrets(BaseSettings):
    openweather_api_key: SecretStr = Field(default="", description="OpenWeather API key")
```

**Step 2: Add default credentials to credentials_store.py**
```python
# backend/integrations/credentials_store.py
def get_all_creds() -> list[Credentials]:
    # ... existing code ...
    
    if settings.secrets.openweather_api_key:
        all_credentials.append(
            APIKeyCredentials(
                id="openweather-default",
                provider="openweather",
                api_key=settings.secrets.openweather_api_key,
                title="Default OpenWeather credentials",
            )
        )
```

**Step 3: Create the block**
```python
# backend/blocks/openweather.py
class OpenWeatherBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="openweather",
            supported_credential_types={"api_key"}
        )
        location: str = SchemaField(description="City name")
```

### After (New SDK System - Single File)

**All configuration in one file:**
```python
# backend/blocks/openweather.py
from backend.sdk import *

# Configure provider with all settings
openweather = (
    ProviderBuilder("openweather")
    .with_api_key("OPENWEATHER_API_KEY", "OpenWeather API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

class OpenWeatherBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = openweather.credentials_field()
        location: str = SchemaField(description="City name")
    
    def __init__(self):
        super().__init__(
            id="openweather-...",
            description="Get weather data",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            block_config=openweather  # Provider config attached
        )
```

**Benefits:**
- No need to modify settings.py
- No need to update credentials_store.py
- All configuration in the block file
- Auto-registration on import

## Usage Examples

### Single Block Provider

```python
# blocks/weather/weather.py
from backend.sdk import *

weather = (
    ProviderBuilder("openweather")
    .with_api_key("test-weather-key", "OpenWeather API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

class WeatherBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = weather.credentials_field()
        location: String = SchemaField(description="City name or coordinates")
    
    class Output(BlockSchema):
        temperature: Float = SchemaField(description="Temperature in Celsius")
        conditions: String = SchemaField(description="Weather conditions")
        error: String = SchemaField(default="")
    
    def __init__(self):
        super().__init__(
            id="weather-current-12345678-1234-1234-1234-123456789012",
            description="Get current weather",
            categories={BlockCategory.DATA},
            input_schema=WeatherBlock.Input,
            output_schema=WeatherBlock.Output,
            block_config=weather
        )
    
    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        # ... implementation
```

### Multi-Block Provider with OAuth and Webhooks

```python
# blocks/github/_config.py
from backend.sdk import *

github = (
    ProviderBuilder("github")
    .with_oauth(GithubOAuthHandler, scopes=["repo", "user", "workflow"])
    .with_api_key("test-github-key", "GitHub PAT")
    .with_webhook_manager(GithubWebhooksManager)
    .with_base_cost(1, BlockCostType.RUN)
    .with_api_client(lambda creds: Github(auth=creds.get_auth()))
    .with_error_handler(handle_github_errors)
    .build()
)

# blocks/github/issues.py
from ._config import github

class GithubGetIssueBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = github.credentials_field()
        repo_url: String = SchemaField(description="Repository URL")
        issue_number: Integer = SchemaField(description="Issue number")
    
    def __init__(self):
        super().__init__(
            id="github-get-issue-...",
            description="Get GitHub issue",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            block_config=github
        )

# blocks/github/webhooks.py
from ._config import github

class GithubPushTriggerBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = github.credentials_field()
        repo: String = SchemaField(description="GitHub repository")
        payload: Dict = SchemaField(hidden=True)  # Webhook payload
        event_filter: Dict = SchemaField(
            description="Filter specific events",
            default={"branches": ["main"]}
        )
    
    def __init__(self):
        super().__init__(
            id="github-push-trigger-...",
            description="Triggered by GitHub push events",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            block_config=github,
            webhook_config=BlockWebhookConfig(
                provider="github",
                webhook_type=WebhookType.REPO,
                resource_format="/repos/{repo}",
                event_filter_input="event_filter",
                event_format="push"
            )
        )
```

## Implementation Steps

1. **Create SDK infrastructure files:**
   - `backend/sdk/registry.py` - Auto-registry system
   - `backend/sdk/builder.py` - ProviderBuilder class
   - `backend/sdk/provider.py` - Provider class

2. **Update `backend/sdk/__init__.py`:**
   - Import new classes
   - Add patching logic
   - Ensure all necessary exports

3. **Test with example block:**
   - Create a simple block using new pattern
   - Verify auto-registration works
   - Test credential flow

4. **Migrate existing blocks gradually:**
   - Start with single-block providers
   - Move to multi-block providers
   - Keep backwards compatibility

## Benefits

1. **Single File Configuration**: All provider config in one place
2. **Auto-Registration**: No manual registration needed
3. **Type Safety**: Builder pattern ensures correct configuration
4. **Reusability**: Multi-block providers share configuration
5. **SDK-Friendly**: Easy for external developers to use
6. **No External Changes**: Works with existing system via patching
7. **Self-Contained Blocks**: Each block file contains all its configuration

## API Key Registration Solution

### Current System
- API keys are defined in `Secrets` class in `settings.py`
- Each provider needs manual addition to `get_all_creds()` in `credentials_store.py`
- Settings uses `extra="allow"` which permits additional fields

### Dynamic API Key Registration

#### 1. **Leverage Pydantic's `extra="allow"`**
Since the `Secrets` class already allows extra fields, we can dynamically add API keys:

```python
# backend/sdk/registry.py
class AutoRegistry:
    _api_key_mappings: Dict[str, str] = {}  # provider -> env_var_name
    
    @classmethod
    def register_api_key(cls, provider: str, env_var_name: str):
        """Register an environment variable as an API key for a provider."""
        cls._api_key_mappings[provider] = env_var_name
        
        # Dynamically check if the env var exists and create credential
        import os
        api_key = os.getenv(env_var_name)
        if api_key:
            credential = APIKeyCredentials(
                id=f"{provider}-default",
                provider=provider,
                api_key=SecretStr(api_key),
                title=f"Default {provider} credentials"
            )
            cls._default_credentials.append(credential)
```

#### 2. **Update ProviderBuilder**
```python
class ProviderBuilder:
    def with_api_key(self, env_var_name: str, title: str) -> "ProviderBuilder":
        """Add API key support with environment variable name."""
        self._supported_auth_types.add("api_key")
        self._env_var_name = env_var_name
        
        # Register the API key mapping
        AutoRegistry.register_api_key(self.name, env_var_name)
        
        # Check if API key exists in environment
        import os
        api_key = os.getenv(env_var_name)
        if api_key:
            self._default_credentials.append(
                APIKeyCredentials(
                    id=f"{self.name}-default",
                    provider=self.name,
                    api_key=SecretStr(api_key),
                    title=title
                )
            )
        return self
```

#### 3. **Usage Example**
```python
# blocks/openweather/weather.py
weather = (
    ProviderBuilder("openweather")
    .with_api_key("OPENWEATHER_API_KEY", "OpenWeather API Key")  # Env var name
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

This allows blocks to register their API keys without modifying `settings.py`. The API key is loaded from the environment variable if it exists.

### Alternative: Direct Settings Access

For providers that need direct access to their API key from settings:

```python
class ProviderBuilder:
    def with_api_key_from_settings(self, settings_attr: str, title: str) -> "ProviderBuilder":
        """Use existing API key from settings."""
        from backend.util.settings import settings
        
        # Try to get the API key from settings
        api_key = getattr(settings.secrets, settings_attr, None)
        if api_key:
            self._default_credentials.append(
                APIKeyCredentials(
                    id=f"{self.name}-default",
                    provider=self.name,
                    api_key=api_key,
                    title=title
                )
            )
        return self
```

## Additional Considerations

### Thread Safety
The AutoRegistry uses class-level dictionaries that could have race conditions during concurrent imports. Consider using threading locks:

```python
import threading

class AutoRegistry:
    _lock = threading.Lock()
    
    @classmethod
    def register_provider(cls, provider: Provider):
        with cls._lock:
            # ... registration logic
```

### Import Order Dependencies
Since blocks auto-register on import, we need to ensure the SDK is fully initialized before any blocks are imported. The `_patch_integrations()` function should run early in the import chain.

### Backwards Compatibility
To maintain compatibility during migration:

1. Keep the existing manual registration methods working
2. Make `block_config` parameter optional in Block.__init__
3. Provide migration utilities to help convert existing blocks

### Error Handling
The SDK should provide clear error messages when:
- A provider is not found
- API keys are missing
- OAuth configuration is incomplete
- Webhook registration fails

## Implementation Priority

1. **Phase 1: Core Infrastructure**
   - Create AutoRegistry class
   - Create ProviderBuilder and Provider classes
   - Implement integration patches

2. **Phase 2: Block Integration**
   - Update Block base class
   - Create example blocks using new pattern
   - Test auto-registration flow

3. **Phase 3: Migration**
   - Create migration guide
   - Convert 2-3 existing blocks as examples
   - Update documentation

4. **Phase 4: Enhancement**
   - Add thread safety
   - Improve error handling
   - Add comprehensive tests

## Open Issues

1. **Testing**: Need comprehensive tests for the SDK components
2. **Documentation**: Need developer documentation for the SDK
3. **Migration Guide**: Need guide for migrating existing blocks
4. **Missing SDK Exports**: Need to add `Settings` and implement stub decorators
5. **Performance**: Monitor import time impact of auto-registration
6. **Type Hints**: Ensure all SDK components have proper type annotations