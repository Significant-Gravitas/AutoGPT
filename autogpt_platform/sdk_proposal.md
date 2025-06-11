# SDK Proposal: Builder Pattern with block_config

## Executive Summary

This proposal outlines a transition from the current decorator-based SDK approach to using a builder pattern with `block_config` passed to Block's `__init__`. This approach improves static code analysis with pyright while maintaining the simplicity and self-contained nature of block definitions.

## Current State Analysis

### Strengths of Current Decorator Approach
- Clean, declarative syntax
- Zero manual configuration required
- Self-contained block definitions
- Automatic registration at import time

### Challenges with Decorators
1. **Static Analysis Limitations**: Decorators can obscure type information from static analyzers
2. **Runtime Registration**: Configuration happens at import time, making it harder to trace
3. **Limited IDE Support**: Decorators may not provide optimal autocomplete/intellisense
4. **Testing Complexity**: Harder to test configuration in isolation

## Proposed Solution: Builder Pattern with block_config

### Core Concept

Every block uses a consistent pattern where configuration is created via a builder and passed to the block's `__init__` method:

```python
from backend.sdk import *

# Step 1: Create provider configuration using builder
github = (
    ProviderBuilder("github")
    .with_oauth(GithubOAuthHandler, scopes=["repo", "user"])
    .with_api_key("test-github-key", "GitHub PAT")
    .with_base_cost(1, BlockCostType.RUN)
    .build()  # This handles auto-registration
)

# Step 2: Create block using the configuration
class GithubGetIssueBlock(Block):
    class Input(BlockSchema):
        # Credentials field automatically configured from provider
        credentials: CredentialsMetaInput = github.credentials_field()
        repo_url: String = SchemaField(description="Repository URL")
        issue_number: Integer = SchemaField(description="Issue number")
    
    class Output(BlockSchema):
        issue: Dict = SchemaField(description="Issue data")
        error: String = SchemaField(default="")
    
    def __init__(self):
        super().__init__(
            id="github-get-issue-12345678-1234-1234-1234-123456789012",
            description="Get a GitHub issue",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetIssueBlock.Input,
            output_schema=GithubGetIssueBlock.Output,
            block_config=github  # Pass configuration here
        )
    
    def run(self, input_data: Input, *, credentials: Union[APIKeyCredentials, OAuth2Credentials], **kwargs) -> BlockOutput:
        try:
            # Use provider utilities
            api = self.block_config.get_api(credentials)
            result = api.get_issue(input_data.repo_url, input_data.issue_number)
            yield "issue", result
        except Exception as e:
            yield "error", str(e)
```

### Single Block Example

Even single-block providers use the same pattern for consistency:

```python
# blocks/weather/weather.py
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

### Multi-Block Provider Benefits

For providers with multiple blocks (e.g., GitHub with 10+ blocks), the pattern eliminates repetition:

```python
# blocks/github/_config.py (optional - for organization)
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

class GithubCreateIssueBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = github.credentials_field()
        repo_url: String = SchemaField(description="Repository URL")
        title: String = SchemaField(description="Issue title")
        body: String = SchemaField(description="Issue body")
    
    def __init__(self):
        super().__init__(
            id="github-create-issue-...",
            description="Create GitHub issue",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            block_config=github
        )

# ... 10+ more blocks, all using the same github config
```

## Implementation Details

### 1. ProviderBuilder Class

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

### 2. Provider Class

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

### 3. Enhanced Block Base Class

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
        block_config: Provider,  # Required parameter
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
        
        # Register this block with its configuration
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

## Benefits of This Approach

### 1. Improved Static Analysis
- `block_config` is a required parameter in `__init__`
- Pyright can verify all blocks have proper configuration
- No runtime magic or decorator introspection needed

### 2. Provider String Consistency
- Provider string defined once in ProviderBuilder
- `credentials_field()` method ensures matching provider strings
- Impossible to have mismatched providers between configuration and credentials

### 3. Better IDE Support
- Standard Python patterns that IDEs understand
- Autocomplete works for `github.credentials_field()`
- Go-to-definition works for all configuration

### 4. Consistency
- Every block uses the same pattern
- No special cases or alternative approaches
- Easy to understand and teach

### 5. DRY for Multi-Block Providers
- GitHub's 10+ blocks share one configuration
- Twitter's 20+ blocks share one configuration
- Changes to provider config apply to all blocks

### 6. Self-Contained
- Everything stays in the blocks folder
- No external configuration files
- Provider config can be in same file or shared module

## Complex Examples

### Webhook Block

```python
# blocks/github/webhooks.py
github = (
    ProviderBuilder("github")
    .with_webhook_manager(GithubWebhooksManager)
    .with_base_cost(0, BlockCostType.RUN)  # Webhooks are free
    .build()
)

class GithubPullRequestWebhook(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = github.credentials_field()
        events: BaseModel = SchemaField(
            description="Events to listen for",
            default={"opened": True, "closed": True, "synchronize": True}
        )
        payload: Dict = SchemaField(
            description="Webhook payload",
            default={},
            hidden=True
        )
    
    class Output(BlockSchema):
        action: String = SchemaField(description="PR action")
        pull_request: Dict = SchemaField(description="PR data")
    
    def __init__(self):
        super().__init__(
            id="github-pr-webhook-12345678-1234-1234-1234-123456789012",
            description="Triggered on pull request events",
            categories={BlockCategory.INPUT},
            input_schema=GithubPullRequestWebhook.Input,
            output_schema=GithubPullRequestWebhook.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider="github",
                webhook_type="pull_request",
                event_filter_input="events",
            ),
            block_config=github
        )
    
    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "action", payload.get("action", "unknown")
        yield "pull_request", payload.get("pull_request", {})
```

### Block with Custom Costs

```python
class GithubActionsBlock(Block):
    def __init__(self):
        # Can override base costs for specific blocks
        custom_github = (
            ProviderBuilder("github")
            .with_oauth(GithubOAuthHandler, scopes=["repo", "workflow"])
            .with_base_cost(5, BlockCostType.RUN)  # More expensive
            .build()
        )
        
        super().__init__(
            id="github-trigger-action-...",
            description="Trigger GitHub Action workflow",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            block_config=custom_github  # Uses custom config
        )
```

## Static Analysis Validation

Since `block_config` is a required parameter in Block's `__init__`, static analysis tools can easily verify proper usage:

```python
# Pyright will error on this:
class BadBlock(Block):
    def __init__(self):
        super().__init__(
            id="bad-block",
            description="Missing block_config",
            categories={BlockCategory.TEXT},
            input_schema=self.Input,
            output_schema=self.Output,
            # ERROR: Missing required parameter 'block_config'
        )
```

## Implementation Plan: Removing Decorators and Adding Builder Pattern

### Phase 1: Add Builder Support (Parallel with Decorators)

#### 1.1 Create New Builder Classes
```python
# backend/sdk/builder.py
class ProviderBuilder:
    """Builder for creating provider configurations."""
    # Implementation as shown above
    
class Provider:
    """A configured provider that blocks can use."""
    # Implementation as shown above
```

#### 1.2 Update Block Base Class
```python
# backend/data/block.py
class Block(ABC):
    def __init__(
        self,
        *,
        block_config: Optional[Provider] = None,  # New optional parameter
        **existing_params
    ):
        # Existing initialization
        super().__init__(**existing_params)
        
        # Handle both old and new approaches
        if block_config:
            self.block_config = block_config
            self._register_block_configuration()
        else:
            # Look for decorator-based config (backward compatibility)
            self._check_decorator_config()
```

#### 1.3 Update AutoRegistry
- Add methods to handle builder-based registration
- Maintain existing decorator support
- Update `setup_auto_registration()` to handle both approaches

### Phase 2: Migration Implementation

#### 2.1 Update SDK __init__.py
```python
# backend/sdk/__init__.py
# Add new imports
from .builder import ProviderBuilder, Provider

# Keep existing imports for backward compatibility
from .decorators import *  # Mark as deprecated

# Add deprecation warning for decorator usage
import warnings

def _deprecated_decorator_warning(name: str):
    warnings.warn(
        f"@{name} decorator is deprecated. Use ProviderBuilder instead.",
        DeprecationWarning,
        stacklevel=3
    )
```

#### 2.2 Update Integration Points

**Server Integration (rest_api.py)**:
```python
# No changes needed - setup_auto_registration() works with both approaches
```

**Provider Discovery (integrations/router.py)**:
```python
# Update to include builder-registered providers
async def get_available_providers() -> list[ProviderInfo]:
    registry = get_registry()
    providers = set()
    
    # Include both decorator and builder registered providers
    providers.update(registry.providers)
    providers.update(registry.builder_providers)  # New registry field
```

**Executor Integration (executor/manager.py)**:
```python
# Update credential resolution to check block_config first
if hasattr(node.block, 'block_config') and node.block.block_config:
    # Use builder-based default credentials
    default_creds = node.block.block_config.default_credentials
else:
    # Fall back to decorator-based approach
    default_creds = registry.get_default_credentials_for_block(node.block)
```

### Phase 3: Migrate Existing Blocks

#### 3.1 Single-Block Migration Example
```python
# Before (decorators)
@provider("exa")
@cost_config(BlockCost(cost_amount=10, cost_type=BlockCostType.RUN))
@default_credentials(
    APIKeyCredentials(
        id="exa-default",
        provider="exa",
        api_key=SecretStr("test-exa-key"),
        title="Exa Default API Key"
    )
)
class ExaSearchBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="exa",
            supported_credential_types={"api_key"}
        )

# After (builder)
exa = (
    ProviderBuilder("exa")
    .with_api_key("test-exa-key", "Exa Default API Key")
    .with_base_cost(10, BlockCostType.RUN)
    .build()
)

class ExaSearchBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field()
    
    def __init__(self):
        super().__init__(
            # ... other params ...
            block_config=exa
        )
```

#### 3.2 Migration Order
1. **Example blocks** (3 blocks) - Test migration process
2. **Exa blocks** (5 blocks) - Single provider with multiple blocks
3. **Future blocks** - Document builder pattern as standard

### Phase 4: Update Testing

#### 4.1 Add Builder Pattern Tests
```python
# backend/test/sdk/test_builder_pattern.py
def test_provider_builder():
    """Test ProviderBuilder creates correct configuration."""
    provider = (
        ProviderBuilder("test")
        .with_api_key("key", "Test Key")
        .with_base_cost(1, BlockCostType.RUN)
        .build()
    )
    
    assert provider.name == "test"
    assert len(provider.default_credentials) == 1
    assert len(provider.base_costs) == 1

def test_block_with_builder_config():
    """Test block properly uses builder configuration."""
    # Test that block_config is required by pyright
    # Test that credentials_field() works correctly
```

#### 4.2 Update Existing Tests
- Add parallel tests for builder pattern
- Ensure backward compatibility tests pass
- Add deprecation warning tests

### Phase 5: Frontend Updates

No frontend changes needed - the API remains the same:
- Provider discovery endpoints work with both approaches
- Dynamic provider validation continues to work
- UI components don't need updates

### Phase 6: Documentation Updates

#### 6.1 Update CLAUDE.md
- Show builder pattern as primary approach
- Move decorator examples to "Legacy" section
- Add migration guide for existing blocks

#### 6.2 Update SDK Documentation
- New "Quick Start" with builder pattern
- Migration guide from decorators
- Best practices for multi-block providers

### Phase 7: Cleanup (After Migration Period)

#### 7.1 Remove Decorator Support
```python
# backend/sdk/__init__.py
# Remove decorator imports
# from .decorators import *  # REMOVED

# backend/sdk/decorators.py
# Delete this file entirely
```

#### 7.2 Make block_config Required
```python
# backend/data/block.py
class Block(ABC):
    def __init__(
        self,
        *,
        block_config: Provider,  # Now required
        **existing_params
    ):
        # Remove backward compatibility code
```

#### 7.3 Clean Up AutoRegistry
- Remove decorator-specific methods
- Simplify to only support builder pattern
- Remove patching for decorator-based config

### Integration with Existing Systems

The builder pattern integrates seamlessly with existing systems:

1. **AutoRegistry**: Same registry, different registration method
2. **patch_existing_systems()**: Works identically
3. **Provider Discovery**: Enhanced to include builder providers
4. **Credential System**: Same patching mechanism
5. **Cost System**: Same integration point
6. **OAuth/Webhooks**: Same registration and patching

### Benefits of This Implementation

1. **Gradual Migration**: Both patterns work simultaneously
2. **No Breaking Changes**: Existing blocks continue to work
3. **Clear Deprecation Path**: Warnings guide developers
4. **Improved Type Safety**: Pyright can validate block_config
5. **Better Developer Experience**: IDE support and clearer patterns

## Testing Strategy

```python
def test_block_has_config():
    """Test that all blocks pass block_config."""
    # This will be caught at import time by pyright
    # but we can also test at runtime
    
    weather = ProviderBuilder("test").build()
    
    # Should raise TypeError without block_config
    with pytest.raises(TypeError, match="block_config"):
        class TestBlock(Block):
            def __init__(self):
                super().__init__(
                    id="test",
                    description="test",
                    categories={BlockCategory.TEXT},
                    input_schema=BlockSchema,
                    output_schema=BlockSchema,
                    # Missing block_config
                )

def test_credentials_field_consistency():
    """Test that credentials field uses correct provider."""
    github = ProviderBuilder("github").with_api_key("key", "GitHub").build()
    
    field = github.credentials_field()
    assert field.provider == "github"
    assert field.supported_credential_types == {"api_key"}
```

## How Builder Pattern Integrates with Current System

### Auto-Registration Flow Comparison

**Current Decorator Flow**:
1. Block file imported → Decorators execute → Register with AutoRegistry
2. Server startup → `setup_auto_registration()` → `patch_existing_systems()`
3. Systems patched → SDK config available everywhere

**New Builder Flow**:
1. Block file imported → `ProviderBuilder.build()` → Register with AutoRegistry
2. Block instantiated → `block_config` passed → Additional registration
3. Same startup and patching process

### Key Integration Points

1. **AutoRegistry (`backend/sdk/auto_registry.py`)**:
   - Continues to be the central storage
   - Enhanced to track builder-created providers separately
   - Same patching mechanism works for both approaches

2. **Server Startup (`backend/server/rest_api.py`)**:
   - No changes needed to `setup_auto_registration()`
   - Both decorator and builder registrations are discovered

3. **Provider Discovery (`backend/server/integrations/router.py`)**:
   - Endpoints already support dynamic providers
   - Will automatically include builder-registered providers

4. **Executor (`backend/executor/manager.py`)**:
   - Already uses registry for default credentials
   - Enhanced to check `block.block_config` first

5. **Frontend (`frontend/src/hooks/useProviders.ts`)**:
   - No changes needed
   - API contract remains the same

### System Patches Applied

The `patch_existing_systems()` function updates these systems:

1. **Block Costs** (`backend/data/block_cost_config.py`):
   ```python
   BLOCK_COSTS.update(registry.get_block_costs_dict())
   ```

2. **Credentials Store** (`backend/integrations/credentials_store.py`):
   - Patches `IntegrationCredentialsStore.get_all_creds()`
   - Adds SDK credentials to every user

3. **OAuth Handlers** (`backend/integrations/oauth/__init__.py`):
   ```python
   HANDLERS_BY_NAME[provider_enum] = handler
   ```

4. **Webhook Managers** (`backend/integrations/webhooks/__init__.py`):
   ```python
   _WEBHOOK_MANAGERS[provider_enum] = manager
   ```

### Backward Compatibility

The implementation ensures:
- Existing decorator-based blocks continue working
- Both patterns can coexist during migration
- No changes to API contracts
- Gradual migration path with deprecation warnings

## Conclusion

The builder pattern with `block_config` provides:

1. **100% Consistency**: Every block uses the same pattern
2. **Static Analysis Support**: Required parameter catches missing configuration
3. **DRY Code**: Multi-block providers share configuration  
4. **Type Safety**: Provider strings guaranteed to match
5. **Self-Contained**: Everything stays in blocks folder
6. **Clear and Explicit**: No magic, just standard Python
7. **Seamless Integration**: Works with all existing systems

This approach maintains the SDK's core benefit (self-contained blocks) while significantly improving developer experience through better static analysis and reduced repetition. The implementation plan ensures a smooth transition without breaking existing functionality.