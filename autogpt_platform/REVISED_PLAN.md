# AutoGPT Platform Block SDK - Complete Simplification Plan

## Executive Summary

This plan provides a **complete solution** to simplify block development by:

1. **Single Import**: `from backend.sdk import *` provides everything needed for block development
2. **Zero External Changes**: Adding new blocks requires **no code changes outside the blocks folder**
3. **Auto-Registration**: Credentials, costs, OAuth, and webhooks are automatically discovered and registered

## Current Problem Analysis

### Manual Registration Locations (Must Be Eliminated)

Currently, adding a new block requires manual updates in **5+ files outside the blocks folder**:

1. **`backend/data/block_cost_config.py`**: Block cost configurations (lines 184-300)
2. **`backend/integrations/credentials_store.py`**: Default credentials (lines 188-210) 
3. **`backend/integrations/oauth/__init__.py`**: OAuth handler registration (lines 16-26)
4. **`backend/integrations/webhooks/__init__.py`**: Webhook manager registration (lines 20-30)
5. **`backend/integrations/providers.py`**: Provider name enum (lines 6-43)

### Import Complexity Problem

Current blocks require **8-15 import statements** from various backend modules:
```python
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.blocks.github._auth import GithubCredentials, GithubCredentialsField
from backend.data.cost import BlockCost, BlockCostType
# ... and more
```

## Proposed Solution: Complete SDK with Auto-Registration

### 1. SDK Module Structure

```
backend/
├── sdk/
│   ├── __init__.py              # Complete re-export of all block dependencies
│   ├── auto_registry.py         # Auto-registration system for costs/credentials/webhooks
│   └── decorators.py            # Registration decorators for blocks
```

### 2. Complete Re-Export System (`backend/sdk/__init__.py`)

```python
"""
AutoGPT Platform Block Development SDK

Complete re-export of all dependencies needed for block development.
Usage: from backend.sdk import *

This module provides:
- All block base classes and types
- All credential and authentication components  
- All cost tracking components
- All webhook components
- All utility functions
- Auto-registration decorators
"""

# === CORE BLOCK SYSTEM ===
from backend.data.block import (
    Block, BlockCategory, BlockOutput, BlockSchema, BlockType,
    BlockWebhookConfig, BlockManualWebhookConfig
)
from backend.data.model import (
    SchemaField, CredentialsField, CredentialsMetaInput,
    APIKeyCredentials, OAuth2Credentials, UserPasswordCredentials,
    NodeExecutionStats
)

# === INTEGRATIONS ===
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks._base import BaseWebhooksManager, ManualWebhookManagerBase

# === COST SYSTEM ===
from backend.data.cost import BlockCost, BlockCostType
from backend.data.credit import UsageTransactionMetadata
from backend.executor.utils import block_usage_cost

# === UTILITIES ===
from backend.util import json
from backend.util.file import store_media_file
from backend.util.type import MediaFileType, convert
from backend.util.text import TextFormatter
from backend.util.logging import TruncatedLogger

# === COMMON TYPES ===
from typing import Any, Dict, List, Literal, Optional, Union, TypeVar, Type
from pydantic import BaseModel, SecretStr, Field
from enum import Enum
import logging
import asyncio

# === TYPE ALIASES ===
String = str
Integer = int  
Float = float
Boolean = bool

# === AUTO-REGISTRATION DECORATORS ===
from .decorators import (
    register_credentials, register_cost, register_oauth, register_webhook_manager,
    provider, cost_config, webhook_config, default_credentials
)

# === RE-EXPORT PROVIDER-SPECIFIC COMPONENTS ===
# Dynamically import and re-export provider-specific components
try:
    from backend.blocks.github._auth import (
        GithubCredentials, GithubCredentialsInput, GithubCredentialsField
    )
except ImportError:
    pass

try:
    from backend.blocks.google._auth import (
        GoogleCredentials, GoogleCredentialsInput, GoogleCredentialsField  
    )
except ImportError:
    pass

try:
    from backend.integrations.oauth.github import GitHubOAuthHandler
    from backend.integrations.oauth.google import GoogleOAuthHandler
    from backend.integrations.oauth.base import BaseOAuthHandler
except ImportError:
    pass

try:
    from backend.integrations.webhooks.github import GitHubWebhooksManager
    from backend.integrations.webhooks.generic import GenericWebhookManager
except ImportError:
    pass

# === COMPREHENSIVE __all__ EXPORT ===
__all__ = [
    # Core Block System
    "Block", "BlockCategory", "BlockOutput", "BlockSchema", "BlockType",
    "BlockWebhookConfig", "BlockManualWebhookConfig",
    
    # Schema and Model Components
    "SchemaField", "CredentialsField", "CredentialsMetaInput", 
    "APIKeyCredentials", "OAuth2Credentials", "UserPasswordCredentials",
    "NodeExecutionStats",
    
    # Cost System
    "BlockCost", "BlockCostType", "UsageTransactionMetadata", "block_usage_cost",
    
    # Integrations  
    "ProviderName", "BaseWebhooksManager", "ManualWebhookManagerBase",
    
    # Provider-Specific (when available)
    "GithubCredentials", "GithubCredentialsInput", "GithubCredentialsField",
    "GoogleCredentials", "GoogleCredentialsInput", "GoogleCredentialsField",
    "BaseOAuthHandler", "GitHubOAuthHandler", "GoogleOAuthHandler", 
    "GitHubWebhooksManager", "GenericWebhookManager",
    
    # Utilities
    "json", "store_media_file", "MediaFileType", "convert", "TextFormatter", 
    "TruncatedLogger", "logging", "asyncio",
    
    # Types
    "String", "Integer", "Float", "Boolean", "List", "Dict", "Optional", 
    "Any", "Literal", "Union", "TypeVar", "Type", "BaseModel", "SecretStr", 
    "Field", "Enum",
    
    # Auto-Registration Decorators
    "register_credentials", "register_cost", "register_oauth", "register_webhook_manager",
    "provider", "cost_config", "webhook_config", "default_credentials",
]
```

### 3. Auto-Registration System (`backend/sdk/auto_registry.py`)

```python
"""
Auto-Registration System for AutoGPT Platform

Automatically discovers and registers:
- Block costs
- Default credentials  
- OAuth handlers
- Webhook managers
- Provider names

This eliminates the need to manually update configuration files
outside the blocks folder when adding new blocks.
"""

from typing import Dict, List, Set, Type, Any
import inspect
from dataclasses import dataclass, field

# === GLOBAL REGISTRIES ===
class AutoRegistry:
    """Central registry for auto-discovered block configurations."""
    
    def __init__(self):
        self.block_costs: Dict[Type, List] = {}
        self.default_credentials: List[Any] = []
        self.oauth_handlers: Dict[str, Type] = {}
        self.webhook_managers: Dict[str, Type] = {}
        self.providers: Set[str] = set()
    
    def register_block_cost(self, block_class: Type, cost_config: List):
        """Register cost configuration for a block."""
        self.block_costs[block_class] = cost_config
    
    def register_default_credential(self, credential):
        """Register a default platform credential."""
        self.default_credentials.append(credential)
    
    def register_oauth_handler(self, provider_name: str, handler_class: Type):
        """Register an OAuth handler for a provider."""
        self.oauth_handlers[provider_name] = handler_class
    
    def register_webhook_manager(self, provider_name: str, manager_class: Type):
        """Register a webhook manager for a provider."""
        self.webhook_managers[provider_name] = manager_class
    
    def register_provider(self, provider_name: str):
        """Register a new provider name."""
        self.providers.add(provider_name)
    
    def get_block_costs_dict(self) -> Dict[Type, List]:
        """Get block costs in format expected by current system."""
        return self.block_costs.copy()
    
    def get_default_credentials_list(self) -> List[Any]:
        """Get default credentials in format expected by current system."""
        return self.default_credentials.copy()
    
    def get_oauth_handlers_dict(self) -> Dict[str, Type]:
        """Get OAuth handlers in format expected by current system."""
        return self.oauth_handlers.copy()
    
    def get_webhook_managers_dict(self) -> Dict[str, Type]:
        """Get webhook managers in format expected by current system."""
        return self.webhook_managers.copy()

# Global registry instance
_registry = AutoRegistry()

def get_registry() -> AutoRegistry:
    """Get the global auto-registry instance."""
    return _registry

# === DISCOVERY FUNCTIONS ===
def discover_block_configurations():
    """
    Discover all block configurations by scanning loaded blocks.
    Called during application startup after blocks are loaded.
    """
    from backend.blocks import load_all_blocks
    
    # Load all blocks (this also imports all block modules)
    load_all_blocks()
    
    # Registry is populated by decorators during import
    return _registry

def patch_existing_systems():
    """
    Patch existing configuration systems to use auto-discovered data.
    This maintains backward compatibility while enabling auto-registration.
    """
    
    # Patch block cost configuration
    import backend.data.block_cost_config as cost_config
    original_block_costs = getattr(cost_config, 'BLOCK_COSTS', {})
    cost_config.BLOCK_COSTS = {**original_block_costs, **_registry.get_block_costs_dict()}
    
    # Patch credentials store
    import backend.integrations.credentials_store as cred_store
    if hasattr(cred_store, 'DEFAULT_CREDENTIALS'):
        cred_store.DEFAULT_CREDENTIALS.extend(_registry.get_default_credentials_list())
    
    # Patch OAuth handlers
    import backend.integrations.oauth as oauth_module
    if hasattr(oauth_module, 'HANDLERS_BY_NAME'):
        oauth_module.HANDLERS_BY_NAME.update(_registry.get_oauth_handlers_dict())
    
    # Patch webhook managers  
    import backend.integrations.webhooks as webhook_module
    if hasattr(webhook_module, '_WEBHOOK_MANAGERS'):
        webhook_module._WEBHOOK_MANAGERS.update(_registry.get_webhook_managers_dict())
```

### 4. Registration Decorators (`backend/sdk/decorators.py`)

```python
"""
Registration Decorators for AutoGPT Platform Blocks

These decorators allow blocks to self-register their configurations:
- @cost_config: Register block cost configuration
- @default_credentials: Register default platform credentials
- @provider: Register new provider name
- @webhook_config: Register webhook manager
- @oauth_config: Register OAuth handler
"""

from typing import List, Type, Any, Optional
from functools import wraps
from .auto_registry import get_registry

def cost_config(*cost_configurations):
    """
    Decorator to register cost configuration for a block.
    
    Usage:
        @cost_config(
            BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
            BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE)
        )
        class MyBlock(Block):
            pass
    """
    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_block_cost(block_class, list(cost_configurations))
        return block_class
    return decorator

def default_credentials(*credentials):
    """
    Decorator to register default platform credentials.
    
    Usage:
        @default_credentials(
            APIKeyCredentials(provider="myservice", api_key="default-key")
        )
        class MyBlock(Block):
            pass
    """
    def decorator(block_class: Type):
        registry = get_registry()
        for credential in credentials:
            registry.register_default_credential(credential)
        return block_class
    return decorator

def provider(provider_name: str):
    """
    Decorator to register a new provider name.
    
    Usage:
        @provider("myservice")
        class MyBlock(Block):
            pass
    """
    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_provider(provider_name)
        return block_class
    return decorator

def webhook_config(provider_name: str, manager_class: Type):
    """
    Decorator to register a webhook manager.
    
    Usage:
        @webhook_config("github", GitHubWebhooksManager)
        class GitHubWebhookBlock(Block):
            pass
    """
    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_webhook_manager(provider_name, manager_class)
        return block_class
    return decorator

def oauth_config(provider_name: str, handler_class: Type):
    """
    Decorator to register an OAuth handler.
    
    Usage:
        @oauth_config("github", GitHubOAuthHandler)  
        class GitHubBlock(Block):
            pass
    """
    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_oauth_handler(provider_name, handler_class)
        return block_class
    return decorator

# === CONVENIENCE DECORATORS ===
def register_credentials(*credentials):
    """Alias for default_credentials decorator."""
    return default_credentials(*credentials)

def register_cost(*cost_configurations):
    """Alias for cost_config decorator.""" 
    return cost_config(*cost_configurations)

def register_oauth(provider_name: str, handler_class: Type):
    """Alias for oauth_config decorator."""
    return oauth_config(provider_name, handler_class)

def register_webhook_manager(provider_name: str, manager_class: Type):
    """Alias for webhook_config decorator."""
    return webhook_config(provider_name, manager_class)
```

### 5. Integration with Existing Systems

To maintain backward compatibility, we need to patch the existing configuration loading:

#### A. Patch Application Startup (`backend/app.py` or `backend/server/rest_api.py`)

```python
# Add this after block loading
from backend.sdk.auto_registry import discover_block_configurations, patch_existing_systems

# During application startup (after initialize_blocks())
def setup_auto_registration():
    """Set up auto-registration system."""
    # Discover all block configurations
    registry = discover_block_configurations()
    
    # Patch existing systems to use discovered configurations
    patch_existing_systems()
    
    print(f"Auto-registered {len(registry.block_costs)} block costs")
    print(f"Auto-registered {len(registry.default_credentials)} default credentials") 
    print(f"Auto-registered {len(registry.oauth_handlers)} OAuth handlers")
    print(f"Auto-registered {len(registry.webhook_managers)} webhook managers")
    print(f"Auto-registered {len(registry.providers)} providers")

# Call during lifespan startup
setup_auto_registration()
```

#### B. Extend Provider Enum Dynamically

```python
# backend/integrations/providers.py - Add dynamic extension
def extend_provider_enum():
    """Dynamically extend ProviderName enum with auto-discovered providers."""
    from backend.sdk.auto_registry import get_registry
    
    registry = get_registry()
    for provider_name in registry.providers:
        if not hasattr(ProviderName, provider_name.upper()):
            setattr(ProviderName, provider_name.upper(), provider_name)
```

### 6. Example Block with Auto-Registration

```python
# Example: backend/blocks/my_service.py
from backend.sdk import *

@provider("myservice")
@cost_config(
    BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
    BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE)
)
@default_credentials(
    APIKeyCredentials(
        id="myservice-default",
        provider="myservice", 
        api_key=SecretStr("default-key-from-env"),
        title="MyService Default API Key"
    )
)
class MyServiceBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="myservice",
            supported_credential_types={"api_key"}
        )
        text: String = SchemaField(description="Text to process")
        
    class Output(BlockSchema):
        result: String = SchemaField(description="Processing result")
        error: String = SchemaField(description="Error message if failed")
    
    def __init__(self):
        super().__init__(
            id="myservice-block-uuid",
            description="Process text using MyService API", 
            categories={BlockCategory.TEXT},
            input_schema=MyServiceBlock.Input,
            output_schema=MyServiceBlock.Output,
        )
    
    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            # Use MyService API with credentials
            api_key = credentials.api_key.get_secret_value()
            # ... implementation
            yield "result", f"Processed: {input_data.text}"
        except Exception as e:
            yield "error", str(e)
```

**Key Benefits of This Example:**
1. **Single Import**: `from backend.sdk import *` provides everything needed
2. **Self-Contained**: All configuration is in the block file via decorators
3. **No External Changes**: Adding this block requires zero changes outside the blocks folder
4. **Auto-Discovery**: Provider, costs, and credentials are automatically registered

### 7. Migration Strategy

#### Phase 1: Implement SDK and Auto-Registration (Week 1)
1. Create `backend/sdk/__init__.py` with complete re-exports
2. Create `backend/sdk/auto_registry.py` with registration system
3. Create `backend/sdk/decorators.py` with registration decorators
4. Patch application startup to use auto-registration
5. Test with existing blocks (should work unchanged)

#### Phase 2: Migrate Configuration (Week 2) 
1. Move cost configurations from `block_cost_config.py` to block decorators
2. Move default credentials from `credentials_store.py` to block decorators  
3. Move OAuth handlers from `oauth/__init__.py` to block decorators
4. Move webhook managers from `webhooks/__init__.py` to block decorators
5. Update 5-10 example blocks to demonstrate new patterns

#### Phase 3: Documentation and Adoption (Week 3)
1. Update developer documentation with new patterns
2. Create migration guide for existing blocks
3. Add VS Code snippets for new decorators
4. Create video tutorial showing complete workflow

#### Phase 4: Cleanup (Week 4)
1. Remove old configuration files (optional, for backward compatibility)
2. Migrate remaining blocks to new pattern
3. Simplify application startup code
4. Performance optimizations

### 8. Implementation Checklist

#### Core SDK Implementation
- [ ] Create `backend/sdk/__init__.py` with complete re-exports (~200 lines)
- [ ] Create `backend/sdk/auto_registry.py` with registry system (~150 lines)
- [ ] Create `backend/sdk/decorators.py` with decorators (~100 lines)
- [ ] Test that `from backend.sdk import *` provides all needed imports
- [ ] Test that existing blocks work unchanged

#### Auto-Registration Implementation  
- [ ] Patch application startup to call auto-registration
- [ ] Patch `block_cost_config.py` to use auto-discovered costs
- [ ] Patch `credentials_store.py` to use auto-discovered credentials
- [ ] Patch `oauth/__init__.py` to use auto-discovered handlers
- [ ] Patch `webhooks/__init__.py` to use auto-discovered managers
- [ ] Extend `ProviderName` enum dynamically

#### Testing and Migration
- [ ] Create test blocks using new decorators
- [ ] Migrate 5 existing blocks to demonstrate patterns
- [ ] Add comprehensive tests for auto-registration
- [ ] Performance test auto-discovery system
- [ ] Create migration guide and documentation

### 9. Benefits Summary

#### For Block Developers
- **Single Import**: `from backend.sdk import *` provides everything needed
- **Zero External Changes**: Adding blocks requires no modifications outside blocks folder
- **Self-Documenting**: All configuration is visible in the block file
- **Type Safety**: Full IDE support and type checking

#### For Platform Maintainers  
- **Eliminates Manual Updates**: No more updating 5+ configuration files
- **Reduces Errors**: No risk of forgetting to update configuration files
- **Easier Code Reviews**: All configuration changes are in the block PR
- **Better Modularity**: Blocks are truly self-contained

#### For the Platform
- **Faster Development**: New blocks can be added without cross-cutting changes
- **Better Scalability**: System handles 100s of blocks without complexity
- **Improved Documentation**: Configuration is co-located with implementation
- **Easier Testing**: Blocks can be tested in isolation

This comprehensive solution achieves both goals: **complete import simplification** via `from backend.sdk import *` and **zero external configuration** via auto-registration decorators.