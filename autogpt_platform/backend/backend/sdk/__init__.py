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

# Third-party imports
from pydantic import BaseModel, Field, SecretStr

# === CORE BLOCK SYSTEM ===
from backend.data.block import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchema,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
    BlockWebhookConfig,
)
from backend.data.integrations import Webhook, update_webhook
from backend.data.model import APIKeyCredentials, Credentials, CredentialsField
from backend.data.model import CredentialsMetaInput as _CredentialsMetaInput
from backend.data.model import (
    NodeExecutionStats,
    OAuth2Credentials,
    SchemaField,
    UserPasswordCredentials,
)

# === INTEGRATIONS ===
from backend.integrations.providers import ProviderName
from backend.sdk.builder import ProviderBuilder
from backend.sdk.cost_integration import cost
from backend.sdk.provider import Provider

# === NEW SDK COMPONENTS (imported early for patches) ===
from backend.sdk.registry import AutoRegistry, BlockConfiguration

# === UTILITIES ===
from backend.util import json
from backend.util.request import Requests

# === OPTIONAL IMPORTS WITH TRY/EXCEPT ===
# Webhooks
try:
    from backend.integrations.webhooks._base import BaseWebhooksManager
except ImportError:
    BaseWebhooksManager = None

try:
    from backend.integrations.webhooks._manual_base import ManualWebhookManagerBase
except ImportError:
    ManualWebhookManagerBase = None

# Cost System
try:
    from backend.data.block import BlockCost, BlockCostType
except ImportError:
    from backend.data.block_cost_config import BlockCost, BlockCostType

try:
    from backend.data.credit import UsageTransactionMetadata
except ImportError:
    UsageTransactionMetadata = None

try:
    from backend.executor.utils import block_usage_cost
except ImportError:
    block_usage_cost = None

# Utilities
try:
    from backend.util.file import store_media_file
except ImportError:
    store_media_file = None

try:
    from backend.util.type import MediaFileType, convert
except ImportError:
    MediaFileType = None
    convert = None

try:
    from backend.util.text import TextFormatter
except ImportError:
    TextFormatter = None

try:
    from backend.util.logging import TruncatedLogger
except ImportError:
    TruncatedLogger = None


# OAuth handlers
try:
    from backend.integrations.oauth.base import BaseOAuthHandler
except ImportError:
    BaseOAuthHandler = None


# Credential type with proper provider name
from typing import Literal as _Literal

CredentialsMetaInput = _CredentialsMetaInput[
    ProviderName, _Literal["api_key", "oauth2", "user_password"]
]


# === COMPREHENSIVE __all__ EXPORT ===
__all__ = [
    # Core Block System
    "Block",
    "BlockCategory",
    "BlockOutput",
    "BlockSchema",
    "BlockSchemaInput",
    "BlockSchemaOutput",
    "BlockType",
    "BlockWebhookConfig",
    "BlockManualWebhookConfig",
    # Schema and Model Components
    "SchemaField",
    "Credentials",
    "CredentialsField",
    "CredentialsMetaInput",
    "APIKeyCredentials",
    "OAuth2Credentials",
    "UserPasswordCredentials",
    "NodeExecutionStats",
    # Cost System
    "BlockCost",
    "BlockCostType",
    "UsageTransactionMetadata",
    "block_usage_cost",
    # Integrations
    "ProviderName",
    "BaseWebhooksManager",
    "ManualWebhookManagerBase",
    "Webhook",
    "update_webhook",
    # Provider-Specific (when available)
    "BaseOAuthHandler",
    # Utilities
    "json",
    "store_media_file",
    "MediaFileType",
    "convert",
    "TextFormatter",
    "TruncatedLogger",
    "BaseModel",
    "Field",
    "SecretStr",
    "Requests",
    # SDK Components
    "AutoRegistry",
    "BlockConfiguration",
    "Provider",
    "ProviderBuilder",
    "cost",
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]
