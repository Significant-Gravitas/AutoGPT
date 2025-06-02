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

# === WEBHOOKS ===
try:
    from backend.integrations.webhooks._base import BaseWebhooksManager
except ImportError:
    BaseWebhooksManager = None

try:
    from backend.integrations.webhooks._manual_base import ManualWebhookManagerBase
except ImportError:
    ManualWebhookManagerBase = None

# === COST SYSTEM ===
try:
    from backend.data.cost import BlockCost, BlockCostType
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

# === UTILITIES ===
from backend.util import json

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
    from logging import getLogger as TruncatedLogger

# === COMMON TYPES ===
from typing import Any, Dict, List, Literal, Optional, Union, TypeVar, Type, Tuple, Set
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
# GitHub components
try:
    from backend.blocks.github._auth import (
        GithubCredentials, GithubCredentialsInput, GithubCredentialsField
    )
except ImportError:
    GithubCredentials = None
    GithubCredentialsInput = None
    GithubCredentialsField = None

# Google components
try:
    from backend.blocks.google._auth import (
        GoogleCredentials, GoogleCredentialsInput, GoogleCredentialsField  
    )
except ImportError:
    GoogleCredentials = None
    GoogleCredentialsInput = None
    GoogleCredentialsField = None

# OAuth handlers
try:
    from backend.integrations.oauth.base import BaseOAuthHandler
except ImportError:
    BaseOAuthHandler = None

try:
    from backend.integrations.oauth.github import GitHubOAuthHandler
except ImportError:
    GitHubOAuthHandler = None

try:
    from backend.integrations.oauth.google import GoogleOAuthHandler
except ImportError:
    GoogleOAuthHandler = None

# Webhook managers
try:
    from backend.integrations.webhooks.github import GithubWebhooksManager
    GitHubWebhooksManager = GithubWebhooksManager  # Alias for consistency
except ImportError:
    GitHubWebhooksManager = None
    GithubWebhooksManager = None

try:
    from backend.integrations.webhooks.generic import GenericWebhooksManager
    GenericWebhookManager = GenericWebhooksManager  # Alias for consistency
except ImportError:
    GenericWebhookManager = None
    GenericWebhooksManager = None

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
    "GitHubWebhooksManager", "GithubWebhooksManager", "GenericWebhookManager", "GenericWebhooksManager",
    
    # Utilities
    "json", "store_media_file", "MediaFileType", "convert", "TextFormatter", 
    "TruncatedLogger", "logging", "asyncio",
    
    # Types
    "String", "Integer", "Float", "Boolean", "List", "Dict", "Optional", 
    "Any", "Literal", "Union", "TypeVar", "Type", "Tuple", "Set",
    "BaseModel", "SecretStr", "Field", "Enum",
    
    # Auto-Registration Decorators
    "register_credentials", "register_cost", "register_oauth", "register_webhook_manager",
    "provider", "cost_config", "webhook_config", "default_credentials",
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]