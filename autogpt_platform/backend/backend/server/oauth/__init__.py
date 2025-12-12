"""
OAuth 2.0 Provider module for AutoGPT Platform.

This module implements AutoGPT as an OAuth 2.0 Authorization Server,
allowing external applications to authenticate users and access
platform resources with user consent.

Key components:
- router.py: OAuth authorization and token endpoints
- discovery_router.py: OIDC discovery endpoints
- client_router.py: OAuth client management
- token_service.py: JWT generation and validation
- service.py: Core OAuth business logic
"""

from backend.server.oauth.client_router import client_router
from backend.server.oauth.discovery_router import discovery_router
from backend.server.oauth.router import oauth_router

__all__ = ["oauth_router", "discovery_router", "client_router"]
