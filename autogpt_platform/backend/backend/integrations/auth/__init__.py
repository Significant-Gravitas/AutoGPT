"""
SAML Authentication Integration Module

This module provides SAML 2.0 authentication support for AutoGPT,
enabling enterprise single sign-on (SSO) with identity providers like:
- Okta
- Azure Active Directory
- ADFS
- Shibboleth
- and other SAML 2.0 compliant IdPs

Components:
- saml.py: Core SAML client and provider management
- saml_data.py: Database operations for SAML entities
- saml_test.py: Test suite for SAML functionality

Usage:
1. Configure SAML providers in the database
2. Use the API routes to initiate authentication
3. Process SAML responses to authenticate users
4. Manage user sessions and logout

Configuration:
The module reads configuration from:
- Environment variables (prefixed with SAML_)
- Database provider configurations
- Runtime provider registration

Security Features:
- Signed and encrypted assertions (configurable)
- Certificate-based trust
- Session management
- Request tracking for security auditing
"""

from .saml import (
    SAMLAuthManager,
    SAMLProviderConfig,
    SAMLUserAttributes,
    configure_saml_providers,
    get_saml_manager,
)

from .saml_data import (
    SAMLAuthService,
    SAMLAuthRequestData,
    SAMLProviderData,
    SAMLUserData,
)

__all__ = [
    "SAMLAuthManager",
    "SAMLProviderConfig",
    "SAMLUserAttributes",
    "configure_saml_providers",
    "get_saml_manager",
    "SAMLAuthService",
    "SAMLAuthRequestData",
    "SAMLProviderData",
    "SAMLUserData",
]
