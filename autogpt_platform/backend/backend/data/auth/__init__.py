"""
Native authentication module for AutoGPT Platform.

This module provides authentication functionality that replaces Supabase Auth,
including:
- Password hashing with Argon2id
- JWT token generation and validation
- Magic links for email verification and password reset
- Email service for auth-related emails
- User migration from Supabase

Usage:
    from backend.data.auth.password import hash_password, verify_password
    from backend.data.auth.tokens import create_access_token, create_token_pair
    from backend.data.auth.magic_links import create_password_reset_link
    from backend.data.auth.email_service import get_auth_email_service
"""

from backend.data.auth.email_service import AuthEmailService, get_auth_email_service
from backend.data.auth.magic_links import (
    MagicLinkPurpose,
    create_email_verification_link,
    create_password_reset_link,
    verify_email_token,
    verify_password_reset_token,
)
from backend.data.auth.password import hash_password, needs_rehash, verify_password
from backend.data.auth.tokens import (
    TokenPair,
    create_access_token,
    create_token_pair,
    decode_access_token,
    revoke_all_user_refresh_tokens,
    validate_refresh_token,
)

__all__ = [
    # Password
    "hash_password",
    "verify_password",
    "needs_rehash",
    # Tokens
    "TokenPair",
    "create_access_token",
    "create_token_pair",
    "decode_access_token",
    "validate_refresh_token",
    "revoke_all_user_refresh_tokens",
    # Magic Links
    "MagicLinkPurpose",
    "create_email_verification_link",
    "create_password_reset_link",
    "verify_email_token",
    "verify_password_reset_token",
    # Email Service
    "AuthEmailService",
    "get_auth_email_service",
]
