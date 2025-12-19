"""
Core authentication service for password verification and token management.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from autogpt_libs.auth.config import get_settings
from autogpt_libs.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    hash_token,
)
from prisma.models import User as PrismaUser

from backend.data.db import prisma

logger = logging.getLogger(__name__)


class AuthService:
    """Handles authentication operations including password verification and token management."""

    def __init__(self):
        self.settings = get_settings()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against a bcrypt hash."""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            logger.warning(f"Password verification failed: {e}")
            return False

    async def register_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
    ) -> PrismaUser:
        """
        Register a new user with email and password.

        Creates both a User record and a Profile record.

        :param email: User's email address
        :param password: User's password (will be hashed)
        :param name: Optional display name
        :return: Created user record
        :raises ValueError: If email is already registered
        """
        # Check if user already exists
        existing = await prisma.user.find_unique(where={"email": email})
        if existing:
            raise ValueError("Email already registered")

        password_hash = self.hash_password(password)

        # Generate a unique username from email
        base_username = email.split("@")[0].lower()
        # Remove any characters that aren't alphanumeric or underscore
        base_username = re.sub(r"[^a-z0-9_]", "", base_username)
        if not base_username:
            base_username = "user"

        # Check if username is unique, if not add a number suffix
        username = base_username
        counter = 1
        while await prisma.profile.find_unique(where={"username": username}):
            username = f"{base_username}{counter}"
            counter += 1

        user = await prisma.user.create(
            data={
                "email": email,
                "passwordHash": password_hash,
                "name": name,
                "emailVerified": False,
                "role": "authenticated",
            }
        )

        # Create profile for the user
        display_name = name or base_username
        await prisma.profile.create(
            data={
                "userId": user.id,
                "name": display_name,
                "username": username,
                "description": "",
                "links": [],
            }
        )

        logger.info(f"Registered new user: {user.id} with profile username: {username}")
        return user

    async def authenticate_user(
        self, email: str, password: str
    ) -> Optional[PrismaUser]:
        """
        Authenticate a user with email and password.

        :param email: User's email address
        :param password: User's password
        :return: User record if authentication successful, None otherwise
        """
        user = await prisma.user.find_unique(where={"email": email})

        if not user:
            logger.debug(f"Authentication failed: user not found for email {email}")
            return None

        if not user.passwordHash:
            logger.debug(
                f"Authentication failed: no password set for user {user.id} "
                "(likely OAuth-only user)"
            )
            return None

        if self.verify_password(password, user.passwordHash):
            logger.debug(f"Authentication successful for user {user.id}")
            return user

        logger.debug(f"Authentication failed: invalid password for user {user.id}")
        return None

    async def create_tokens(self, user: PrismaUser) -> dict:
        """
        Create access and refresh tokens for a user.

        :param user: The user to create tokens for
        :return: Dictionary with access_token, refresh_token, token_type, and expires_in
        """
        # Create access token
        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role or "authenticated",
            email_verified=user.emailVerified,
        )

        # Create and store refresh token
        raw_refresh_token, hashed_refresh_token = create_refresh_token()
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

        await prisma.refreshtoken.create(
            data={
                "token": hashed_refresh_token,
                "userId": user.id,
                "expiresAt": expires_at,
            }
        )

        logger.debug(f"Created tokens for user {user.id}")

        return {
            "access_token": access_token,
            "refresh_token": raw_refresh_token,
            "token_type": "bearer",
            "expires_in": self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    async def refresh_access_token(self, refresh_token: str) -> Optional[dict]:
        """
        Refresh an access token using a refresh token.

        Implements token rotation: the old refresh token is revoked and a new one is issued.

        :param refresh_token: The refresh token
        :return: New tokens if successful, None if refresh token is invalid/expired
        """
        hashed_token = hash_token(refresh_token)

        # Find the refresh token
        stored_token = await prisma.refreshtoken.find_first(
            where={
                "token": hashed_token,
                "revokedAt": None,
                "expiresAt": {"gt": datetime.now(timezone.utc)},
            },
            include={"User": True},
        )

        if not stored_token or not stored_token.User:
            logger.debug("Refresh token not found or expired")
            return None

        # Revoke the old token (token rotation)
        await prisma.refreshtoken.update(
            where={"id": stored_token.id},
            data={"revokedAt": datetime.now(timezone.utc)},
        )

        logger.debug(f"Refreshed tokens for user {stored_token.User.id}")

        # Create new tokens
        return await self.create_tokens(stored_token.User)

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        Revoke a refresh token (logout).

        :param refresh_token: The refresh token to revoke
        :return: True if token was found and revoked, False otherwise
        """
        hashed_token = hash_token(refresh_token)

        result = await prisma.refreshtoken.update_many(
            where={"token": hashed_token, "revokedAt": None},
            data={"revokedAt": datetime.now(timezone.utc)},
        )

        if result > 0:
            logger.debug("Refresh token revoked")
            return True

        logger.debug("Refresh token not found or already revoked")
        return False

    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all refresh tokens for a user (logout from all devices).

        :param user_id: The user's ID
        :return: Number of tokens revoked
        """
        result = await prisma.refreshtoken.update_many(
            where={"userId": user_id, "revokedAt": None},
            data={"revokedAt": datetime.now(timezone.utc)},
        )

        logger.debug(f"Revoked {result} tokens for user {user_id}")
        return result

    async def get_user_by_google_id(self, google_id: str) -> Optional[PrismaUser]:
        """Get a user by their Google OAuth ID."""
        return await prisma.user.find_unique(where={"googleId": google_id})

    async def get_user_by_email(self, email: str) -> Optional[PrismaUser]:
        """Get a user by their email address."""
        return await prisma.user.find_unique(where={"email": email})

    async def create_or_update_google_user(
        self,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        email_verified: bool = False,
    ) -> PrismaUser:
        """
        Create or update a user from Google OAuth.

        If a user with the Google ID exists, return them.
        If a user with the email exists but no Google ID, link the account.
        Otherwise, create a new user.

        :param google_id: Google's unique user ID
        :param email: User's email from Google
        :param name: User's name from Google
        :param email_verified: Whether Google has verified the email
        :return: The user record
        """
        # Check if user exists with this Google ID
        user = await self.get_user_by_google_id(google_id)
        if user:
            return user

        # Check if user exists with this email
        user = await self.get_user_by_email(email)
        if user:
            # Link Google account to existing user
            updated_user = await prisma.user.update(
                where={"id": user.id},
                data={
                    "googleId": google_id,
                    "emailVerified": email_verified or user.emailVerified,
                },
            )
            if updated_user:
                logger.info(f"Linked Google account to existing user {updated_user.id}")
                return updated_user
            return user

        # Create new user with profile
        # Generate a unique username from email
        base_username = email.split("@")[0].lower()
        base_username = re.sub(r"[^a-z0-9_]", "", base_username)
        if not base_username:
            base_username = "user"

        username = base_username
        counter = 1
        while await prisma.profile.find_unique(where={"username": username}):
            username = f"{base_username}{counter}"
            counter += 1

        user = await prisma.user.create(
            data={
                "email": email,
                "googleId": google_id,
                "name": name,
                "emailVerified": email_verified,
                "role": "authenticated",
            }
        )

        # Create profile for the user
        display_name = name or base_username
        await prisma.profile.create(
            data={
                "userId": user.id,
                "name": display_name,
                "username": username,
                "description": "",
                "links": [],
            }
        )

        logger.info(
            f"Created new user from Google OAuth: {user.id} with profile: {username}"
        )
        return user

    async def create_password_reset_token(self, user_id: str) -> str:
        """
        Create a password reset token for a user.

        :param user_id: The user's ID
        :return: The raw token to send to the user
        """
        raw_token, hashed_token = create_refresh_token()  # Reuse token generation
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        await prisma.passwordresettoken.create(
            data={
                "token": hashed_token,
                "userId": user_id,
                "expiresAt": expires_at,
            }
        )

        return raw_token

    async def verify_password_reset_token(self, token: str) -> Optional[str]:
        """
        Verify a password reset token and return the user ID.

        :param token: The raw token from the user
        :return: User ID if valid, None otherwise
        """
        hashed_token = hash_token(token)

        stored_token = await prisma.passwordresettoken.find_first(
            where={
                "token": hashed_token,
                "usedAt": None,
                "expiresAt": {"gt": datetime.now(timezone.utc)},
            }
        )

        if not stored_token:
            return None

        return stored_token.userId

    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset a user's password using a password reset token.

        :param token: The password reset token
        :param new_password: The new password
        :return: True if successful, False if token is invalid
        """
        hashed_token = hash_token(token)

        # Find and validate token
        stored_token = await prisma.passwordresettoken.find_first(
            where={
                "token": hashed_token,
                "usedAt": None,
                "expiresAt": {"gt": datetime.now(timezone.utc)},
            }
        )

        if not stored_token:
            return False

        # Update password
        password_hash = self.hash_password(new_password)
        await prisma.user.update(
            where={"id": stored_token.userId},
            data={"passwordHash": password_hash},
        )

        # Mark token as used
        await prisma.passwordresettoken.update(
            where={"id": stored_token.id},
            data={"usedAt": datetime.now(timezone.utc)},
        )

        # Revoke all refresh tokens for security
        await self.revoke_all_user_tokens(stored_token.userId)

        logger.info(f"Password reset for user {stored_token.userId}")
        return True
