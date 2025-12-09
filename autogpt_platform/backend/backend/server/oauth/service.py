"""
Core OAuth 2.0 service logic.

Handles:
- Client validation and lookup
- Authorization code generation and exchange
- Token issuance and refresh
- User consent management
- Audit logging
"""

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from prisma.enums import OAuthClientStatus
from prisma.models import OAuthAuthorization, OAuthClient, User

from backend.data.db import prisma
from backend.server.oauth.errors import (
    InvalidClientError,
    InvalidGrantError,
    InvalidRequestError,
    InvalidScopeError,
)
from backend.server.oauth.models import TokenResponse
from backend.server.oauth.pkce import verify_code_challenge
from backend.server.oauth.token_service import OAuthTokenService, get_token_service


class OAuthService:
    """Core OAuth 2.0 service."""

    def __init__(self, token_service: Optional[OAuthTokenService] = None):
        self.token_service = token_service or get_token_service()

    # ================================================================
    # Client Operations
    # ================================================================

    async def get_client(self, client_id: str) -> Optional[OAuthClient]:
        """Get an OAuth client by client_id."""
        return await prisma.oauthclient.find_unique(where={"clientId": client_id})

    async def validate_client(
        self,
        client_id: str,
        redirect_uri: str,
        scopes: list[str],
    ) -> OAuthClient:
        """
        Validate a client for authorization.

        Args:
            client_id: Client identifier
            redirect_uri: Requested redirect URI
            scopes: Requested scopes

        Returns:
            Validated OAuthClient

        Raises:
            InvalidClientError: Client not found or inactive
            InvalidRequestError: Invalid redirect URI
            InvalidScopeError: Invalid scopes requested
        """
        client = await self.get_client(client_id)

        if not client:
            raise InvalidClientError(f"Client '{client_id}' not found")

        if client.status != OAuthClientStatus.ACTIVE:
            raise InvalidClientError(f"Client '{client_id}' is not active")

        # Validate redirect URI (exact match required)
        if redirect_uri not in client.redirectUris:
            raise InvalidRequestError(
                f"Redirect URI '{redirect_uri}' is not registered for this client"
            )

        # Validate scopes
        invalid_scopes = set(scopes) - set(client.allowedScopes)
        if invalid_scopes:
            raise InvalidScopeError(
                f"Scopes not allowed for this client: {', '.join(invalid_scopes)}"
            )

        return client

    async def validate_client_secret(
        self,
        client_id: str,
        client_secret: Optional[str],
    ) -> OAuthClient:
        """
        Validate client authentication for token endpoint.

        Args:
            client_id: Client identifier
            client_secret: Client secret (for confidential clients)

        Returns:
            Validated OAuthClient

        Raises:
            InvalidClientError: Invalid client or credentials
        """
        client = await self.get_client(client_id)

        if not client:
            raise InvalidClientError(f"Client '{client_id}' not found")

        if client.status != OAuthClientStatus.ACTIVE:
            raise InvalidClientError(f"Client '{client_id}' is not active")

        # Confidential clients must provide secret
        if client.clientType == "confidential":
            if not client_secret:
                raise InvalidClientError("Client secret required")

            # Hash and compare
            secret_hash = self._hash_secret(
                client_secret, client.clientSecretSalt or ""
            )
            if not secrets.compare_digest(secret_hash, client.clientSecretHash or ""):
                raise InvalidClientError("Invalid client credentials")

        return client

    @staticmethod
    def _hash_secret(secret: str, salt: str) -> str:
        """Hash a client secret with salt."""
        return hashlib.sha256(f"{salt}{secret}".encode()).hexdigest()

    # ================================================================
    # Authorization Code Operations
    # ================================================================

    async def create_authorization_code(
        self,
        user_id: str,
        client_id: str,
        redirect_uri: str,
        scopes: list[str],
        code_challenge: str,
        code_challenge_method: str = "S256",
        nonce: Optional[str] = None,
    ) -> str:
        """
        Create a new authorization code.

        Args:
            user_id: User who authorized
            client_id: Client being authorized
            redirect_uri: Redirect URI for callback
            scopes: Granted scopes
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE method (S256)
            nonce: OIDC nonce (optional)

        Returns:
            Authorization code string
        """
        code = secrets.token_urlsafe(32)
        code_hash = self.token_service.hash_token(code)

        # Get the OAuthClient to link
        client = await self.get_client(client_id)
        if not client:
            raise InvalidClientError(f"Client '{client_id}' not found")

        await prisma.oauthauthorizationcode.create(
            data={  # type: ignore[typeddict-item]
                "codeHash": code_hash,
                "userId": user_id,
                "clientId": client.id,
                "redirectUri": redirect_uri,
                "scopes": scopes,
                "codeChallenge": code_challenge,
                "codeChallengeMethod": code_challenge_method,
                "nonce": nonce,
                "expiresAt": datetime.now(timezone.utc) + timedelta(minutes=10),
            }
        )

        return code

    async def exchange_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> TokenResponse:
        """
        Exchange an authorization code for tokens.

        Args:
            code: Authorization code
            client_id: Client identifier
            redirect_uri: Must match original redirect URI
            code_verifier: PKCE code verifier

        Returns:
            TokenResponse with access token, refresh token, etc.

        Raises:
            InvalidGrantError: Invalid or expired code
            InvalidRequestError: PKCE verification failed
        """
        code_hash = self.token_service.hash_token(code)

        # Find the authorization code
        auth_code = await prisma.oauthauthorizationcode.find_unique(
            where={"codeHash": code_hash},
            include={"Client": True, "User": True},
        )

        if not auth_code:
            raise InvalidGrantError("Authorization code not found")

        # Ensure Client relation is loaded
        if not auth_code.Client:
            raise InvalidGrantError("Authorization code client not found")

        # Check if already used
        if auth_code.usedAt:
            # Code reuse is a security incident - revoke all tokens for this authorization
            await self._revoke_tokens_for_client_user(
                auth_code.Client.clientId, auth_code.userId
            )
            raise InvalidGrantError("Authorization code has already been used")

        # Check expiration
        if auth_code.expiresAt < datetime.now(timezone.utc):
            raise InvalidGrantError("Authorization code has expired")

        # Validate client
        if auth_code.Client.clientId != client_id:
            raise InvalidGrantError("Client ID mismatch")

        # Validate redirect URI
        if auth_code.redirectUri != redirect_uri:
            raise InvalidGrantError("Redirect URI mismatch")

        # Verify PKCE
        if not verify_code_challenge(
            code_verifier, auth_code.codeChallenge, auth_code.codeChallengeMethod
        ):
            raise InvalidRequestError("PKCE verification failed")

        # Mark code as used
        await prisma.oauthauthorizationcode.update(
            where={"id": auth_code.id},
            data={"usedAt": datetime.now(timezone.utc)},
        )

        # Create or update authorization record
        await self._upsert_authorization(
            auth_code.userId, auth_code.Client.id, auth_code.scopes
        )

        # Generate tokens
        return await self._create_tokens(
            user_id=auth_code.userId,
            client=auth_code.Client,
            scopes=auth_code.scopes,
            nonce=auth_code.nonce,
            user=auth_code.User,
        )

    async def refresh_access_token(
        self,
        refresh_token: str,
        client_id: str,
        requested_scopes: Optional[list[str]] = None,
    ) -> TokenResponse:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Refresh token string
            client_id: Client identifier
            requested_scopes: Optionally request fewer scopes

        Returns:
            New TokenResponse

        Raises:
            InvalidGrantError: Invalid or expired refresh token
        """
        token_hash = self.token_service.hash_token(refresh_token)

        # Find the refresh token
        stored_token = await prisma.oauthrefreshtoken.find_unique(
            where={"tokenHash": token_hash},
            include={"Client": True, "User": True},
        )

        if not stored_token:
            raise InvalidGrantError("Refresh token not found")

        # Ensure Client relation is loaded
        if not stored_token.Client:
            raise InvalidGrantError("Refresh token client not found")

        # Check if revoked
        if stored_token.revokedAt:
            raise InvalidGrantError("Refresh token has been revoked")

        # Check expiration
        if stored_token.expiresAt < datetime.now(timezone.utc):
            raise InvalidGrantError("Refresh token has expired")

        # Validate client
        if stored_token.Client.clientId != client_id:
            raise InvalidGrantError("Client ID mismatch")

        # Determine scopes
        scopes = stored_token.scopes
        if requested_scopes:
            # Can only request a subset of original scopes
            invalid = set(requested_scopes) - set(stored_token.scopes)
            if invalid:
                raise InvalidScopeError(
                    f"Cannot request scopes not in original grant: {', '.join(invalid)}"
                )
            scopes = requested_scopes

        # Generate new tokens (rotates refresh token)
        return await self._create_tokens(
            user_id=stored_token.userId,
            client=stored_token.Client,
            scopes=scopes,
            user=stored_token.User,
            old_refresh_token_id=stored_token.id,
        )

    # ================================================================
    # Token Operations
    # ================================================================

    async def _create_tokens(
        self,
        user_id: str,
        client: OAuthClient,
        scopes: list[str],
        user: Optional[User] = None,
        nonce: Optional[str] = None,
        old_refresh_token_id: Optional[str] = None,
    ) -> TokenResponse:
        """
        Create access and refresh tokens.

        Args:
            user_id: User ID
            client: OAuth client
            scopes: Granted scopes
            user: User object (for ID token claims)
            nonce: OIDC nonce
            old_refresh_token_id: ID of refresh token being rotated

        Returns:
            TokenResponse
        """
        # Generate access token
        access_token, access_expires_at = self.token_service.generate_access_token(
            user_id=user_id,
            client_id=client.clientId,
            scopes=scopes,
            expires_in=client.tokenLifetimeSecs,
        )

        # Store access token hash
        await prisma.oauthaccesstoken.create(
            data={  # type: ignore[typeddict-item]
                "tokenHash": self.token_service.hash_token(access_token),
                "userId": user_id,
                "clientId": client.id,
                "scopes": scopes,
                "expiresAt": access_expires_at,
            }
        )

        # Generate refresh token
        refresh_token = self.token_service.generate_refresh_token()
        refresh_expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=client.refreshTokenLifetimeSecs
        )

        await prisma.oauthrefreshtoken.create(
            data={  # type: ignore[typeddict-item]
                "tokenHash": self.token_service.hash_token(refresh_token),
                "userId": user_id,
                "clientId": client.id,
                "scopes": scopes,
                "expiresAt": refresh_expires_at,
            }
        )

        # Revoke old refresh token if rotating
        if old_refresh_token_id:
            await prisma.oauthrefreshtoken.update(
                where={"id": old_refresh_token_id},
                data={"revokedAt": datetime.now(timezone.utc)},
            )

        # Generate ID token if openid scope requested
        id_token = None
        if "openid" in scopes and user:
            email = user.email if "email" in scopes else None
            name = user.name if "profile" in scopes else None
            id_token = self.token_service.generate_id_token(
                user_id=user_id,
                client_id=client.clientId,
                email=email,
                name=name,
                nonce=nonce,
            )

        # Audit log
        await self._audit_log(
            event_type="token.issued",
            user_id=user_id,
            client_id=client.clientId,
            details={"scopes": scopes},
        )

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=client.tokenLifetimeSecs,
            refresh_token=refresh_token,
            scope=" ".join(scopes),
            id_token=id_token,
        )

    async def revoke_token(
        self,
        token: str,
        token_type_hint: Optional[str] = None,
    ) -> bool:
        """
        Revoke an access or refresh token.

        Args:
            token: Token to revoke
            token_type_hint: Hint about token type

        Returns:
            True if token was found and revoked
        """
        token_hash = self.token_service.hash_token(token)
        now = datetime.now(timezone.utc)

        # Try refresh token first if hinted or no hint
        if token_type_hint in (None, "refresh_token"):
            result = await prisma.oauthrefreshtoken.update_many(
                where={"tokenHash": token_hash, "revokedAt": None},
                data={"revokedAt": now},
            )
            if result > 0:
                return True

        # Try access token
        if token_type_hint in (None, "access_token"):
            result = await prisma.oauthaccesstoken.update_many(
                where={"tokenHash": token_hash, "revokedAt": None},
                data={"revokedAt": now},
            )
            if result > 0:
                return True

        return False

    async def _revoke_tokens_for_client_user(
        self,
        client_id: str,
        user_id: str,
    ) -> None:
        """Revoke all tokens for a client-user pair (security incident response)."""
        client = await self.get_client(client_id)
        if not client:
            return

        now = datetime.now(timezone.utc)

        await prisma.oauthaccesstoken.update_many(
            where={"clientId": client.id, "userId": user_id, "revokedAt": None},
            data={"revokedAt": now},
        )

        await prisma.oauthrefreshtoken.update_many(
            where={"clientId": client.id, "userId": user_id, "revokedAt": None},
            data={"revokedAt": now},
        )

        await self._audit_log(
            event_type="tokens.revoked.security",
            user_id=user_id,
            client_id=client_id,
            details={"reason": "authorization_code_reuse"},
        )

    # ================================================================
    # Authorization (Consent) Operations
    # ================================================================

    async def get_authorization(
        self,
        user_id: str,
        client_id: str,
    ) -> Optional[OAuthAuthorization]:
        """Get existing authorization for user-client pair."""
        client = await self.get_client(client_id)
        if not client:
            return None

        return await prisma.oauthauthorization.find_unique(
            where={
                "userId_clientId": {
                    "userId": user_id,
                    "clientId": client.id,
                }
            }
        )

    async def has_valid_authorization(
        self,
        user_id: str,
        client_id: str,
        scopes: list[str],
    ) -> bool:
        """
        Check if user has already authorized these scopes for this client.

        Args:
            user_id: User ID
            client_id: Client identifier
            scopes: Requested scopes

        Returns:
            True if user has already authorized all requested scopes
        """
        auth = await self.get_authorization(user_id, client_id)
        if not auth or auth.revokedAt:
            return False

        # Check if all requested scopes are already authorized
        return set(scopes).issubset(set(auth.scopes))

    async def _upsert_authorization(
        self,
        user_id: str,
        client_db_id: str,
        scopes: list[str],
    ) -> None:
        """Create or update an authorization record."""
        existing = await prisma.oauthauthorization.find_unique(
            where={
                "userId_clientId": {
                    "userId": user_id,
                    "clientId": client_db_id,
                }
            }
        )

        if existing:
            # Merge scopes
            merged_scopes = list(set(existing.scopes) | set(scopes))
            await prisma.oauthauthorization.update(
                where={"id": existing.id},
                data={"scopes": merged_scopes, "revokedAt": None},
            )
        else:
            await prisma.oauthauthorization.create(
                data={  # type: ignore[typeddict-item]
                    "userId": user_id,
                    "clientId": client_db_id,
                    "scopes": scopes,
                }
            )

    # ================================================================
    # Audit Logging
    # ================================================================

    async def _audit_log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        grant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create an audit log entry."""
        # Convert details to JSON for Prisma's Json field
        details_json = json.dumps(details or {})
        await prisma.oauthauditlog.create(
            data={
                "eventType": event_type,
                "userId": user_id,
                "clientId": client_id,
                "grantId": grant_id,
                "ipAddress": ip_address,
                "userAgent": user_agent,
                "details": json.loads(details_json),  # type: ignore[arg-type]
            }
        )


# Module-level singleton
_oauth_service: Optional[OAuthService] = None


def get_oauth_service() -> OAuthService:
    """Get the singleton OAuth service instance."""
    global _oauth_service
    if _oauth_service is None:
        _oauth_service = OAuthService()
    return _oauth_service
