"""
JWT Token Service for OAuth 2.0 Provider.

Handles generation and validation of:
- Access tokens (JWT)
- Refresh tokens (opaque)
- ID tokens (JWT, OIDC)
"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
    generate_private_key,
)
from pydantic import BaseModel

from backend.util.settings import Settings


class TokenClaims(BaseModel):
    """Decoded token claims."""

    iss: str  # Issuer
    sub: str  # Subject (user ID)
    aud: str  # Audience (client ID)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: str  # JWT ID
    scope: str  # Space-separated scopes
    client_id: str  # Client ID


class OAuthTokenService:
    """
    Service for generating and validating OAuth tokens.

    Uses RS256 (RSA with SHA-256) for JWT signing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or Settings()
        self._private_key: Optional[RSAPrivateKey] = None
        self._public_key: Optional[RSAPublicKey] = None
        self._algorithm = "RS256"

    @property
    def issuer(self) -> str:
        """Get the token issuer URL."""
        return self._settings.config.platform_base_url or "https://platform.agpt.co"

    @property
    def key_id(self) -> str:
        """Get the key ID for JWKS."""
        return self._settings.secrets.oauth_jwt_key_id or "default-key-id"

    def _get_private_key(self) -> RSAPrivateKey:
        """Load or generate the private key."""
        if self._private_key is not None:
            return self._private_key

        key_pem = self._settings.secrets.oauth_jwt_private_key
        if key_pem:
            loaded_key = serialization.load_pem_private_key(
                key_pem.encode(), password=None
            )
            if not isinstance(loaded_key, RSAPrivateKey):
                raise ValueError("OAuth JWT private key must be RSA")
            self._private_key = loaded_key
        else:
            # Generate a key for development (should not be used in production)
            self._private_key = generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
        return self._private_key

    def _get_public_key(self) -> RSAPublicKey:
        """Get the public key from the private key."""
        if self._public_key is not None:
            return self._public_key

        key_pem = self._settings.secrets.oauth_jwt_public_key
        if key_pem:
            loaded_key = serialization.load_pem_public_key(key_pem.encode())
            if not isinstance(loaded_key, RSAPublicKey):
                raise ValueError("OAuth JWT public key must be RSA")
            self._public_key = loaded_key
        else:
            self._public_key = self._get_private_key().public_key()
        return self._public_key

    def generate_access_token(
        self,
        user_id: str,
        client_id: str,
        scopes: list[str],
        expires_in: int = 3600,
    ) -> tuple[str, datetime]:
        """
        Generate a JWT access token.

        Args:
            user_id: User ID (subject)
            client_id: Client ID (audience)
            scopes: List of granted scopes
            expires_in: Token lifetime in seconds

        Returns:
            Tuple of (token string, expiration datetime)
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)

        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": client_id,
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "scope": " ".join(scopes),
            "client_id": client_id,
        }

        token = jwt.encode(
            payload,
            self._get_private_key(),
            algorithm=self._algorithm,
            headers={"kid": self.key_id},
        )
        return token, expires_at

    def generate_refresh_token(self) -> str:
        """
        Generate an opaque refresh token.

        Returns:
            URL-safe random token string
        """
        return secrets.token_urlsafe(48)

    def generate_id_token(
        self,
        user_id: str,
        client_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        nonce: Optional[str] = None,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate an OIDC ID token.

        Args:
            user_id: User ID (subject)
            client_id: Client ID (audience)
            email: User's email (optional)
            name: User's name (optional)
            nonce: OIDC nonce for replay protection (optional)
            expires_in: Token lifetime in seconds

        Returns:
            JWT ID token string
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)

        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": client_id,
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "auth_time": int(now.timestamp()),
        }

        if email:
            payload["email"] = email
            payload["email_verified"] = True
        if name:
            payload["name"] = name
        if nonce:
            payload["nonce"] = nonce

        return jwt.encode(
            payload,
            self._get_private_key(),
            algorithm=self._algorithm,
            headers={"kid": self.key_id},
        )

    def verify_access_token(
        self,
        token: str,
        expected_client_id: Optional[str] = None,
    ) -> TokenClaims:
        """
        Verify and decode a JWT access token.

        Args:
            token: JWT token string
            expected_client_id: Expected client ID (audience)

        Returns:
            Decoded token claims

        Raises:
            jwt.ExpiredSignatureError: Token has expired
            jwt.InvalidTokenError: Token is invalid
        """
        options = {}
        if expected_client_id:
            options["audience"] = expected_client_id

        payload = jwt.decode(
            token,
            self._get_public_key(),
            algorithms=[self._algorithm],
            issuer=self.issuer,
            options={"verify_aud": bool(expected_client_id)},
            **options,
        )

        return TokenClaims(
            iss=payload["iss"],
            sub=payload["sub"],
            aud=payload.get("aud", payload.get("client_id", "")),
            exp=payload["exp"],
            iat=payload["iat"],
            jti=payload["jti"],
            scope=payload.get("scope", ""),
            client_id=payload.get("client_id", payload.get("aud", "")),
        )

    @staticmethod
    def hash_token(token: str) -> str:
        """
        Hash a token for secure storage.

        Args:
            token: Token string to hash

        Returns:
            SHA-256 hash of the token
        """
        return hashlib.sha256(token.encode()).hexdigest()

    def get_jwks(self) -> dict:
        """
        Get the JSON Web Key Set (JWKS) for public key distribution.

        Returns:
            JWKS dictionary with public key(s)
        """
        public_key = self._get_public_key()
        public_numbers = public_key.public_numbers()

        # Convert to base64url encoding without padding
        def int_to_base64url(n: int, length: int) -> str:
            data = n.to_bytes(length, byteorder="big")
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        # RSA modulus and exponent
        n = int_to_base64url(public_numbers.n, (public_numbers.n.bit_length() + 7) // 8)
        e = int_to_base64url(public_numbers.e, 3)

        return {
            "keys": [
                {
                    "kty": "RSA",
                    "use": "sig",
                    "kid": self.key_id,
                    "alg": self._algorithm,
                    "n": n,
                    "e": e,
                }
            ]
        }


# Module-level singleton
_token_service: Optional[OAuthTokenService] = None


def get_token_service() -> OAuthTokenService:
    """Get the singleton token service instance."""
    global _token_service
    if _token_service is None:
        _token_service = OAuthTokenService()
    return _token_service
