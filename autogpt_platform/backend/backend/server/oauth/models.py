"""
Pydantic models for OAuth 2.0 requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

# ============================================================
# Enums and Constants
# ============================================================


class OAuthScope(str, Enum):
    """Supported OAuth scopes."""

    # OpenID Connect standard scopes
    OPENID = "openid"
    PROFILE = "profile"
    EMAIL = "email"

    # AutoGPT-specific scopes
    INTEGRATIONS_LIST = "integrations:list"
    INTEGRATIONS_CONNECT = "integrations:connect"
    INTEGRATIONS_DELETE = "integrations:delete"
    AGENTS_EXECUTE = "agents:execute"


SCOPE_DESCRIPTIONS: dict[str, str] = {
    OAuthScope.OPENID.value: "Access your user ID",
    OAuthScope.PROFILE.value: "Access your profile information (name)",
    OAuthScope.EMAIL.value: "Access your email address",
    OAuthScope.INTEGRATIONS_LIST.value: "View your connected integrations",
    OAuthScope.INTEGRATIONS_CONNECT.value: "Connect new integrations on your behalf",
    OAuthScope.INTEGRATIONS_DELETE.value: "Delete integrations on your behalf",
    OAuthScope.AGENTS_EXECUTE.value: "Run agents on your behalf",
}


# ============================================================
# Authorization Request/Response Models
# ============================================================


class AuthorizationRequest(BaseModel):
    """OAuth 2.0 Authorization Request (RFC 6749 Section 4.1.1)."""

    response_type: Literal["code"] = Field(
        ..., description="Must be 'code' for authorization code flow"
    )
    client_id: str = Field(..., description="Client identifier")
    redirect_uri: str = Field(..., description="Redirect URI after authorization")
    scope: str = Field(default="", description="Space-separated list of scopes")
    state: str = Field(..., description="CSRF protection token (required)")
    code_challenge: str = Field(..., description="PKCE code challenge (required)")
    code_challenge_method: Literal["S256"] = Field(
        default="S256", description="PKCE method (only S256 supported)"
    )
    nonce: Optional[str] = Field(None, description="OIDC nonce for replay protection")
    prompt: Optional[Literal["consent", "login", "none"]] = Field(
        None, description="Prompt behavior"
    )


class ConsentFormData(BaseModel):
    """Consent form submission data."""

    consent_token: str = Field(..., description="CSRF token for consent")
    authorize: bool = Field(..., description="Whether user authorized")


# ============================================================
# Token Request/Response Models
# ============================================================


class TokenRequest(BaseModel):
    """OAuth 2.0 Token Request (RFC 6749 Section 4.1.3)."""

    grant_type: Literal["authorization_code", "refresh_token"] = Field(
        ..., description="Grant type"
    )
    code: Optional[str] = Field(
        None, description="Authorization code (for authorization_code grant)"
    )
    redirect_uri: Optional[str] = Field(
        None, description="Must match authorization request"
    )
    client_id: str = Field(..., description="Client identifier")
    client_secret: Optional[str] = Field(
        None, description="Client secret (for confidential clients)"
    )
    code_verifier: Optional[str] = Field(
        None, description="PKCE code verifier (for authorization_code grant)"
    )
    refresh_token: Optional[str] = Field(
        None, description="Refresh token (for refresh_token grant)"
    )
    scope: Optional[str] = Field(
        None, description="Requested scopes (for refresh_token grant)"
    )


class TokenResponse(BaseModel):
    """OAuth 2.0 Token Response (RFC 6749 Section 5.1)."""

    access_token: str = Field(..., description="Access token")
    token_type: Literal["Bearer"] = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token lifetime in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    scope: Optional[str] = Field(None, description="Granted scopes")
    id_token: Optional[str] = Field(None, description="OIDC ID token")


# ============================================================
# UserInfo Response Model
# ============================================================


class UserInfoResponse(BaseModel):
    """OIDC UserInfo Response."""

    sub: str = Field(..., description="User ID (subject)")
    email: Optional[str] = Field(None, description="User email")
    email_verified: Optional[bool] = Field(
        None, description="Whether email is verified"
    )
    name: Optional[str] = Field(None, description="User display name")
    updated_at: Optional[int] = Field(None, description="Last profile update timestamp")


# ============================================================
# OIDC Discovery Models
# ============================================================


class OpenIDConfiguration(BaseModel):
    """OIDC Discovery Document."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    revocation_endpoint: str
    jwks_uri: str
    scopes_supported: list[str]
    response_types_supported: list[str]
    grant_types_supported: list[str]
    token_endpoint_auth_methods_supported: list[str]
    code_challenge_methods_supported: list[str]
    subject_types_supported: list[str]
    id_token_signing_alg_values_supported: list[str]


class JWK(BaseModel):
    """JSON Web Key."""

    kty: str = Field(..., description="Key type (RSA)")
    use: str = Field(default="sig", description="Key use (signature)")
    kid: str = Field(..., description="Key ID")
    alg: str = Field(default="RS256", description="Algorithm")
    n: str = Field(..., description="RSA modulus")
    e: str = Field(..., description="RSA exponent")


class JWKS(BaseModel):
    """JSON Web Key Set."""

    keys: list[JWK]


# ============================================================
# Client Management Models
# ============================================================


class RegisterClientRequest(BaseModel):
    """Request to register a new OAuth client."""

    name: str = Field(..., min_length=1, max_length=100, description="Client name")
    description: Optional[str] = Field(
        None, max_length=500, description="Client description"
    )
    logo_url: Optional[HttpUrl] = Field(None, description="Logo URL")
    homepage_url: Optional[HttpUrl] = Field(None, description="Homepage URL")
    privacy_policy_url: Optional[HttpUrl] = Field(
        None, description="Privacy policy URL"
    )
    terms_of_service_url: Optional[HttpUrl] = Field(
        None, description="Terms of service URL"
    )
    redirect_uris: list[str] = Field(
        ..., min_length=1, description="Allowed redirect URIs"
    )
    client_type: Literal["public", "confidential"] = Field(
        default="public", description="Client type"
    )
    webhook_domains: list[str] = Field(
        default_factory=list, description="Allowed webhook domains"
    )


class UpdateClientRequest(BaseModel):
    """Request to update an OAuth client."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    logo_url: Optional[HttpUrl] = None
    homepage_url: Optional[HttpUrl] = None
    privacy_policy_url: Optional[HttpUrl] = None
    terms_of_service_url: Optional[HttpUrl] = None
    redirect_uris: Optional[list[str]] = None
    webhook_domains: Optional[list[str]] = None


class ClientResponse(BaseModel):
    """OAuth client response."""

    id: str
    client_id: str
    client_type: str
    name: str
    description: Optional[str]
    logo_url: Optional[str]
    homepage_url: Optional[str]
    privacy_policy_url: Optional[str]
    terms_of_service_url: Optional[str]
    redirect_uris: list[str]
    allowed_scopes: list[str]
    webhook_domains: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


class ClientSecretResponse(BaseModel):
    """Response containing newly generated client credentials."""

    client_id: str
    client_secret: str = Field(
        ..., description="Client secret (only shown once, store securely)"
    )
    webhook_secret: str = Field(
        ...,
        description="Webhook secret for HMAC signing (only shown once, store securely)",
    )


# ============================================================
# Token Introspection/Revocation Models
# ============================================================


class TokenRevocationRequest(BaseModel):
    """Token revocation request (RFC 7009)."""

    token: str = Field(..., description="Token to revoke")
    token_type_hint: Optional[Literal["access_token", "refresh_token"]] = Field(
        None, description="Hint about token type"
    )


class TokenIntrospectionRequest(BaseModel):
    """Token introspection request (RFC 7662)."""

    token: str = Field(..., description="Token to introspect")
    token_type_hint: Optional[Literal["access_token", "refresh_token"]] = Field(
        None, description="Hint about token type"
    )


class TokenIntrospectionResponse(BaseModel):
    """Token introspection response."""

    active: bool = Field(..., description="Whether the token is active")
    scope: Optional[str] = Field(None, description="Token scopes")
    client_id: Optional[str] = Field(
        None, description="Client that token was issued to"
    )
    username: Optional[str] = Field(None, description="User identifier")
    token_type: Optional[str] = Field(None, description="Token type")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    sub: Optional[str] = Field(None, description="Subject (user ID)")
    aud: Optional[str] = Field(None, description="Audience")
    iss: Optional[str] = Field(None, description="Issuer")
