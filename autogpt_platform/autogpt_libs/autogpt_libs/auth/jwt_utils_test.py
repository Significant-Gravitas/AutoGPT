"""
Comprehensive tests for JWT token parsing and validation.
Ensures 100% line and branch coverage for JWT security functions.

Verification is JWKS-only: the platform auth service issues asymmetric
(ES256) tokens, and symmetric (HS*) tokens are rejected outright.
"""

import base64
import json
import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from pytest_mock import MockerFixture

from autogpt_libs.auth import config, jwt_utils
from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.models import User

MOCK_JWKS_URL = "http://localhost:3000/api/auth/jwks"
_SIGNING_KID = "test-key-1"
TEST_USER_PAYLOAD = {
    "sub": "test-user-id",
    "role": "user",
    "aud": "authenticated",
    "email": "test@example.com",
}
TEST_ADMIN_PAYLOAD = {
    "sub": "admin-user-id",
    "role": "admin",
    "aud": "authenticated",
    "email": "admin@example.com",
}

# The active JWKS signing key, installed by the autouse fixture so the
# module-level create_token() helper can sign tokens the mocked JWK set trusts.
_signing_key: ec.EllipticCurvePrivateKey | None = None


def make_es256_keypair(kid: str = _SIGNING_KID):
    """Generate an EC P-256 keypair and the matching JWK set document."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return private_key, {"keys": [jwk]}


@pytest.fixture(autouse=True)
def mock_config(mocker: MockerFixture):
    """Configure a JWKS endpoint and serve a single ES256 signing key."""
    global _signing_key
    private_key, jwk_set = make_es256_keypair()
    _signing_key = private_key

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": MOCK_JWKS_URL}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)
    mocker.patch.object(jwt.PyJWKClient, "fetch_data", return_value=jwk_set)
    yield
    _signing_key = None


def create_token(payload, private_key=None, kid: str = _SIGNING_KID) -> str:
    """Helper to create ES256 JWT tokens signed by the active JWKS key."""
    if private_key is None:
        private_key = _signing_key
    assert private_key is not None
    return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": kid})


def test_parse_jwt_token_valid():
    """Test parsing a valid JWT token."""
    token = create_token(TEST_USER_PAYLOAD)
    result = jwt_utils.parse_jwt_token(token)

    assert result["sub"] == "test-user-id"
    assert result["role"] == "user"
    assert result["aud"] == "authenticated"


def test_parse_jwt_token_expired():
    """Test parsing an expired JWT token."""
    expired_payload = {
        **TEST_USER_PAYLOAD,
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    token = create_token(expired_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Token has expired" in str(exc_info.value)


def test_parse_jwt_token_invalid_signature():
    """A token signed by a key that is not in the JWK set is rejected."""
    other_key, _ = make_es256_keypair(kid=_SIGNING_KID)
    token = create_token(TEST_USER_PAYLOAD, private_key=other_key)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_malformed():
    """Test parsing a malformed token."""
    malformed_tokens = [
        "not.a.token",
        "invalid",
        "",
        # Header only
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
        # No signature
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0",
    ]

    for token in malformed_tokens:
        with pytest.raises(ValueError) as exc_info:
            jwt_utils.parse_jwt_token(token)
        assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_wrong_audience():
    """Test parsing a token with wrong audience."""
    wrong_aud_payload = {**TEST_USER_PAYLOAD, "aud": "wrong-audience"}
    token = create_token(wrong_aud_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_missing_audience():
    """Test parsing a token without audience claim."""
    no_aud_payload = {k: v for k, v in TEST_USER_PAYLOAD.items() if k != "aud"}
    token = create_token(no_aud_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


async def test_get_jwt_payload_with_valid_token():
    """Test extracting JWT payload with valid bearer token."""
    token = create_token(TEST_USER_PAYLOAD)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    result = await jwt_utils.get_jwt_payload(credentials)
    assert result["sub"] == "test-user-id"
    assert result["role"] == "user"


async def test_get_jwt_payload_no_credentials():
    """Test JWT payload when no credentials provided."""
    with pytest.raises(HTTPException) as exc_info:
        await jwt_utils.get_jwt_payload(None)
    assert exc_info.value.status_code == 401
    assert "Authorization header is missing" in exc_info.value.detail


async def test_get_jwt_payload_invalid_token():
    """Test JWT payload extraction with invalid token."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid.token.here"
    )

    with pytest.raises(HTTPException) as exc_info:
        await jwt_utils.get_jwt_payload(credentials)
    assert exc_info.value.status_code == 401
    assert "Invalid token" in exc_info.value.detail


def test_verify_user_with_valid_user():
    """Test verifying a valid user."""
    user = jwt_utils.verify_user(TEST_USER_PAYLOAD, admin_only=False)
    assert isinstance(user, User)
    assert user.user_id == "test-user-id"
    assert user.role == "user"
    assert user.email == "test@example.com"


def test_verify_user_with_admin():
    """Test verifying an admin user."""
    user = jwt_utils.verify_user(TEST_ADMIN_PAYLOAD, admin_only=True)
    assert isinstance(user, User)
    assert user.user_id == "admin-user-id"
    assert user.role == "admin"


def test_verify_user_admin_only_with_regular_user():
    """Test verifying regular user when admin is required."""
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(TEST_USER_PAYLOAD, admin_only=True)
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


def test_verify_user_no_payload():
    """Test verifying user with no payload."""
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(None, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "Authorization header is missing" in exc_info.value.detail


def test_verify_user_missing_sub():
    """Test verifying user with payload missing 'sub' field."""
    invalid_payload = {"role": "user", "email": "test@example.com"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_empty_sub():
    """Test verifying user with empty 'sub' field."""
    invalid_payload = {"sub": "", "role": "user"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_none_sub():
    """Test verifying user with None 'sub' field."""
    invalid_payload = {"sub": None, "role": "user"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_missing_role_admin_check():
    """A missing 'role' claim under admin_only must fail closed (403), not
    KeyError -> 500."""
    no_role_payload = {"sub": "user-id"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(no_role_payload, admin_only=True)
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


# ======================== EDGE CASES ======================== #


def test_jwt_with_additional_claims():
    """Test JWT token with additional custom claims."""
    extra_claims_payload = {
        "sub": "user-id",
        "role": "user",
        "aud": "authenticated",
        "custom_claim": "custom_value",
        "permissions": ["read", "write"],
        "metadata": {"key": "value"},
    }
    token = create_token(extra_claims_payload)

    result = jwt_utils.parse_jwt_token(token)
    assert result["sub"] == "user-id"
    assert result["custom_claim"] == "custom_value"
    assert result["permissions"] == ["read", "write"]


def test_jwt_with_numeric_sub():
    """Test JWT token with numeric user ID."""
    payload = {
        "sub": 12345,  # Numeric ID
        "role": "user",
        "aud": "authenticated",
    }
    # Should convert to string internally
    user = jwt_utils.verify_user(payload, admin_only=False)
    assert user.user_id == 12345


def test_jwt_with_very_long_sub():
    """Test JWT token with very long user ID."""
    long_id = "a" * 1000
    payload = {
        "sub": long_id,
        "role": "user",
        "aud": "authenticated",
    }
    user = jwt_utils.verify_user(payload, admin_only=False)
    assert user.user_id == long_id


def test_jwt_with_special_characters_in_claims():
    """Test JWT token with special characters in claims."""
    payload = {
        "sub": "user@example.com/special-chars!@#$%",
        "role": "admin",
        "aud": "authenticated",
        "email": "test+special@example.com",
    }
    user = jwt_utils.verify_user(payload, admin_only=True)
    assert "special-chars!@#$%" in user.user_id


def test_jwt_with_future_iat():
    """Test JWT token with issued-at time in future."""
    future_payload = {
        "sub": "user-id",
        "role": "user",
        "aud": "authenticated",
        "iat": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token = create_token(future_payload)

    # PyJWT validates iat claim and should reject future tokens
    with pytest.raises(ValueError, match="not yet valid"):
        jwt_utils.parse_jwt_token(token)


# =============== SYMMETRIC (HS*) TOKENS ARE REJECTED =============== #


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_parse_jwt_token_symmetric_rejected(algorithm: str):
    """Symmetric tokens are refused outright.

    The platform auth service only issues asymmetric (ES256) tokens, so the
    HS* path is never verified, regardless of configuration.
    """
    payload = {"sub": "user-id", "role": "admin", "aud": "authenticated"}
    secret = "shared-secret-at-least-64-bytes-long-for-hs512-hmac-key-padding!!"
    token = jwt.encode(payload, secret, algorithm=algorithm)

    with pytest.raises(ValueError, match="symmetric tokens are not accepted"):
        jwt_utils.parse_jwt_token(token)


def test_parse_jwt_token_none_algorithm_rejected():
    """The unsigned 'none' algorithm must never verify."""
    payload = {"sub": "user-id", "role": "admin", "aud": "authenticated"}
    token = jwt.encode(payload, "", algorithm="none")

    with pytest.raises(ValueError, match="Invalid token"):
        jwt_utils.parse_jwt_token(token)


def _token_with_header(header: dict) -> str:
    """Build a syntactically valid JWT with an arbitrary (unsigned) header.

    jwt.encode() always writes a string `alg`, so malformed headers have to be
    assembled by hand. The signature is garbage: these tokens must be rejected
    before verification is ever attempted.
    """
    segments = [
        json.dumps(header).encode(),
        json.dumps(TEST_USER_PAYLOAD).encode(),
        b"not-a-signature",
    ]
    return ".".join(base64.urlsafe_b64encode(s).rstrip(b"=").decode() for s in segments)


@pytest.mark.parametrize(
    "header",
    [
        pytest.param({"alg": None, "kid": _SIGNING_KID}, id="null-alg"),
        pytest.param({"alg": 256, "kid": _SIGNING_KID}, id="numeric-alg"),
        pytest.param({"kid": _SIGNING_KID}, id="missing-alg"),
        pytest.param({"alg": "none", "kid": _SIGNING_KID}, id="none-alg"),
        pytest.param({"alg": "ES384", "kid": _SIGNING_KID}, id="unlisted-alg"),
    ],
)
def test_parse_jwt_token_bad_algorithm_rejected_before_jwks(
    mocker: MockerFixture, header: dict
):
    """A non-string, missing or unlisted `alg` fails as an auth error (401 at
    the HTTP layer), not an AttributeError → 500, and never reaches the JWK
    set — so junk headers can't trigger a JWKS fetch."""
    jwks_client = mocker.patch.object(jwt_utils, "_get_jwks_client")
    token = _token_with_header(header)

    with pytest.raises(ValueError, match="signing algorithm is not accepted"):
        jwt_utils.parse_jwt_token(token)
    jwks_client.assert_not_called()


# ==================== JWKS (ASYMMETRIC) VERIFICATION ==================== #


def test_parse_jwt_token_unknown_kid_rejected():
    """A token whose kid is absent from the JWK set is rejected."""
    other_key, _ = make_es256_keypair(kid="unknown-key")
    token = create_token(TEST_USER_PAYLOAD, private_key=other_key, kid="unknown-key")

    with pytest.raises(ValueError, match="Invalid token"):
        jwt_utils.parse_jwt_token(token)


@pytest.mark.parametrize("declares_alg", [True, False], ids=["jwk-alg", "no-jwk-alg"])
@pytest.mark.parametrize(
    "algorithm, private_key",
    [
        ("RS256", rsa.generate_private_key(public_exponent=65537, key_size=2048)),
        ("EdDSA", ed25519.Ed25519PrivateKey.generate()),
    ],
    ids=["RS256", "EdDSA"],
)
def test_parse_jwt_token_algorithm_other_than_its_keys_rejected(
    mocker: MockerFixture,
    algorithm: str,
    private_key: rsa.RSAPrivateKey | ed25519.Ed25519PrivateKey,
    declares_alg: bool,
):
    """A token claiming an accepted algorithm its EC key can't serve is an auth
    error, not a 500 from loading the key under that algorithm."""
    if not declares_alg:
        assert _signing_key is not None
        _serve_jwk(mocker, ECAlgorithm.to_jwk(_signing_key.public_key(), as_dict=True))
    token = jwt.encode(
        TEST_USER_PAYLOAD,
        private_key,
        algorithm=algorithm,
        headers={"kid": _SIGNING_KID},
    )

    with pytest.raises(ValueError, match="signing algorithm does not match its key"):
        jwt_utils.parse_jwt_token(token)


def test_parse_jwt_token_payload_not_an_object_rejected():
    """The key lookup parses the payload before verification; a payload that
    isn't a JSON object must fail there as an auth error too."""
    assert _signing_key is not None
    token = jwt.api_jws.encode(
        b"[]", _signing_key, algorithm="ES256", headers={"kid": _SIGNING_KID}
    )

    with pytest.raises(ValueError, match="Invalid token"):
        jwt_utils.parse_jwt_token(token)


@pytest.mark.parametrize("declared_alg", ["ES256", None], ids=["jwk-alg", "no-jwk-alg"])
def test_parse_jwt_token_key_unusable_under_its_algorithm_rejected(
    mocker: MockerFixture, declared_alg: str | None
):
    """An ES256 token checked against a P-384 key, whether or not the JWK labels
    it ES256, fails on the curve as an auth error, not a 500."""
    p384_key = ec.generate_private_key(ec.SECP384R1())
    jwk = ECAlgorithm.to_jwk(p384_key.public_key(), as_dict=True)
    if declared_alg:
        jwk["alg"] = declared_alg
    _serve_jwk(mocker, jwk)

    with pytest.raises(ValueError, match="Invalid token") as exc_info:
        jwt_utils.parse_jwt_token(create_token(TEST_USER_PAYLOAD))
    assert isinstance(exc_info.value.__cause__, jwt.InvalidKeyError)


@pytest.mark.parametrize("algorithm", ["RS512", "PS256"])
def test_parse_jwt_token_rsa_jwk_without_alg_verifies_listed_rsa_algorithms(
    mocker: MockerFixture, algorithm: str
):
    """A JWK's `alg` is optional: an RSA key without one verifies every listed
    RSA algorithm, not only the RS256 PyJWT infers for it."""
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _serve_jwk(
        mocker,
        RSAAlgorithm.to_jwk(rsa_key.public_key(), as_dict=True),
        algorithms="ES256,RS512,PS256",
    )
    token = jwt.encode(
        TEST_USER_PAYLOAD, rsa_key, algorithm=algorithm, headers={"kid": _SIGNING_KID}
    )

    assert jwt_utils.parse_jwt_token(token)["sub"] == "test-user-id"


def test_parse_jwt_token_jwk_alg_pins_its_algorithm(mocker: MockerFixture):
    """A JWK that declares `alg` verifies that algorithm only, though its key
    type could serve others."""
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = RSAAlgorithm.to_jwk(rsa_key.public_key(), as_dict=True) | {"alg": "RS256"}
    _serve_jwk(mocker, jwk, algorithms="ES256,RS256,RS512")
    token = jwt.encode(
        TEST_USER_PAYLOAD, rsa_key, algorithm="RS512", headers={"kid": _SIGNING_KID}
    )

    with pytest.raises(ValueError, match="signing algorithm does not match its key"):
        jwt_utils.parse_jwt_token(token)


def _serve_jwk(mocker: MockerFixture, jwk: dict, algorithms: str | None = None):
    """Serve `jwk` as the JWK set's only key, under the signing kid, optionally
    with its own `JWT_JWKS_ALGORITHMS`."""
    if algorithms is not None:
        mocker.patch.dict(os.environ, {"JWT_JWKS_ALGORITHMS": algorithms})
        mocker.patch.object(config, "_settings", Settings())
    jwk_set = {"keys": [{**jwk, "kid": _SIGNING_KID}]}
    mocker.patch.object(jwt.PyJWKClient, "fetch_data", return_value=jwk_set)


def test_jwks_client_rekeys_when_url_changes(mocker: MockerFixture):
    """A changed JWT_JWKS_URL must produce a new client, not reuse the old
    one pointed at the previous endpoint."""
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": "http://first/jwks"}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    first = jwt_utils._get_jwks_client()
    assert jwt_utils._get_jwks_client() is first  # same URL -> cached

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": "http://second/jwks"}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    second = jwt_utils._get_jwks_client()

    assert second is not first
    assert second.uri == "http://second/jwks"


def test_verify_user_missing_role_is_not_a_server_error():
    """A token with no `role` claim must resolve as a plain user, not KeyError.

    verify_user already fails closed for admin_only; this covers the ordinary
    path, where indexing payload["role"] would surface as a 500 instead of a
    normal authenticated request.
    """
    no_role_payload = {"sub": "user-id", "email": "user@example.com"}

    user = jwt_utils.verify_user(no_role_payload, admin_only=False)

    assert user.user_id == "user-id"
    assert user.role == "user"
