"""
PKCE (Proof Key for Code Exchange) implementation for OAuth 2.0.

RFC 7636: https://tools.ietf.org/html/rfc7636
"""

import base64
import hashlib
import secrets


def generate_code_verifier(length: int = 64) -> str:
    """
    Generate a cryptographically random code verifier.

    Args:
        length: Length of the verifier (43-128 characters, default 64)

    Returns:
        URL-safe base64 encoded random string
    """
    if not 43 <= length <= 128:
        raise ValueError("Code verifier length must be between 43 and 128")
    return secrets.token_urlsafe(length)[:length]


def generate_code_challenge(verifier: str, method: str = "S256") -> str:
    """
    Generate a code challenge from the verifier.

    Args:
        verifier: The code verifier string
        method: Challenge method ("S256" or "plain")

    Returns:
        The code challenge string
    """
    if method == "S256":
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        # URL-safe base64 encoding without padding
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    elif method == "plain":
        return verifier
    else:
        raise ValueError(f"Unsupported code challenge method: {method}")


def verify_code_challenge(
    verifier: str,
    challenge: str,
    method: str = "S256",
) -> bool:
    """
    Verify that a code verifier matches the stored challenge.

    Args:
        verifier: The code verifier from the token request
        challenge: The code challenge stored during authorization
        method: The challenge method used

    Returns:
        True if the verifier matches the challenge
    """
    expected = generate_code_challenge(verifier, method)
    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(expected, challenge)
