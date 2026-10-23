"""OIDC token verification against an issuer served from memory."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from backend.data.auth import oidc
from backend.data.auth.oidc import OIDCVerificationError, verify_oidc_token

ISSUER = "https://token.example.com"
AUDIENCE = "https://backend.agpt.co"
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeIssuer:
    """Serves a discovery document and JWKS, and counts what was fetched."""

    def __init__(self, keys: list[dict]):
        self.keys = keys
        self.discovery = {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> dict:
        self.fetched.append(url)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return self.discovery
        if url == f"{ISSUER}/jwks":
            return {"keys": self.keys}
        raise ValueError(f"unexpected fetch of {url}")


@pytest.fixture
def issuer(mocker) -> FakeIssuer:
    oidc._fetch_issuer_keys.cache_clear()
    oidc._forced_refresh_at.clear()
    oidc._failed_fetch_at.clear()
    fake = FakeIssuer([_jwk(_KEY, "key-1")])
    mocker.patch.object(oidc, "_fetch_json", side_effect=fake.fetch)
    return fake


@pytest.mark.asyncio
async def test_a_valid_token_yields_its_claims(issuer):
    claims = await _verify(_token(sub="repo:acme/api:ref:refs/heads/main"))

    assert claims["sub"] == "repo:acme/api:ref:refs/heads/main"
    assert issuer.fetched == [
        f"{ISSUER}/.well-known/openid-configuration",
        f"{ISSUER}/jwks",
    ]


@pytest.mark.asyncio
async def test_signing_keys_are_cached(issuer):
    await _verify(_token())
    await _verify(_token())

    assert len(issuer.fetched) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"aud": "https://someone-else.example"}, "audience"),
        ({"iss": "https://attacker.example"}, "issuer"),
        ({"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200}, "expired"),
        ({"sub": None}, "sub"),
    ],
)
async def test_a_token_for_the_wrong_party_or_time_is_refused(
    issuer, overrides, reason
):
    with pytest.raises(OIDCVerificationError):
        await _verify(_token(**overrides))
    assert all(url.startswith(ISSUER) for url in issuer.fetched), reason


@pytest.mark.asyncio
async def test_a_token_signed_by_another_key_is_refused(issuer):
    with pytest.raises(OIDCVerificationError):
        await _verify(_token(key=_OTHER_KEY))


@pytest.mark.asyncio
async def test_a_long_lived_token_is_refused(issuer):
    now = int(time.time())
    with pytest.raises(OIDCVerificationError, match="24 hours"):
        await _verify(_token(iat=now, exp=now + 25 * 3600))


@pytest.mark.asyncio
async def test_symmetric_and_unsigned_tokens_are_refused_before_any_fetch(issuer):
    hs256 = jwt.encode(_claims(), "a-shared-secret-of-at-least-32-bytes", "HS256")
    unsigned = jwt.encode(_claims(), None, algorithm="none")

    for token in (hs256, unsigned):
        with pytest.raises(OIDCVerificationError, match="not accepted"):
            await _verify(token)
    assert issuer.fetched == []


@pytest.mark.asyncio
async def test_a_symmetric_key_in_the_jwks_is_never_used(issuer):
    issuer.keys = [{"kty": "oct", "kid": "key-1", "k": "c2hhcmVkLXNlY3JldA"}]

    with pytest.raises(OIDCVerificationError):
        await _verify(_token())


@pytest.mark.asyncio
async def test_a_rotated_key_is_picked_up_by_one_refetch(issuer):
    await _verify(_token())
    issuer.keys = [_jwk(_KEY, "key-1"), _jwk(_OTHER_KEY, "key-2")]

    claims = await _verify(_token(key=_OTHER_KEY, kid="key-2"))

    assert claims["sub"] == "workload"
    assert len(issuer.fetched) == 4


@pytest.mark.asyncio
async def test_unknown_key_ids_cannot_force_a_refetch_per_request(issuer):
    await _verify(_token())
    for kid in ("bogus-1", "bogus-2", "bogus-3"):
        with pytest.raises(OIDCVerificationError):
            await _verify(_token(kid=kid))

    assert len(issuer.fetched) == 4


@pytest.mark.asyncio
async def test_a_discovery_document_for_another_issuer_is_refused(issuer):
    issuer.discovery["issuer"] = "https://attacker.example"

    with pytest.raises(OIDCVerificationError, match="Could not fetch"):
        await _verify(_token())


@pytest.mark.asyncio
async def test_a_plain_http_jwks_uri_is_refused(issuer):
    issuer.discovery["jwks_uri"] = "http://token.example.com/jwks"

    with pytest.raises(OIDCVerificationError, match="Could not fetch"):
        await _verify(_token())


@pytest.mark.asyncio
async def test_a_failed_fetch_is_not_retried_immediately(issuer):
    issuer.discovery["issuer"] = "https://attacker.example"
    for _ in range(3):
        with pytest.raises(OIDCVerificationError):
            await _verify(_token())

    assert len(issuer.fetched) == 1


@pytest.mark.asyncio
async def test_an_ec_signed_token_is_accepted(issuer):
    ec_key = ec.generate_private_key(ec.SECP256R1())
    issuer.keys = [_jwk(ec_key, "ec-1", alg="ES256")]

    claims = await _verify(_token(key=ec_key, kid="ec-1", alg="ES256"))

    assert claims["sub"] == "workload"


async def _verify(token: str) -> dict:
    return await verify_oidc_token(token, issuer=ISSUER, audience=AUDIENCE)


def _claims(**overrides) -> dict:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "workload",
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def _token(key=_KEY, kid: str = "key-1", alg: str = "RS256", **overrides) -> str:
    return jwt.encode(_claims(**overrides), key, algorithm=alg, headers={"kid": kid})


def _jwk(private_key, kid: str, alg: str = "RS256") -> dict:
    if isinstance(private_key, rsa.RSAPrivateKey):
        jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    else:
        jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "alg": alg, "use": "sig"}
