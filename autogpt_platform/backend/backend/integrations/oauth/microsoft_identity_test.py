import time
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft_identity import (
    PERSONAL_TENANT_ID,
    populate_microsoft_identity,
)


@pytest.fixture
def signing_key(mocker):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client = Mock()
    client.get_signing_key_from_jwt.return_value.key = key.public_key()
    mocker.patch(
        "backend.integrations.oauth.microsoft_identity._signing_keys",
        return_value=client,
    )
    return key


def claims(**updates):
    return {
        "aud": "shared-client",
        "iss": f"https://login.microsoftonline.com/{PERSONAL_TENANT_ID}/v2.0",
        "tid": PERSONAL_TENANT_ID,
        "sub": "account-subject",
        "oid": "stable-account",
        "exp": int(time.time()) + 300,
        "preferred_username": "same@company.example",
        **updates,
    }


@pytest.mark.asyncio
async def test_signed_identity_classifies_personal_custom_domain(signing_key):
    credential = OAuth2Credentials(
        provider="microsoft", access_token="opaque", scopes=[]
    )
    await populate_microsoft_identity(
        credential,
        jwt.encode(claims(), signing_key, algorithm="RS256"),
        "shared-client",
    )
    assert credential.username == "same@company.example"
    assert credential.metadata["microsoft_identity"]["account_type"] == "personal"


@pytest.mark.parametrize(
    "updates",
    [{"aud": "another-app"}, {"iss": "https://attacker.example"}, {"exp": 1}],
)
@pytest.mark.asyncio
async def test_untrusted_identity_cannot_classify_or_merge_accounts(
    signing_key, updates
):
    credential = OAuth2Credentials(
        provider="microsoft", access_token="opaque", scopes=[]
    )
    with pytest.raises(jwt.InvalidTokenError):
        await populate_microsoft_identity(
            credential,
            jwt.encode(claims(**updates), signing_key, algorithm="RS256"),
            "shared-client",
        )
    assert credential.metadata == {}
