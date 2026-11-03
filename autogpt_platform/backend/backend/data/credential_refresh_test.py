from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.data.user import get_near_expiry_oauth_credentials
from backend.util.encryption import JSONCryptor


@pytest.mark.parametrize(
    "access_expiry,refresh_expiry,eligible",
    [
        (None, None, False),
        (790_399, None, True),
        (790_400, None, True),
        (790_401, None, False),
        (1_999_999, None, False),
        (2_000_000, None, False),
        (2_000_100, None, False),
        (790_399, 1_999_999, False),
        (790_399, 2_000_000, False),
        (790_399, 2_000_001, True),
    ],
)
async def test_refresh_candidate_expiry_boundaries(
    mocker, access_expiry, refresh_expiry, eligible
):
    credentials = OAuth2Credentials(
        id="credential",
        provider="github",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        access_token_expires_at=access_expiry,
        refresh_token_expires_at=refresh_expiry,
        scopes=[],
    )
    row = MagicMock(
        id=credentials.id,
        ownerId="owner",
        encryptedPayload=JSONCryptor().encrypt(credentials.model_dump()),
    )
    database = mocker.patch("backend.data.user.prisma")
    database.integrationcredential.find_many = AsyncMock(return_value=[row])
    mocker.patch("time.time", return_value=2_000_000)

    result = await get_near_expiry_oauth_credentials(2_000_000 - 14 * 86_400)

    assert result == ([("owner", credentials)] if eligible else [])


async def test_corrupt_candidate_does_not_log_decrypted_secrets(mocker, caplog):
    secret = "do-not-log-this-access-token"
    row = MagicMock(
        id="corrupt",
        ownerId="owner",
        encryptedPayload=JSONCryptor().encrypt(
            {"access_token": secret, "type": "unsupported"}
        ),
    )
    database = mocker.patch("backend.data.user.prisma")
    database.integrationcredential.find_many = AsyncMock(return_value=[row])

    assert await get_near_expiry_oauth_credentials(1_300) == []
    assert "Corrupt credential row corrupt" in caplog.text
    assert secret not in caplog.text
