"""Tests for the per-workspace bot install store (encryption round-trip)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import bot_installs
from backend.platform_linking.models import Platform

_BI = "backend.data.bot_installs.BotInstall"


@pytest.mark.asyncio
async def test_upsert_encrypts_token_and_get_decrypts_it():
    upsert = AsyncMock()
    with patch(_BI) as model:
        model.prisma.return_value.upsert = upsert
        await bot_installs.upsert_bot_install(
            platform=Platform.SLACK,
            team_id="T1",
            bot_token="xoxb-secret",
            bot_user_id="UBOT",
        )

    data = upsert.await_args.kwargs["data"]
    encrypted = data["create"]["credentials"]
    # The raw token must never be persisted in the clear.
    assert "xoxb-secret" not in encrypted

    row = MagicMock(revokedAt=None, credentials=encrypted, botUserId="UBOT", appId=None)
    row.name = "Acme"  # `name` is a reserved MagicMock kwarg — set it explicitly.
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=row)
        creds = await bot_installs.get_bot_install(Platform.SLACK, "T1")

    assert creds is not None
    assert creds.bot_token == "xoxb-secret"
    assert creds.bot_user_id == "UBOT"
    assert creds.name == "Acme"


@pytest.mark.asyncio
async def test_get_returns_none_for_revoked_install():
    row = MagicMock(revokedAt=object(), credentials="whatever")
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=row)
        assert await bot_installs.get_bot_install(Platform.SLACK, "T1") is None


@pytest.mark.asyncio
async def test_get_returns_none_when_not_installed():
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=None)
        assert await bot_installs.get_bot_install(Platform.SLACK, "T1") is None


@pytest.mark.asyncio
async def test_is_install_revoked_true_for_revoked_row():
    row = MagicMock(revokedAt=object())
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=row)
        assert await bot_installs.is_install_revoked(Platform.SLACK, "T1") is True


@pytest.mark.asyncio
async def test_is_install_revoked_false_when_never_installed():
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=None)
        assert await bot_installs.is_install_revoked(Platform.SLACK, "T1") is False


@pytest.mark.asyncio
async def test_get_returns_none_for_corrupt_credentials():
    # JSONCryptor.decrypt never raises — corrupt ciphertext decrypts to {} and
    # the missing-token path returns None.
    row = MagicMock(revokedAt=None, credentials="not-a-valid-ciphertext")
    with patch(_BI) as model:
        model.prisma.return_value.find_unique = AsyncMock(return_value=row)
        assert await bot_installs.get_bot_install(Platform.SLACK, "T1") is None
