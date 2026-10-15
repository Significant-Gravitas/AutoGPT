"""Cancelling a dream pass's batch at Anthropic: the SDK call it makes, and
that nothing about it (no key, a refusal, a hang) reaches the stop that asked
for it."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from . import provider_batch
from .provider_batch import anthropic_api_key, cancel_provider_batch


@pytest.fixture
def client(mocker) -> MagicMock:
    """The Anthropic client ``cancel_batch`` builds, with the key it gets."""
    mocker.patch.object(provider_batch, "anthropic_api_key", return_value="sk-test")
    anthropic_client = MagicMock()
    anthropic_client.messages.batches.cancel = AsyncMock()
    mocker.patch(
        "backend.util.llm.providers.anthropic.AsyncAnthropic",
        return_value=anthropic_client,
    )
    return anthropic_client


async def test_the_batch_is_cancelled_through_the_anthropic_client(client):
    await cancel_provider_batch("msgbatch_1")

    client.messages.batches.cancel.assert_awaited_once_with("msgbatch_1")


async def test_a_refused_cancel_is_logged_not_raised(client, caplog):
    client.messages.batches.cancel.side_effect = RuntimeError("batch has ended")

    with caplog.at_level(logging.WARNING, logger=provider_batch.logger.name):
        await cancel_provider_batch("msgbatch_1")

    assert "did not cancel dream batch msgbatch_1" in caplog.text


async def test_a_cancel_that_hangs_is_abandoned_at_its_deadline(
    client, monkeypatch, caplog
):
    monkeypatch.setattr(provider_batch, "PROVIDER_CANCEL_TIMEOUT_SECONDS", 0.05)

    async def hang(*_args, **_kwargs) -> None:
        await asyncio.Event().wait()

    client.messages.batches.cancel.side_effect = hang

    with caplog.at_level(logging.WARNING, logger=provider_batch.logger.name):
        await asyncio.wait_for(cancel_provider_batch("msgbatch_1"), 5)

    assert "Cancelling dream batch msgbatch_1 failed" in caplog.text


async def test_without_a_key_nothing_is_sent(mocker, caplog):
    mocker.patch.object(provider_batch, "anthropic_api_key", return_value=None)
    anthropic_cls = mocker.patch("backend.util.llm.providers.anthropic.AsyncAnthropic")

    with caplog.at_level(logging.WARNING, logger=provider_batch.logger.name):
        await cancel_provider_batch("msgbatch_1")

    anthropic_cls.assert_not_called()
    assert "No Anthropic key" in caplog.text


def test_the_key_is_the_copilot_configs_before_the_shared_one(mocker):
    config = mocker.patch.object(provider_batch, "ChatConfig")
    config.return_value.direct_anthropic_api_key = "sk-config"
    settings = mocker.patch.object(provider_batch, "Settings")
    settings.return_value.secrets.anthropic_api_key = "sk-shared"

    assert anthropic_api_key() == "sk-config"
    config.return_value.direct_anthropic_api_key = None
    assert anthropic_api_key() == "sk-shared"
    settings.return_value.secrets.anthropic_api_key = ""
    assert anthropic_api_key() is None
