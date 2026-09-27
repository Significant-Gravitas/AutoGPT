"""Cancelling a dream pass's batch at Anthropic: the SDK call it makes, that
nothing about it (no key, a refusal, a hang) reaches the stop that asked for
it, and when the cleanup after a pass takes its batch to have stopped."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from . import provider_batch
from .provider_batch import (
    anthropic_api_key,
    cancel_provider_batch,
    provider_batch_stopped,
)


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
    assert await cancel_provider_batch("msgbatch_1") is True

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


class TestWhetherABatchHasStopped:
    """What the cleanup after a pass asks: its batch has stopped once the
    cancel is acknowledged, or the batch is found ended; anything less is no
    stop, and the cleanup stays unfinished."""

    async def test_an_acknowledged_cancel_is_a_stop(self, client, batch_status):
        assert await provider_batch_stopped("msgbatch_1") is True

        batch_status.assert_not_awaited()

    @pytest.mark.parametrize(
        ("status", "stopped"), [("ended", True), ("processing", False)]
    )
    async def test_a_refused_cancel_asks_whether_the_batch_ended(
        self, client, batch_status, status, stopped
    ):
        client.messages.batches.cancel.side_effect = RuntimeError("batch has ended")
        batch_status.return_value = status

        assert await provider_batch_stopped("msgbatch_1") is stopped

        batch_status.assert_awaited_once()
        assert batch_status.await_args.kwargs["provider_batch_id"] == "msgbatch_1"

    async def test_a_status_the_provider_cannot_give_is_no_stop(
        self, client, batch_status, caplog
    ):
        client.messages.batches.cancel.side_effect = RuntimeError("unreachable")
        batch_status.side_effect = ConnectionError("unreachable")

        with caplog.at_level(logging.WARNING, logger=provider_batch.logger.name):
            assert await provider_batch_stopped("msgbatch_1") is False

        assert "Could not read the status of dream batch msgbatch_1" in caplog.text

    async def test_without_a_key_nothing_is_confirmed(self, mocker, batch_status):
        mocker.patch.object(provider_batch, "anthropic_api_key", return_value=None)

        assert await provider_batch_stopped("msgbatch_1") is False

        batch_status.assert_not_awaited()
