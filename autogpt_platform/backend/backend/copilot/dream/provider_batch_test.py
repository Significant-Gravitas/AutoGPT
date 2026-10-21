"""Cancelling a dream pass's batch at Anthropic: the SDK call it makes, that
nothing about it (no key, a refusal, a hang) reaches the stop that asked for
it, when the cleanup after a pass takes its batch to have stopped, and that
a cancel sent without an acknowledgement is not sent again before the
batch's status is read."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from . import provider_batch
from .provider_batch import (
    CANCEL_SENT_PREFIX,
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


class TestACancelThatWasSent:
    """A cancel and a status read may not fit one reaper attempt together:
    a cancel not acknowledged is noted, so the next stop of the batch reads
    its status first, and a status that reads running sends it again."""

    async def test_is_not_sent_again_before_the_status_is_read(
        self, client, batch_status, fake_dream_redis
    ):
        """The reaper's attempt is cut during the status read: the next stop
        of the batch goes straight to the read."""
        client.messages.batches.cancel.side_effect = RuntimeError("batch has ended")
        reading = asyncio.Event()

        async def cut_during_the_read(**_kwargs) -> str:
            reading.set()
            await asyncio.Event().wait()
            return "ended"

        batch_status.side_effect = cut_during_the_read
        stopping = asyncio.ensure_future(provider_batch_stopped("msgbatch_1"))
        await asyncio.wait_for(reading.wait(), 5)
        stopping.cancel()
        await asyncio.wait([stopping])
        assert f"{CANCEL_SENT_PREFIX}:msgbatch_1" in fake_dream_redis.store

        batch_status.side_effect = None
        batch_status.return_value = "ended"
        assert await provider_batch_stopped("msgbatch_1") is True
        client.messages.batches.cancel.assert_awaited_once()

    async def test_is_sent_again_after_a_status_read_that_failed(
        self, client, batch_status, fake_dream_redis
    ):
        client.messages.batches.cancel.side_effect = RuntimeError("unreachable")
        batch_status.side_effect = ConnectionError("unreachable")

        assert await provider_batch_stopped("msgbatch_1") is False
        assert f"{CANCEL_SENT_PREFIX}:msgbatch_1" not in fake_dream_redis.store

        client.messages.batches.cancel.side_effect = None
        assert await provider_batch_stopped("msgbatch_1") is True
        assert client.messages.batches.cancel.await_count == 2

    async def test_is_sent_again_once_the_batch_reads_still_running(
        self, client, batch_status, fake_dream_redis
    ):
        client.messages.batches.cancel.side_effect = RuntimeError("unreachable")
        fake_dream_redis.store[f"{CANCEL_SENT_PREFIX}:msgbatch_1"] = "1"
        batch_status.return_value = "processing"

        assert await provider_batch_stopped("msgbatch_1") is False
        client.messages.batches.cancel.assert_not_awaited()
        assert f"{CANCEL_SENT_PREFIX}:msgbatch_1" not in fake_dream_redis.store

        client.messages.batches.cancel.side_effect = None
        assert await provider_batch_stopped("msgbatch_1") is True
        client.messages.batches.cancel.assert_awaited_once()

    async def test_acknowledged_leaves_no_note(
        self, client, batch_status, fake_dream_redis
    ):
        assert await provider_batch_stopped("msgbatch_1") is True

        assert not [
            k for k in fake_dream_redis.store if k.startswith(CANCEL_SENT_PREFIX)
        ]

    async def test_without_redis_to_say_is_sent_each_time(
        self, client, batch_status, monkeypatch, fake_dream_redis
    ):
        client.messages.batches.cancel.side_effect = RuntimeError("batch has ended")
        batch_status.return_value = "processing"

        async def down(*_args, **_kwargs):
            raise ConnectionError("redis down")

        monkeypatch.setattr(fake_dream_redis, "exists", down)
        monkeypatch.setattr(fake_dream_redis, "set", down)

        assert await provider_batch_stopped("msgbatch_1") is False
        assert await provider_batch_stopped("msgbatch_1") is False
        assert client.messages.batches.cancel.await_count == 2
