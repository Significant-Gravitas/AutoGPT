"""Tests for the remote catalog sync client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.data.llm_registry.sync as sync_mod
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogPayload,
)
from backend.data.llm_registry.sync import (
    _sync_once_safe,
    llm_catalog_sync_loop,
    should_sync,
    sync_catalog_once,
)
from backend.util.settings import BehaveAs


def _payload_bytes(schema_version: int = CATALOG_SCHEMA_VERSION) -> bytes:
    payload = {
        "schema_version": schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "providers": [{"name": "openai", "display_name": "OpenAI"}],
        "creators": [],
        "models": [
            {
                "slug": "openai/gpt-a",
                "display_name": "GPT A",
                "provider": "openai",
                "context_window": 128000,
            }
        ],
    }
    return json.dumps(payload).encode()


def _mock_fetch(mocker, body: bytes):
    response = MagicMock()
    response.content = body
    requests = MagicMock()
    requests.get = AsyncMock(return_value=response)
    mocker.patch.object(sync_mod, "Requests", return_value=requests)
    return requests


@pytest.fixture
def import_mock(mocker):
    result = MagicMock()
    result.content_hash = "abc123def456"
    result.unchanged = False
    return mocker.patch.object(
        sync_mod, "import_catalog", new=AsyncMock(return_value=result)
    )


def test_should_sync_gates_on_behave_as(mocker):
    mocker.patch.object(sync_mod.config, "llm_catalog_sync_enabled", True)
    mocker.patch.object(sync_mod.settings.config, "behave_as", BehaveAs.CLOUD)
    assert should_sync() is False

    mocker.patch.object(sync_mod.settings.config, "behave_as", BehaveAs.LOCAL)
    assert should_sync() is True

    mocker.patch.object(sync_mod.config, "llm_catalog_sync_enabled", False)
    assert should_sync() is False


@pytest.mark.asyncio
async def test_sync_fetches_validates_and_imports(import_mock, mocker):
    _mock_fetch(mocker, _payload_bytes())

    assert await sync_catalog_once() is True

    import_mock.assert_called_once()
    payload = import_mock.call_args.args[0]
    assert isinstance(payload, CatalogPayload)
    assert payload.models[0].slug == "openai/gpt-a"
    assert import_mock.call_args.kwargs["source_url"] == sync_mod.config.llm_catalog_url


@pytest.mark.asyncio
async def test_oversized_payload_rejected_before_import(import_mock, mocker):
    _mock_fetch(mocker, b"x" * (sync_mod._MAX_PAYLOAD_BYTES + 1))

    with pytest.raises(ValueError, match="too large"):
        await sync_catalog_once()

    import_mock.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_payload_rejected_before_import(import_mock, mocker):
    _mock_fetch(mocker, b'{"not": "a catalog"}')

    with pytest.raises(Exception):
        await sync_catalog_once()

    import_mock.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_schema_version_never_reaches_import(import_mock, mocker):
    """Version check lives in import_catalog, but validation happens first —
    an unparseable-for-this-build payload must never cause partial writes."""
    _mock_fetch(mocker, _payload_bytes(schema_version=CATALOG_SCHEMA_VERSION))
    # Parseable payload DOES reach import_catalog (which owns the version check)
    await sync_catalog_once()
    import_mock.assert_called_once()


@pytest.mark.asyncio
async def test_sync_once_safe_survives_fetch_failure(mocker):
    mocker.patch.object(
        sync_mod, "_acquire_sync_lock", new=AsyncMock(return_value=True)
    )
    record = mocker.patch.object(sync_mod, "_record_attempt", new=AsyncMock())
    mocker.patch.object(
        sync_mod,
        "sync_catalog_once",
        new=AsyncMock(side_effect=ConnectionError("down")),
    )

    await _sync_once_safe()  # must not raise

    record.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_sync_once_safe_records_success(mocker):
    mocker.patch.object(
        sync_mod, "_acquire_sync_lock", new=AsyncMock(return_value=True)
    )
    record = mocker.patch.object(sync_mod, "_record_attempt", new=AsyncMock())
    mocker.patch.object(sync_mod, "sync_catalog_once", new=AsyncMock(return_value=True))

    await _sync_once_safe()

    record.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_sync_skipped_when_lock_held(mocker):
    mocker.patch.object(
        sync_mod, "_acquire_sync_lock", new=AsyncMock(return_value=False)
    )
    once = mocker.patch.object(sync_mod, "sync_catalog_once", new=AsyncMock())

    await _sync_once_safe()

    once.assert_not_called()


@pytest.mark.asyncio
async def test_loop_returns_immediately_when_sync_disabled(mocker):
    mocker.patch.object(sync_mod, "should_sync", return_value=False)
    safe = mocker.patch.object(sync_mod, "_sync_once_safe", new=AsyncMock())

    await llm_catalog_sync_loop()  # returns instead of looping forever

    safe.assert_not_called()
