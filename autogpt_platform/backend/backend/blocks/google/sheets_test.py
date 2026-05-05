"""Edge-case tests for Google Sheets block credential handling.

These pin the contract for the systemic auto-credential None-guard in
``Block._execute()``: any block with an auto-credential field (via
``GoogleDriveFileField`` etc.) that's called without resolved
credentials must surface a clean, user-facing ``BlockExecutionError``
— never a wrapped ``TypeError`` (missing required kwarg) or
``AttributeError`` deep in the provider SDK.
"""

import pytest

from backend.blocks.google.sheets import GoogleSheetsReadBlock
from backend.util.exceptions import BlockExecutionError


@pytest.mark.asyncio
async def test_sheets_read_missing_credentials_yields_clean_error():
    """Valid spreadsheet but no resolved credentials -> the systemic
    None-guard in ``Block._execute()`` yields a ``Missing credentials``
    error before ``run()`` is entered."""
    block = GoogleSheetsReadBlock()
    input_data = {
        "spreadsheet": {
            "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "name": "Test Spreadsheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        "range": "Sheet1!A1:B2",
    }

    with pytest.raises(BlockExecutionError, match="Missing credentials"):
        async for _ in block.execute(input_data):
            pass


@pytest.mark.asyncio
async def test_sheets_read_no_spreadsheet_still_hits_credentials_guard():
    """When neither spreadsheet nor credentials are present, the
    credentials guard fires first (it runs before we hand off to
    ``run()``). The user-facing message should still be the clean
    ``Missing credentials`` one, not an opaque ``TypeError``."""
    block = GoogleSheetsReadBlock()
    input_data = {"range": "Sheet1!A1:B2"}  # no spreadsheet, no credentials

    with pytest.raises(BlockExecutionError, match="Missing credentials"):
        async for _ in block.execute(input_data):
            pass


@pytest.mark.asyncio
async def test_sheets_read_upstream_chained_value_skips_guard(mocker):
    """A spreadsheet value chained in from an upstream input block (e.g.
    ``AgentGoogleDriveFileInputBlock``) carries a resolved
    ``_credentials_id`` that ``_acquire_auto_credentials`` didn't have
    visibility into at prep time. The systemic None-guard must NOT
    preempt run() in that case — otherwise every chained Drive-picker
    pattern crashes with a bogus ``Missing credentials`` error.

    We short-circuit past the guard by patching the Google API client
    build; any error that escapes from run() is fine as long as the
    ``Missing credentials`` message never surfaces."""
    # Patch out the real Google Sheets client build so we don't hit the
    # network and can detect we reached the provider SDK.
    mocker.patch(
        "backend.blocks.google.sheets.build",
        side_effect=RuntimeError("api-boundary-reached"),
    )

    block = GoogleSheetsReadBlock()
    input_data = {
        "spreadsheet": {
            "_credentials_id": "upstream-chained-cred-id",
            "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "name": "Upstream-chained sheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        "range": "Sheet1!A1:B2",
    }

    with pytest.raises(Exception) as exc_info:
        async for _ in block.execute(input_data):
            pass

    # The guard should skip (chained data present) and let us reach run(),
    # which then hits the patched provider-SDK boundary. A "Missing
    # credentials" error here would mean the None-guard broke the
    # documented AgentGoogleDriveFileInputBlock chaining pattern.
    assert "Missing credentials" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_sheets_read_upstream_chained_with_explicit_none_cred_id_skips_guard(
    mocker,
):
    """Sentry HIGH regression (thread PRRT_kwDOJKSTjM58sJfA): the
    documented chained-upstream pattern ships the spreadsheet dict with
    ``_credentials_id=None`` — the executor fills in the resolved id
    between prep time and ``run()``. The previous ``_base.py`` guard
    used ``field_value.get("_credentials_id")`` and treated the falsy
    ``None`` value as "missing", raising ``BlockExecutionError`` on
    every chained graph.

    Pin the contract: the presence of the ``_credentials_id`` key — not
    its truthiness — is what signals "trust the skip". A dict with
    ``_credentials_id: None`` must not preempt run()."""
    mocker.patch(
        "backend.blocks.google.sheets.build",
        side_effect=RuntimeError("api-boundary-reached"),
    )

    block = GoogleSheetsReadBlock()
    input_data = {
        "spreadsheet": {
            "_credentials_id": None,  # explicit None — chained-upstream shape
            "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "name": "Upstream-chained sheet (None cred_id)",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        "range": "Sheet1!A1:B2",
    }

    with pytest.raises(Exception) as exc_info:
        async for _ in block.execute(input_data):
            pass

    # The guard must not raise "Missing credentials" for this shape.
    # We expect to reach run() and hit the patched provider-SDK boundary.
    assert "Missing credentials" not in str(exc_info.value)
