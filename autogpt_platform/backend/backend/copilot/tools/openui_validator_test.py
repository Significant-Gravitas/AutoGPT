import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.tools import openui_validator


@pytest.mark.asyncio
async def test_real_validator_accepts_literal_data_and_rejects_bad_rows():
    valid = await openui_validator.validate_openui_source(
        'root = Workspace("Quoted $(data)", "Not executable", '
        '[DataTable("Quotes", ["Name", "Cost"], [["Kite", "$1550"]])])'
    )
    assert valid.valid
    invalid = await openui_validator.validate_openui_source(
        'root = Workspace("Quotes", "Data", '
        '[DataTable("Quotes", ["Name", "Cost"], [["Kite"]])])'
    )
    assert not invalid.valid
    assert "DataTable.rows.0" in invalid.error


@pytest.mark.asyncio
async def test_real_validator_checks_connected_fields_and_nullable_bounds():
    source = (
        'root = Workspace("Estimate", "Supplied data", [brief, total])\n'
        'brief = Form("plan", "Adjust", [NumberField("days", "Days", '
        '2, null, null, 1)], "Continue", "Use these values")\n'
        'total = CalculatedMetric("plan", "Total", "days * 25 + 10", '
        '"currency", "USD", 2)'
    )
    assert (await openui_validator.validate_openui_source(source)).valid
    invalid = await openui_validator.validate_openui_source(
        source.replace("days * 25 + 10", "missing * 25 + 10")
    )
    assert not invalid.valid
    assert "missing" in invalid.error


@pytest.mark.asyncio
async def test_missing_runtime_is_unavailable(monkeypatch):
    monkeypatch.setattr(openui_validator.shutil, "which", lambda _: None)
    with pytest.raises(openui_validator.ValidatorUnavailable):
        await openui_validator.validate_openui_source("unused")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, asyncio.CancelledError])
async def test_timeout_and_cancellation_reap_the_process(monkeypatch, error):
    process = MagicMock(returncode=None)
    process.communicate = AsyncMock(side_effect=[error(), (b"", b"")])
    monkeypatch.setattr(
        openui_validator.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    expected = (
        asyncio.CancelledError
        if error is asyncio.CancelledError
        else openui_validator.ValidatorUnavailable
    )
    with pytest.raises(expected):
        await openui_validator.validate_openui_source("unused")
    process.kill.assert_called_once()
    assert process.communicate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, output", [(1, b""), (0, b"not json"), (0, b"{}"), (0, b"x" * 8193)]
)
async def test_bad_validator_responses_fail_closed(monkeypatch, code, output):
    process = MagicMock(returncode=code)
    process.communicate = AsyncMock(return_value=(output, b"private internal details"))
    monkeypatch.setattr(
        openui_validator.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    with pytest.raises(openui_validator.ValidatorUnavailable):
        await openui_validator.validate_openui_source("unused")


@pytest.mark.asyncio
async def test_source_only_enters_stdin_and_node_options_are_not_inherited(monkeypatch):
    process = MagicMock(returncode=0)
    process.communicate = AsyncMock(return_value=(b'{"valid":true,"error":""}', b""))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(openui_validator.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setenv("NODE_OPTIONS", "--require untrusted.js")
    source = 'root = Workspace("$(touch /tmp/no)", "literal", [])'
    assert (await openui_validator.validate_openui_source(source)).valid
    assert source not in spawn.call_args.args
    assert "NODE_OPTIONS" not in spawn.call_args.kwargs["env"]
    assert process.communicate.call_args.args == (source.encode(),)
