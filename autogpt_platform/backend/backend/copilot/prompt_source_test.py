"""Tests for ``system_prompt_source``: which base prompt a turn's system prompt used."""

from unittest.mock import MagicMock

import pytest

from backend.copilot import service
from backend.copilot.service import (
    _CACHEABLE_SYSTEM_PROMPT,
    _revalidate_prompt,
    config,
    system_prompt_source,
)


@pytest.fixture(autouse=True)
def cold_prompt_cache(monkeypatch):
    monkeypatch.setattr(service, "_cached_prompt", None)
    monkeypatch.setattr(service, "_cached_prompt_identity", None)


def _langfuse_serving(text: str, version: object) -> MagicMock:
    prompt = MagicMock(version=version)
    prompt.compile.return_value = text
    client = MagicMock()
    client.get_prompt.return_value = prompt
    return client


def test_in_code_prompt_reads_as_fallback():
    assert system_prompt_source(_CACHEABLE_SYSTEM_PROMPT + "\nsupplement") == (
        "fallback",
        None,
    )


@pytest.mark.asyncio
async def test_langfuse_prompt_reads_as_its_name_and_version(monkeypatch):
    monkeypatch.setattr(
        service, "_get_langfuse", lambda: _langfuse_serving("v7 prompt", 7)
    )

    await _revalidate_prompt()

    assert system_prompt_source("v7 prompt\nsupplement") == (
        config.langfuse_prompt_name,
        7,
    )


@pytest.mark.asyncio
async def test_a_prompt_this_process_no_longer_holds_has_no_version(monkeypatch):
    monkeypatch.setattr(
        service, "_get_langfuse", lambda: _langfuse_serving("v8 prompt", 8)
    )

    await _revalidate_prompt()

    assert system_prompt_source("v7 prompt\nsupplement") == (
        config.langfuse_prompt_name,
        None,
    )


@pytest.mark.asyncio
async def test_a_non_integer_version_is_dropped(monkeypatch):
    monkeypatch.setattr(
        service, "_get_langfuse", lambda: _langfuse_serving("vx prompt", "7")
    )

    await _revalidate_prompt()

    assert system_prompt_source("vx prompt") == (config.langfuse_prompt_name, None)
