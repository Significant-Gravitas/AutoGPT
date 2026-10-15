"""Execution-path resolver — drives whether a dream goes batch or sync."""

from __future__ import annotations

import pytest

from .routing import resolve_dream_execution_path


@pytest.mark.parametrize(
    "has_anthropic_key,batch_enabled,expected",
    [
        # Anthropic key + batch on: anthropic_batch.
        (True, True, "anthropic_batch"),
        # Batch flag off — falls back to sync_baseline with or without a key.
        (True, False, "sync_baseline"),
        (False, False, "sync_baseline"),
        # No key — the flag alone cannot open the batch path.
        (False, True, "sync_baseline"),
    ],
)
def test_routing_branches(has_anthropic_key, batch_enabled, expected):
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=has_anthropic_key,
            batch_processing_enabled=batch_enabled,
        )
        == expected
    )


@pytest.mark.parametrize("transport_name", ["local", "subscription"])
def test_local_and_subscription_transports_force_sync_baseline(transport_name):
    """Local installs have no batch endpoint; subscription users
    shouldn't dual-bill an unrelated ``ANTHROPIC_API_KEY`` for the
    dream pass when the chat layer is on Claude Code OAuth.

    Both transports veto the batch path even when
    ``batch_processing_enabled=True`` and ``has_anthropic_key=True``."""
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=True,
            batch_processing_enabled=True,
            transport_name=transport_name,
        )
        == "sync_baseline"
    )


@pytest.mark.parametrize("transport_name", ["openrouter", "direct_anthropic", None])
def test_batch_eligible_transports_unaffected(transport_name):
    """openrouter / direct_anthropic / ``None`` (no override) preserve
    the historical key-driven behaviour — the ``transport_name``
    gate only fires for ``local`` and ``subscription``."""
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=True,
            batch_processing_enabled=True,
            transport_name=transport_name,
        )
        == "anthropic_batch"
    )
