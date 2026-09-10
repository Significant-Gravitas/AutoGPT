"""Execution-path resolver — drives whether a dream goes batch or sync."""

from __future__ import annotations

import pytest

from .routing import resolve_dream_execution_path


@pytest.mark.parametrize(
    "has_anthropic_key,has_openai_key,batch_enabled,expected",
    [
        # Both keys + batch on: Anthropic wins (better cache pricing).
        (True, True, True, "anthropic_batch"),
        # Anthropic only + batch on: anthropic_batch.
        (True, False, True, "anthropic_batch"),
        # OpenAI only + batch on: openai_batch.
        (False, True, True, "openai_batch"),
        # Batch flag off — every combination falls back to sync_baseline.
        (True, True, False, "sync_baseline"),
        (True, False, False, "sync_baseline"),
        (False, True, False, "sync_baseline"),
        (False, False, True, "sync_baseline"),
        (False, False, False, "sync_baseline"),
    ],
)
def test_routing_branches(has_anthropic_key, has_openai_key, batch_enabled, expected):
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=has_anthropic_key,
            has_openai_key=has_openai_key,
            batch_processing_enabled=batch_enabled,
        )
        == expected
    )


def test_has_openai_key_defaults_to_false_for_backward_compat():
    """Callers from before the multi-provider expansion only passed
    ``has_anthropic_key``. Keep that working by defaulting
    ``has_openai_key`` to False — they'll still route correctly to
    anthropic_batch or sync_baseline."""
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=True, batch_processing_enabled=True
        )
        == "anthropic_batch"
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
            has_openai_key=True,
            batch_processing_enabled=True,
            transport_name=transport_name,
        )
        == "sync_baseline"
    )


@pytest.mark.parametrize("transport_name", ["openrouter", "direct_anthropic", None])
def test_batch_eligible_transports_unaffected(transport_name):
    """openrouter / direct_anthropic / ``None`` (no override) preserve
    the historical key-driven behaviour — the new ``transport_name``
    gate only fires for ``local`` and ``subscription``."""
    assert (
        resolve_dream_execution_path(
            has_anthropic_key=True,
            batch_processing_enabled=True,
            transport_name=transport_name,
        )
        == "anthropic_batch"
    )
