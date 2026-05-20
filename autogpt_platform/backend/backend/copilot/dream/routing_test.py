"""Execution-path resolver — drives whether a dream goes batch or sync."""

from __future__ import annotations

import pytest

from .routing import resolve_dream_execution_path


@pytest.mark.parametrize(
    "has_anthropic_key,batch_enabled,expected",
    [
        (True, True, "batch"),
        (True, False, "sync_baseline"),
        (False, True, "sync_baseline"),
        (False, False, "sync_baseline"),
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
