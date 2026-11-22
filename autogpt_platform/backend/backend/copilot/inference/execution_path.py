"""Execution-path types and pricing shared by background inference callers."""

from typing import Literal

ExecutionPath = Literal["sync_baseline", "anthropic_batch"]

_ANTHROPIC_BATCH_DISCOUNT = 0.5


def batch_discount(path: ExecutionPath) -> float:
    """The provider discount for a background call's execution path."""
    return _ANTHROPIC_BATCH_DISCOUNT if path == "anthropic_batch" else 0.0
