"""Backward-compatibility shim — ``split_camelcase`` now lives in backend.util.text."""

from backend.util.text import split_camelcase

__all__ = ["split_camelcase"]
