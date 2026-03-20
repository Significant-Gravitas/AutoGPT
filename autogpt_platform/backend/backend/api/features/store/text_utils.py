"""Backward-compatibility shim — ``split_camelcase`` now lives in backend.util.text."""

from backend.util.text import split_camelcase  # noqa: F401

__all__ = ["split_camelcase"]
