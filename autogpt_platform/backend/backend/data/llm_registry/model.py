"""Type definitions for LLM model metadata.

Re-exports ModelMetadata from blocks.llm to avoid type collision.
In PR #5 (block integration), this will become the canonical location.
"""

from backend.blocks.llm import ModelMetadata

__all__ = ["ModelMetadata"]
