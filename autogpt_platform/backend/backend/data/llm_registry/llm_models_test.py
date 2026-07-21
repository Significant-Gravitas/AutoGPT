"""Unit tests for model identity + catalog projections (llm_models.py)."""

from __future__ import annotations

import pydantic

from backend.data.llm_registry import llm_models
from backend.data.llm_registry.llm_models import DEFAULT_LLM_MODEL, LlmModel


def test_default_model_is_the_catalog_recommendation():
    """DEFAULT_LLM_MODEL derives from the catalog's is_recommended flag —
    the same fact must not be encoded twice."""
    assert DEFAULT_LLM_MODEL is LlmModel.GPT5_2


def _schema_metadata() -> dict:
    class _M(pydantic.BaseModel):
        model: LlmModel

    return _M.model_json_schema()["$defs"]["LlmModel"]["llm_model_metadata"]


def test_kill_switched_model_hidden_from_picker_metadata(monkeypatch):
    """is_enabled=False drops a model from the picker metadata while the
    enum value stays valid, so stored graphs keep validating/executing."""
    victim = LlmModel.KIMI_K3
    monkeypatch.setattr(llm_models, "_DISABLED_SLUGS", frozenset({victim.value}))
    metadata = _schema_metadata()
    assert victim.value not in metadata
    assert LlmModel.GPT5_2.value in metadata
    assert LlmModel(victim.value) is victim


def test_all_models_visible_when_none_disabled():
    metadata = _schema_metadata()
    assert len(metadata) == len(list(LlmModel))
