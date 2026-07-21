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
    monkeypatch.setattr(llm_models, "_PICKER_HIDDEN_SLUGS", frozenset({victim.value}))
    metadata = _schema_metadata()
    assert victim.value not in metadata
    assert LlmModel.GPT5_2.value in metadata
    assert LlmModel(victim.value) is victim


def test_all_models_visible_when_none_disabled():
    metadata = _schema_metadata()
    assert len(metadata) == len(list(LlmModel))


def test_default_model_survives_killing_the_recommended_entry(monkeypatch):
    """Killing the only is_recommended model must not crash boot — the
    default falls back to the first enabled block-selectable model."""
    from backend.data.llm_registry import llm_models
    from backend.data.llm_registry.catalog import get_catalog

    payload = get_catalog()
    killed = payload.model_copy(
        update={
            "models": [
                (m.model_copy(update={"is_enabled": False}) if m.is_recommended else m)
                for m in payload.models
            ]
        }
    )
    monkeypatch.setattr(llm_models, "get_catalog", lambda: killed)
    fallback = llm_models._default_model_from_catalog()
    assert fallback is not None
    assert not any(m.is_recommended and m.is_enabled for m in killed.models)


def test_non_ga_model_hidden_from_picker_metadata(monkeypatch):
    """visibility != GA drops a model from picker metadata (the catalog's
    documented "who can SEE this" contract) while the enum value stays
    valid for stored graphs."""
    from backend.data.llm_registry import llm_models
    from backend.data.llm_registry.llm_models import LlmModel

    victim = LlmModel.KIMI_K3
    monkeypatch.setattr(llm_models, "_PICKER_HIDDEN_SLUGS", frozenset({victim.value}))
    metadata = _schema_metadata()
    assert victim.value not in metadata
    assert LlmModel(victim.value) is victim
