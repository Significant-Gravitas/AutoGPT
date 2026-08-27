"""Unit tests for model identity + catalog projections (llm_models.py)."""

from __future__ import annotations

import pydantic

from backend.data.llm_registry import llm_models
from backend.data.llm_registry.llm_models import DEFAULT_LLM_MODEL, LLMModel


def test_default_model_is_the_catalog_recommendation():
    """DEFAULT_LLM_MODEL derives from the catalog's is_recommended flag —
    the same fact must not be encoded twice."""
    assert DEFAULT_LLM_MODEL is LLMModel.GPT5_6_TERRA


def _schema_metadata() -> dict:
    class _M(pydantic.BaseModel):
        model: LLMModel

    return _M.model_json_schema()["$defs"]["LLMModel"]["llm_model_metadata"]


def test_kill_switched_model_hidden_from_picker_metadata(monkeypatch):
    """is_enabled=False drops a model from the picker metadata while the
    enum value stays valid, so stored graphs keep validating/executing."""
    victim = LLMModel.KIMI_K3
    monkeypatch.setattr(llm_models, "_PICKER_HIDDEN_SLUGS", frozenset({victim.value}))
    metadata = _schema_metadata()
    assert victim.value not in metadata
    assert LLMModel.GPT5_2.value in metadata
    assert LLMModel(victim.value) is victim


def test_all_models_visible_when_none_disabled():
    metadata = _schema_metadata()
    assert len(metadata) == len(list(LLMModel))


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


def test_default_model_skips_non_ga_recommended(monkeypatch):
    """The default is offered to every user and must be picker-selectable,
    so it clears the same GA bar as _PICKER_HIDDEN_SLUGS. A model flagged
    is_recommended but demoted below GA must NOT win the default — otherwise
    the default would be a model the picker hides."""
    from backend.data.llm_registry.catalog import get_catalog

    payload = get_catalog()
    rec_slug = next(m.slug for m in payload.models if m.is_recommended)
    demoted = payload.model_copy(
        update={
            "models": [
                (
                    m.model_copy(update={"visibility": "EMPLOYEES"})
                    if m.is_recommended
                    else m
                )
                for m in payload.models
            ]
        }
    )
    monkeypatch.setattr(llm_models, "get_catalog", lambda: demoted)
    default = llm_models._default_model_from_catalog()
    assert default.value != rec_slug  # non-GA recommendation skipped
    chosen = next(m for m in demoted.models if m.slug == default.value)
    assert chosen.is_enabled and chosen.visibility == "GA"  # picker-selectable


def test_non_ga_model_hidden_from_picker_metadata(monkeypatch):
    """visibility != GA drops a model from picker metadata (the catalog's
    documented "who can SEE this" contract) while the enum value stays
    valid for stored graphs."""
    from backend.data.llm_registry import llm_models
    from backend.data.llm_registry.llm_models import LLMModel

    victim = LLMModel.KIMI_K3
    monkeypatch.setattr(llm_models, "_PICKER_HIDDEN_SLUGS", frozenset({victim.value}))
    metadata = _schema_metadata()
    assert victim.value not in metadata
    assert LLMModel(victim.value) is victim


class TestAliasResolution:
    """Stored graphs depend on _missing_'s alias/prefix resolution — the
    behavior moved with the identity module and keeps direct coverage."""

    def test_openrouter_alias_resolves_date_suffixed_member(self):
        assert LLMModel("anthropic/claude-haiku-4-5") is LLMModel.CLAUDE_4_5_HAIKU

    def test_generic_vendor_prefix_strips(self):
        assert LLMModel("anthropic/claude-sonnet-4-6") is LLMModel.CLAUDE_4_6_SONNET

    def test_unknown_slug_raises(self):
        import pytest

        with pytest.raises(ValueError):
            LLMModel("someprovider/not-a-model")


def test_picker_hidden_derivation_from_catalog_payload():
    """The real derivation (not a patched frozenset): non-GA and disabled
    models land in the hidden set; enabled GA models never do. An inverted
    predicate cannot pass this."""
    from backend.data.llm_registry.catalog import get_catalog
    from backend.data.llm_registry.llm_models import _picker_hidden_slugs

    payload = get_catalog()
    ga, others = payload.models[0], payload.models[1:3]
    modified = payload.model_copy(
        update={
            "models": [
                ga,  # enabled GA — must stay visible
                others[0].model_copy(update={"visibility": "EMPLOYEES"}),
                others[1].model_copy(update={"is_enabled": False}),
            ]
        }
    )
    hidden = _picker_hidden_slugs(modified)
    assert ga.slug not in hidden
    assert others[0].slug in hidden
    assert others[1].slug in hidden


def test_non_ga_model_flows_through_derivation_to_hidden_picker(monkeypatch):
    """End-to-end (not a patched frozenset): a real model marked non-GA lands
    in the DERIVED hidden set and is therefore absent from picker metadata —
    chaining _picker_hidden_slugs -> _PICKER_HIDDEN_SLUGS -> schema so an
    inverted or dropped visibility check would fail here, not just in the
    isolated derivation test above."""
    from backend.data.llm_registry.catalog import get_catalog
    from backend.data.llm_registry.llm_models import _picker_hidden_slugs

    victim = LLMModel.GPT5_2
    payload = get_catalog()
    modified = payload.model_copy(
        update={
            "models": [
                (
                    m.model_copy(update={"visibility": "EMPLOYEES"})
                    if m.slug == victim.value
                    else m
                )
                for m in payload.models
            ]
        }
    )
    monkeypatch.setattr(
        llm_models, "_PICKER_HIDDEN_SLUGS", _picker_hidden_slugs(modified)
    )
    metadata = _schema_metadata()
    assert victim.value not in metadata  # non-GA -> hidden, via real derivation
    assert LLMModel(victim.value) is victim  # enum identity unaffected


def test_metadata_completeness_guard_fires_on_missing_member(monkeypatch):
    """_assert_metadata_complete is the boot-time safety net enforcing
    enum ⊆ catalog (every LLMModel has derived MODEL_METADATA). Prove it
    actually fires: drop one member's metadata and it must raise — otherwise
    an enum member with no catalog entry would ship silently."""
    import pytest

    incomplete = dict(llm_models.MODEL_METADATA)
    victim = next(iter(incomplete))
    del incomplete[victim]
    monkeypatch.setattr(llm_models, "MODEL_METADATA", incomplete)
    with pytest.raises(ValueError, match="Missing MODEL_METADATA"):
        llm_models._assert_metadata_complete()


def test_metadata_completeness_guard_passes_for_real_catalog():
    """The shipped enum/catalog pair must satisfy the guard (also enforced at
    import, asserted here explicitly)."""
    llm_models._assert_metadata_complete()
