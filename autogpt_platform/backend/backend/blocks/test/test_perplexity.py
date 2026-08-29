"""Unit tests for PerplexityBlock model fallback behavior."""

import pytest

from backend.blocks.perplexity import (
    TEST_CREDENTIALS_INPUT,
    PerplexityBlock,
    PerplexityModel,
)


def _make_input(**overrides) -> dict:
    defaults = {
        "prompt": "test query",
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    defaults.update(overrides)
    return defaults


class TestPerplexityModelFallback:
    """Tests for fallback_invalid_model field_validator."""

    def test_invalid_model_falls_back_to_sonar(self):
        inp = PerplexityBlock.Input(**_make_input(model="gpt-5.2-2025-12-11"))
        assert inp.model == PerplexityModel.SONAR

    def test_another_invalid_model_falls_back_to_sonar(self):
        inp = PerplexityBlock.Input(**_make_input(model="gpt-4o"))
        assert inp.model == PerplexityModel.SONAR

    def test_valid_model_string_is_kept(self):
        inp = PerplexityBlock.Input(**_make_input(model="perplexity/sonar-pro"))
        assert inp.model == PerplexityModel.SONAR_PRO

    def test_valid_enum_value_is_kept(self):
        inp = PerplexityBlock.Input(
            **_make_input(model=PerplexityModel.SONAR_DEEP_RESEARCH)
        )
        assert inp.model == PerplexityModel.SONAR_DEEP_RESEARCH

    def test_default_model_when_omitted(self):
        inp = PerplexityBlock.Input(**_make_input())
        assert inp.model == PerplexityModel.SONAR

    @pytest.mark.parametrize(
        "model_value",
        [
            "perplexity/sonar",
            "perplexity/sonar-pro",
            "perplexity/sonar-deep-research",
        ],
    )
    def test_all_valid_models_accepted(self, model_value: str):
        inp = PerplexityBlock.Input(**_make_input(model=model_value))
        assert inp.model.value == model_value


class TestPerplexityValidateData:
    """Tests for validate_data which runs during block execution (before
    Pydantic instantiation). Invalid models must be sanitized here so
    JSON schema validation does not reject them."""

    def test_invalid_model_sanitized_before_schema_validation(self):
        data = _make_input(model="gpt-5.2-2025-12-11")
        error = PerplexityBlock.Input.validate_data(data)
        assert error is None
        assert data["model"] == PerplexityModel.SONAR.value

    def test_valid_model_unchanged_by_validate_data(self):
        data = _make_input(model="perplexity/sonar-pro")
        error = PerplexityBlock.Input.validate_data(data)
        assert error is None
        assert data["model"] == "perplexity/sonar-pro"

    def test_missing_model_uses_default(self):
        data = _make_input()  # no model key
        error = PerplexityBlock.Input.validate_data(data)
        assert error is None
        inp = PerplexityBlock.Input(**data)
        assert inp.model == PerplexityModel.SONAR
