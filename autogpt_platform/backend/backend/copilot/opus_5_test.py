import os

import pytest

import backend.copilot.model_router as router
import backend.data.llm_registry.registry as registry
from backend.copilot.config import ChatConfig
from backend.copilot.model_normalize import normalize_model_for_transport
from backend.util.settings import BehaveAs


@pytest.fixture(autouse=True)
def isolated_catalog_and_config(monkeypatch):
    for key in os.environ:
        if key.startswith("CHAT_"):
            monkeypatch.delenv(key)
    for field in ("_dynamic_models", "_date_stripped_models", "_routes"):
        monkeypatch.setattr(registry, field, {})
    monkeypatch.setattr(registry, "_loaded", False)
    monkeypatch.setattr(router.settings.config, "behave_as", BehaveAs.CLOUD)
    registry.load_catalog()


@pytest.mark.parametrize("use_openrouter", [True, False])
async def test_thinking_advanced_routes_opus_5_without_catalog_refusal(
    use_openrouter, caplog, mocker
):
    cfg = ChatConfig(
        use_local=False,
        use_openrouter=use_openrouter,
        use_claude_code_subscription=False,
        api_key="test-key",
        aux_api_key="test-key",
    )
    mocker.patch.object(router, "get_feature_flag_value", return_value=None)
    route = await router.resolve_model_route("thinking", "advanced", "user", config=cfg)

    assert route.model == "anthropic/claude-opus-5"
    assert route.source == "env"
    assert "refused" not in caplog.text
    expected = "anthropic/claude-opus-5" if use_openrouter else "claude-opus-5"
    assert normalize_model_for_transport(route.model, cfg) == expected
    assert cfg.fast_advanced_model == "anthropic/claude-opus-4-8"
    assert cfg.thinking_standard_model == "anthropic/claude-sonnet-5"


def test_opus_5_has_public_metadata_and_provider_prices():
    model = router.catalog_lookup("anthropic/claude-opus-5")
    assert model is not None
    assert model.is_enabled
    assert model.visibility == "GA"
    assert model.display_name == "Claude Opus 5"
    assert model.metadata.context_window == 200_000
    assert model.metadata.max_output_tokens == 128_000
    assert model.supports_tools and model.supports_reasoning
    assert model.cost is not None
    assert model.cost.provider_input_usd_per_1m == 5.0
    assert model.cost.provider_output_usd_per_1m == 25.0
