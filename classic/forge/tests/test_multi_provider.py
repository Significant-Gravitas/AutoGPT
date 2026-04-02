"""Tests for MultiProvider: routing, caching, credentials, model registry."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.llm.providers.anthropic import AnthropicModelName
from forge.llm.providers.groq import GroqModelName
from forge.llm.providers.multi import CHAT_MODELS, MultiProvider
from forge.llm.providers.openai import OPEN_AI_CHAT_MODELS, OpenAIModelName
from forge.llm.providers.schema import (
    ChatMessage,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderName,
    ModelProviderSettings,
)


# ---------------------------------------------------------------------------
# CHAT_MODELS registry
# ---------------------------------------------------------------------------
class TestChatModelsRegistry:
    def test_contains_openai_models(self):
        assert OpenAIModelName.GPT4_O in CHAT_MODELS

    def test_contains_anthropic_models(self):
        assert AnthropicModelName.CLAUDE4_SONNET_v1 in CHAT_MODELS

    def test_contains_groq_models(self):
        assert GroqModelName.MIXTRAL_8X7B in CHAT_MODELS

    def test_gpt5_models_registered(self):
        assert OpenAIModelName.GPT5 in CHAT_MODELS
        assert OpenAIModelName.GPT5_2 in CHAT_MODELS
        assert OpenAIModelName.GPT5_3 in CHAT_MODELS
        assert OpenAIModelName.GPT5_4 in CHAT_MODELS

    def test_gpt5_pro_models_registered(self):
        assert OpenAIModelName.GPT5_PRO in CHAT_MODELS
        assert OpenAIModelName.GPT5_2_PRO in CHAT_MODELS
        assert OpenAIModelName.GPT5_3_PRO in CHAT_MODELS
        assert OpenAIModelName.GPT5_4_PRO in CHAT_MODELS

    def test_claude_opus_46_registered(self):
        assert AnthropicModelName.CLAUDE4_6_OPUS_v1 in CHAT_MODELS

    def test_rolling_aliases_resolve_in_registry(self):
        """Rolling aliases must be resolvable in CHAT_MODELS."""
        assert AnthropicModelName.CLAUDE_OPUS in CHAT_MODELS
        assert AnthropicModelName.CLAUDE_SONNET in CHAT_MODELS
        assert AnthropicModelName.CLAUDE_HAIKU in CHAT_MODELS

    def test_every_registered_model_has_provider(self):
        """Every model in the registry must have a valid provider_name."""
        valid_providers = set(ModelProviderName)
        for model_name, info in CHAT_MODELS.items():
            assert (
                info.provider_name in valid_providers
            ), f"Model {model_name} has unknown provider {info.provider_name}"

    def test_every_registered_model_has_positive_max_tokens(self):
        for model_name, info in CHAT_MODELS.items():
            assert info.max_tokens > 0, f"{model_name} has max_tokens={info.max_tokens}"


# ---------------------------------------------------------------------------
# MultiProvider initialization
# ---------------------------------------------------------------------------
class TestMultiProviderInit:
    def test_default_settings(self):
        provider = MultiProvider()
        assert provider._provider_instances == {}
        assert isinstance(provider._budget, ModelProviderBudget)

    def test_custom_settings(self):
        settings = ModelProviderSettings(
            name="custom",
            description="Custom provider",
            configuration=ModelProviderConfiguration(retries_per_request=3),
            budget=ModelProviderBudget(),
        )
        provider = MultiProvider(settings=settings)
        assert provider._configuration.retries_per_request == 3


# ---------------------------------------------------------------------------
# _get_provider_class
# ---------------------------------------------------------------------------
class TestGetProviderClass:
    def test_openai(self):
        from forge.llm.providers.openai import OpenAIProvider

        cls = MultiProvider._get_provider_class(ModelProviderName.OPENAI)
        assert cls is OpenAIProvider

    def test_anthropic(self):
        from forge.llm.providers.anthropic import AnthropicProvider

        cls = MultiProvider._get_provider_class(ModelProviderName.ANTHROPIC)
        assert cls is AnthropicProvider

    def test_groq(self):
        from forge.llm.providers.groq import GroqProvider

        cls = MultiProvider._get_provider_class(ModelProviderName.GROQ)
        assert cls is GroqProvider

    def test_llamafile(self):
        from forge.llm.providers.llamafile import LlamafileProvider

        cls = MultiProvider._get_provider_class(ModelProviderName.LLAMAFILE)
        assert cls is LlamafileProvider

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not a known provider"):
            MultiProvider._get_provider_class("nonexistent")  # type: ignore


# ---------------------------------------------------------------------------
# get_model_provider routing — actually calls the router
# ---------------------------------------------------------------------------
class TestGetModelProvider:
    def test_routes_openai_model_to_openai_provider(self):
        from forge.llm.providers.openai import OpenAIProvider

        provider = MultiProvider()
        mock_openai = MagicMock(spec=OpenAIProvider)
        provider._provider_instances[ModelProviderName.OPENAI] = mock_openai

        result = provider.get_model_provider(OpenAIModelName.GPT4_O)
        assert result is mock_openai

    def test_routes_anthropic_model_to_anthropic_provider(self):
        from forge.llm.providers.anthropic import AnthropicProvider

        provider = MultiProvider()
        mock_anthropic = MagicMock(spec=AnthropicProvider)
        provider._provider_instances[ModelProviderName.ANTHROPIC] = mock_anthropic

        result = provider.get_model_provider(AnthropicModelName.CLAUDE4_SONNET_v1)
        assert result is mock_anthropic

    def test_routes_groq_model_to_groq_provider(self):
        from forge.llm.providers.groq import GroqProvider

        provider = MultiProvider()
        mock_groq = MagicMock(spec=GroqProvider)
        provider._provider_instances[ModelProviderName.GROQ] = mock_groq

        result = provider.get_model_provider(GroqModelName.MIXTRAL_8X7B)
        assert result is mock_groq

    def test_unknown_model_raises_key_error(self):
        provider = MultiProvider()
        with pytest.raises(KeyError):
            provider.get_model_provider("nonexistent-model")  # type: ignore

    def test_different_models_same_provider_return_same_instance(self):
        """Two OpenAI models should route to the same provider instance."""
        provider = MultiProvider()
        mock_openai = MagicMock()
        provider._provider_instances[ModelProviderName.OPENAI] = mock_openai

        p1 = provider.get_model_provider(OpenAIModelName.GPT4_O)
        p2 = provider.get_model_provider(OpenAIModelName.GPT5)
        assert p1 is p2


# ---------------------------------------------------------------------------
# _get_provider caching — tests the actual initialization path
# ---------------------------------------------------------------------------
class TestProviderCaching:
    def test_second_call_returns_cached_instance(self):
        """After first init, the same object is returned without re-creating."""
        provider = MultiProvider()
        mock_instance = MagicMock()
        # Simulate first call already populated the cache
        provider._provider_instances[ModelProviderName.OPENAI] = mock_instance

        p1 = provider._get_provider(ModelProviderName.OPENAI)
        p2 = provider._get_provider(ModelProviderName.OPENAI)
        assert p1 is p2 is mock_instance

    def test_different_providers_not_shared(self):
        provider = MultiProvider()
        mock_openai = MagicMock()
        mock_anthropic = MagicMock()
        provider._provider_instances[ModelProviderName.OPENAI] = mock_openai
        provider._provider_instances[ModelProviderName.ANTHROPIC] = mock_anthropic

        assert provider._get_provider(ModelProviderName.OPENAI) is not (
            provider._get_provider(ModelProviderName.ANTHROPIC)
        )


# ---------------------------------------------------------------------------
# Token limit / count delegation
# ---------------------------------------------------------------------------
class TestMultiProviderDelegation:
    def test_get_token_limit_delegates(self):
        provider = MultiProvider()
        mock_sub = MagicMock()
        mock_sub.get_token_limit.return_value = 128000
        provider._provider_instances[ModelProviderName.OPENAI] = mock_sub

        limit = provider.get_token_limit(OpenAIModelName.GPT4_O)
        assert limit == 128000
        mock_sub.get_token_limit.assert_called_once()

    def test_count_tokens_delegates(self):
        provider = MultiProvider()
        mock_sub = MagicMock()
        mock_sub.count_tokens.return_value = 42
        provider._provider_instances[ModelProviderName.OPENAI] = mock_sub

        count = provider.count_tokens("hello world", OpenAIModelName.GPT4_O)
        assert count == 42

    def test_count_message_tokens_delegates(self):
        provider = MultiProvider()
        mock_sub = MagicMock()
        mock_sub.count_message_tokens.return_value = 10
        provider._provider_instances[ModelProviderName.OPENAI] = mock_sub

        msg = ChatMessage.user("test")
        count = provider.count_message_tokens(msg, OpenAIModelName.GPT4_O)
        assert count == 10

    @pytest.mark.asyncio
    async def test_create_chat_completion_delegates(self):
        provider = MultiProvider()
        mock_sub = MagicMock()
        mock_result = MagicMock()
        mock_sub.create_chat_completion = AsyncMock(return_value=mock_result)
        provider._provider_instances[ModelProviderName.OPENAI] = mock_sub

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=OpenAIModelName.GPT4_O,
        )
        assert result is mock_result
        mock_sub.create_chat_completion.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI model definitions
# ---------------------------------------------------------------------------
class TestOpenAIModelDefinitions:
    def test_gpt5_family_all_support_reasoning_and_tools(self):
        """Every GPT-5 variant must support reasoning_effort and function calls."""
        gpt5_models = [
            OpenAIModelName.GPT5,
            OpenAIModelName.GPT5_1,
            OpenAIModelName.GPT5_2,
            OpenAIModelName.GPT5_3,
            OpenAIModelName.GPT5_4,
            OpenAIModelName.GPT5_MINI,
            OpenAIModelName.GPT5_NANO,
            OpenAIModelName.GPT5_PRO,
            OpenAIModelName.GPT5_2_PRO,
            OpenAIModelName.GPT5_3_PRO,
            OpenAIModelName.GPT5_4_PRO,
            OpenAIModelName.GPT5_4_MINI,
            OpenAIModelName.GPT5_4_NANO,
        ]
        for model in gpt5_models:
            info = OPEN_AI_CHAT_MODELS[model]
            assert info.supports_reasoning_effort is True, f"{model} missing reasoning"
            assert info.has_function_call_api is True, f"{model} missing function calls"
            assert info.max_tokens in (
                400_000,
                1_000_000,
            ), f"{model} unexpected context size {info.max_tokens}"

    def test_pro_models_cost_more_than_base(self):
        """Pro variants must cost more than their base counterparts."""
        pairs = [
            (OpenAIModelName.GPT5, OpenAIModelName.GPT5_PRO),
            (OpenAIModelName.GPT5_2, OpenAIModelName.GPT5_2_PRO),
            (OpenAIModelName.GPT5_3, OpenAIModelName.GPT5_3_PRO),
            (OpenAIModelName.GPT5_4, OpenAIModelName.GPT5_4_PRO),
        ]
        for base, pro in pairs:
            base_info = OPEN_AI_CHAT_MODELS[base]
            pro_info = OPEN_AI_CHAT_MODELS[pro]
            assert (
                pro_info.prompt_token_cost > base_info.prompt_token_cost
            ), f"{pro} should cost more than {base}"

    def test_mini_and_nano_cost_less_than_base(self):
        base_cost = OPEN_AI_CHAT_MODELS[OpenAIModelName.GPT5].prompt_token_cost
        assert (
            OPEN_AI_CHAT_MODELS[OpenAIModelName.GPT5_MINI].prompt_token_cost < base_cost
        )
        assert (
            OPEN_AI_CHAT_MODELS[OpenAIModelName.GPT5_NANO].prompt_token_cost < base_cost
        )

    def test_cost_ordering_across_gpt5_tiers(self):
        """nano < mini < base < pro (by prompt cost)."""
        costs = [
            OPEN_AI_CHAT_MODELS[m].prompt_token_cost
            for m in [
                OpenAIModelName.GPT5_NANO,
                OpenAIModelName.GPT5_MINI,
                OpenAIModelName.GPT5,
                OpenAIModelName.GPT5_PRO,
            ]
        ]
        assert costs == sorted(costs), f"Cost ordering violated: {costs}"
