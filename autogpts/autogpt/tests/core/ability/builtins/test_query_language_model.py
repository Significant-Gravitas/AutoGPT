"""
This file provides all the necessary tests for the query_lanuguage_model.py file
defined in autogpt/core/ability/builtins/
"""

import logging
from typing import Any, Callable

import pytest

from autogpt.core.ability.base import AbilityConfiguration
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration.schema import Configurable
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers.openai import OpenAIModelName
from autogpt.core.resource.model_providers.schema import (
    LanguageModelFunction,
    LanguageModelMessage,
    LanguageModelProviderModelInfo,
    LanguageModelProviderModelResponse,
    ModelProviderName,
    ModelProviderService,
)


class MockLanguageModelProvider(
    Configurable,
    # LanguageModelProvider,
):
    """
    A mock language model provider to be used inside TestQueryLanguageModel
    class.
    """

    async def create_language_completion(
        self,
        model_prompt: list[LanguageModelMessage],
        functions: list[LanguageModelFunction],
        model_name: OpenAIModelName,
        completion_parser: Callable[[dict], dict],
        **kwargs: Any,
    ) -> LanguageModelProviderModelResponse:
        """
        A mock create_language_completion method to be used
        test_call_method(...) inside TestQueryLanguageModel class.
        """

        response_args = {
            "model_info": LanguageModelProviderModelInfo(
                name="fake-model",
                service=ModelProviderService.LANGUAGE,
                provider_name=ModelProviderName.OPENAI,
                max_tokens=0,
            ),
            "prompt_tokens_used": 0,
            "completion_tokens_used": 0,
            "content": {"content": "fake response content"},
        }

        response = LanguageModelProviderModelResponse(**response_args)
        return response


class TestQueryLanguageModel:
    """
    A test class for the QueryLanguageModel class.
    """

    def setup_method(self) -> None:
        """
        Sets up the test class.
        """
        self.ability_configuration = AbilityConfiguration(
            location=PluginLocation(
                storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                storage_route="autogpt.core.ability.builtins.query_language_model.QueryLanguageModel",
            ),
            language_model_required=LanguageModelConfiguration(
                model_name=OpenAIModelName.GPT3,
                provider_name=ModelProviderName.OPENAI,
                temperature=0.9,
            ),
        )

        self.logger = logging.getLogger(__name__)

        self.language_model_provider = MockLanguageModelProvider()

        self.query_language_model = QueryLanguageModel(
            logger=self.logger,
            configuration=self.ability_configuration,
            language_model_provider=self.language_model_provider,
        )

    def test_init(self) -> None:
        """Tests the initizaliztion of the QueryLanguageModel class."""
        assert self.query_language_model._logger == self.logger
        assert self.query_language_model._configuration == self.ability_configuration
        assert (
            self.query_language_model._language_model_provider
            == self.language_model_provider
        )

    @staticmethod
    def test_description_method() -> None:
        """
        Tests the description method of the QueryLanguageModel class.
        """
        assert (
            QueryLanguageModel.description()
            == "Query a language model. A query should be a question and any relevant context."
        )

    @staticmethod
    def test_arguments_method() -> None:
        """
        Tests the arguments method of the QueryLanguageModel class.
        """
        assert QueryLanguageModel.arguments() == {
            "query": {
                "type": "string",
                "description": "A query for a language model. A query should contain a question and any relevant context.",
            },
        }

    @staticmethod
    def test_required_arguments_method() -> None:
        """
        Tests the required_arguments method of the QueryLanguageModel class.
        """
        assert QueryLanguageModel.required_arguments() == ["query"]

    @pytest.mark.asyncio
    async def test_call_method(self) -> None:
        """
        Tests the __call__ method of the QueryLanguageModel class.
        """

        # self.query_language_model._language_model_provider._credentials.api_key = 'fake-api-key'

        result = await self.query_language_model("sample query")
        assert isinstance(result, AbilityResult)
        assert result.message == "fake response content"
        assert result.success
        assert result.ability_args == {"query": "sample query"}
        assert result.ability_name == self.query_language_model.name()
