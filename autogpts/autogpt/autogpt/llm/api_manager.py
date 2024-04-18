from __future__ import annotations

import logging
from typing import List, Optional

from openai import APIError, AzureOpenAI, OpenAI
from openai.types import Model

from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_MODELS,
    OpenAICredentials,
)
from autogpt.core.resource.model_providers.schema import ChatModelInfo
from autogpt.singleton import Singleton

logger = logging.getLogger(__name__)


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # the .model property in API responses can contain version suffixes like -v2
        model = model[:-3] if model.endswith("-v2") else model
        model_info = OPEN_AI_MODELS[model]

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += prompt_tokens * model_info.prompt_token_cost / 1000
        if isinstance(model_info, ChatModelInfo):
            self.total_cost += (
                completion_tokens * model_info.completion_token_cost / 1000
            )

        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget

    def get_models(self, openai_credentials: OpenAICredentials) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
            list[Model]: List of available GPT models.
        """
        if self.models is not None:
            return self.models

        try:
            if openai_credentials.api_type == "azure":
                all_models = (
                    AzureOpenAI(**openai_credentials.get_api_access_kwargs())
                    .models.list()
                    .data
                )
            else:
                all_models = (
                    OpenAI(**openai_credentials.get_api_access_kwargs())
                    .models.list()
                    .data
                )
            self.models = [model for model in all_models if "gpt" in model.id]
        except APIError as e:
            logger.error(e.message)
            exit(1)

        return self.models

    def get_models_llamafile(self, openai_credentials: OpenAICredentials) -> List[Model]:
        """
        Same as `get_models` but doesn't filter out non-'gpt' models.
        TODO: No real reason for this to be a separate method but I don't know
          what effect it will have on the OpenAI-related to remove the
          'gpt-only' filter from the original method.
        """
        if self.models is not None:
            return self.models

        try:
            all_models = (
                OpenAI(**openai_credentials.get_api_access_kwargs())
                .models.list()
                .data
            )
            self.models = [model for model in all_models]
        except APIError as e:
            logger.error(e.message)
            exit(1)

        return self.models
