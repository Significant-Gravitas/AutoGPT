import math

from pydantic import Field, SecretStr, validator

from autogpt.core.configuration import Credentials, ResourceBudget, SystemConfiguration
from autogpt.core.model.base import ModelResponse


class ProviderModelCredentials(Credentials):
    """Credentials for a specific model from a provider.

    Model credentials are used to authenticate with a specific model
    from a provider. They share the same keys as the `ProviderCredentials`
    class, which can be used to set global parameters for all models from
    a provider if certain credentials are used for multiple models.

    Attributes:
        api_key: The API key for the model.
        api_type: The type of API to use.
        api_base: The base URL for the API.
        api_version: The version of the API.
        deployment_id: The deployment ID for the API.

    """

    api_key: SecretStr | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    deployment_id: SecretStr | None = None


class ProviderCredentials(Credentials):
    """Credentials for a model provider.

    Provider credentials are used to authenticate with a model provider.
    They share the same keys as the `ModelCredentials` class, which can
    be used to authenticate with a specific model from a provider, but
    they can be used to set global parameters for all models from a
    provider if certain credentials are used for multiple models.

    Attributes:
        api_key: The API key for the provider.
        api_type: The type of API to use.
        api_base: The base URL for the API.
        api_version: The version of the API.
        deployment_id: The deployment ID for the API.

    """

    api_key: SecretStr | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    deployment_id: SecretStr | None = None
    models: dict[str, ProviderModelCredentials] = Field(default_factory=dict)

    @validator("models", pre=True)
    def at_least_one_model(cls, v):
        if not v:
            raise ValueError("At least one model must be defined.")
        return v

    def get_model_credentials(self) -> dict[str, ProviderModelCredentials]:
        provider_credentials = self.dict(exclude_unset=True, exclude={"models"})
        for model_name, model_credentials in self.models.items():
            model_dict = model_credentials.dict(exclude_unset=True)
            model_dict.update(provider_credentials)
            self.models[model_name] = ProviderModelCredentials(**model_dict)
        return self.models


class ProviderBudgetUsage(SystemConfiguration):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ProviderBudget(ResourceBudget):
    """Budget for a model provider."""

    usage: ProviderBudgetUsage
    graceful_shutdown_threshold: float
    warning_threshold: float

    def update_usage_and_cost(self, llm_response: ModelResponse) -> None:
        model_info = llm_response.model_info
        self.usage.prompt_tokens += llm_response.prompt_tokens_used
        self.usage.completion_tokens += llm_response.completion_tokens_used
        self.usage.total_tokens += (
            llm_response.prompt_tokens_used + llm_response.completion_tokens_used
        )
        incremental_cost = (
            llm_response.prompt_tokens_used * model_info.prompt_token_cost
            + llm_response.completion_tokens_used * model_info.completion_token_cost
        ) / 1000
        self.total_cost += incremental_cost
        self.remaining_budget -= incremental_cost

    def get_resource_budget_prompt(self) -> str:
        """Get the prompt to be used for the resource budget."""
        if self.total_budget == math.inf:
            return ""

        resource_prompt = f"Your remaining API budget is ${self.remaining_budget:.3f}"
        if self.remaining_budget <= 0:
            resource_prompt += " BUDGET EXCEEDED! SHUT DOWN!\n\n"
        elif self.remaining_budget < self.graceful_shutdown_threshold:
            resource_prompt += " Budget very nearly exceeded! Shut down gracefully!\n\n"
        elif self.remaining_budget < self.warning_threshold:
            resource_prompt += " Budget nearly exceeded. Finish up.\n\n"

        return resource_prompt


class ProviderConfiguration(SystemConfiguration):
    pass
