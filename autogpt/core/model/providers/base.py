from pydantic import Field, SecretStr, validator

from autogpt.core.configuration import Credentials


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
