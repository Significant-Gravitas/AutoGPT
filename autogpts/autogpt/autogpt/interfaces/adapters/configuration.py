import abc

from pydantic import BaseModel, SecretBytes, SecretStr

from AFAAS.configs.config import SystemSettings, UserConfigurable
from autogpt.core.configuration import SystemConfiguration
, update_model_config
from pydantic import  ConfigDict
from pydantic.fields import Field


class BaseProviderUsage(SystemConfiguration, abc.ABC):
    @abc.abstractmethod
    def update_usage(self, *args, **kwargs) -> None:
        """Update the usage of the resource."""
        ...


class BaseProviderBudget(SystemConfiguration):
    total_budget: float = Field()
    total_cost: float
    remaining_budget: float
    usage: BaseProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...


class BaseProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    def unmasked(self) -> dict:
        return self.unmask(self)

    model_config = update_model_config(original= SystemConfiguration.model_config , 
       new =  {
            'json_encoders' : {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
        }}
    )

    def unmask(model: BaseModel):
        unmasked_fields = {}
        for field_name, _ in model.model_fields.items():
            value = getattr(model, field_name)
            if isinstance(value, SecretStr):
                unmasked_fields[field_name] = value.get_secret_value()
            else:
                unmasked_fields[field_name] = value
        return unmasked_fields


class BaseProviderSettings(SystemSettings):
    credentials: BaseProviderCredentials | None = None
    budget: BaseProviderBudget | None = None
