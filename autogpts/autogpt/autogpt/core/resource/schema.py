import abc
import enum

from pydantic import SecretBytes, SecretField, SecretStr

from autogpts.autogpt.autogpt.core.configuration import (SystemConfiguration,
                                                         SystemSettings,
                                                         UserConfigurable)


class ResourceType(str, enum.Enum):
    """An enumeration of resource types."""

    MODEL = "model"
    MEMORY = "memory"


class BaseProviderUsage(SystemConfiguration, abc.ABC):
    @abc.abstractmethod
    def update_usage(self, *args, **kwargs) -> None:
        """Update the usage of the resource."""
        ...


class BaseProviderBudget(SystemConfiguration):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: BaseProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...


class BaseProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    class Config(SystemConfiguration.Config):
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


class BaseProviderSettings(SystemSettings):
    resource_type: ResourceType
    credentials: BaseProviderCredentials | None = None
    budget: BaseProviderBudget | None = None


# Used both by model providers and memory providers
Embedding = list[float]
