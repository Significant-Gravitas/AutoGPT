import abc
import enum

from pydantic import SecretBytes, SecretField, SecretStr

from autogpt.core.configuration import (
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)


class ResourceType(str, enum.Enum):
    """An enumeration of resource types."""

    MODEL = "model"
    MEMORY = "memory"


class ProviderUsage(SystemConfiguration, abc.ABC):
    @abc.abstractmethod
    def update_usage(self, *args, **kwargs) -> None:
        """Update the usage of the resource."""
        ...


class ProviderBudget(SystemConfiguration):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: ProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...


class ProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    class Config:
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


class ProviderSettings(SystemSettings):
    resource_type: ResourceType
    credentials: ProviderCredentials | None = None
    budget: ProviderBudget | None = None


# Used both by model providers and memory providers
Embedding = list[float]
