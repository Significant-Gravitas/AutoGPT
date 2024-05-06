import abc
import enum
import math

from pydantic import BaseModel, SecretBytes, SecretField, SecretStr

from forge.config.schema import SystemConfiguration, SystemSettings, UserConfigurable


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
    total_budget: float = UserConfigurable(math.inf)
    total_cost: float = 0
    remaining_budget: float = math.inf
    usage: ProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> float:
        """Update the usage and cost of the provider.

        Returns:
            float: The (calculated) cost of the given model response.
        """
        ...


class ProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    def unmasked(self) -> dict:
        return unmask(self)

    class Config:
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


def unmask(model: BaseModel):
    unmasked_fields = {}
    for field_name, _ in model.__fields__.items():
        value = getattr(model, field_name)
        if isinstance(value, SecretStr):
            unmasked_fields[field_name] = value.get_secret_value()
        else:
            unmasked_fields[field_name] = value
    return unmasked_fields


class ProviderSettings(SystemSettings):
    resource_type: ResourceType
    credentials: ProviderCredentials | None = None
    budget: ProviderBudget | None = None


# Used both by model providers and memory providers
Embedding = list[float]
