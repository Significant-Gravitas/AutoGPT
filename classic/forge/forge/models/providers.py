import abc
import enum
import math
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Secret, SecretBytes, SecretStr

from forge.models.config import SystemConfiguration, UserConfigurable

_T = TypeVar("_T")


class ResourceType(str, enum.Enum):
    """An enumeration of resource types."""

    MODEL = "model"


class ProviderBudget(SystemConfiguration, Generic[_T]):
    total_budget: float = UserConfigurable(math.inf)
    total_cost: float = 0
    remaining_budget: float = math.inf
    usage: _T

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

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            Secret: lambda v: v.get_secret_value() if v else None,
        }
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


# Used both by model providers and memory providers
Embedding = list[float]
