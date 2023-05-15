import abc
import typing

from pydantic import BaseModel, SecretField


class SystemConfiguration(BaseModel):
    pass


class Credentials(SystemConfiguration):
    class Config:
        json_encoders = {
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


class ResourceBudget(SystemConfiguration):
    total_budget: float
    total_cost: float
    remaining_budget: float
    usage: typing.Any

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...


class SystemSettings(BaseModel):
    name: str
    description: str
    configuration: SystemConfiguration | None = None
    credentials: Credentials | None = None
    budget: ResourceBudget | None = None


class Configurable(abc.ABC):
    """A base class for all configurable objects."""

    defaults: typing.ClassVar[SystemSettings]

    @classmethod
    def process_configuration(cls, configuration: dict) -> SystemSettings:
        # TODO: Robust validation.

        final_configuration = cls.defaults.dict()
        final_configuration.update(configuration)
        return cls.defaults.__class__.parse_obj(configuration)
