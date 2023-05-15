import abc
import typing

from pydantic import BaseModel


class SystemConfiguration(BaseModel):
    """A base class for all system configuration."""

    pass


class SystemSettings(BaseModel):
    """A base class for all system settings."""

    name: str
    description: str
    configuration: SystemConfiguration | None = None


class Configurable(abc.ABC):
    """A base class for all configurable objects."""

    defaults: typing.ClassVar[SystemSettings]

    @classmethod
    def process_configuration(cls, configuration: dict) -> SystemSettings:
        # TODO: Robust validation.

        final_configuration = cls.defaults.dict()
        final_configuration.update(configuration)
        return cls.defaults.__class__.parse_obj(configuration)
