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

    class Config:
        extra = "forbid"


class Configurable(abc.ABC):
    """A base class for all configurable objects."""

    prefix: str = ""
    defaults: typing.ClassVar[SystemSettings]

    @classmethod
    def process_user_configuration(cls, configuration: dict) -> SystemSettings:
        """Process the configuration for this object."""

        final_configuration = cls.defaults.dict()
        final_configuration.update(configuration)

        return cls.defaults.__class__.parse_obj(final_configuration)
