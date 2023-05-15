import abc
import typing

from pydantic import BaseModel, SecretField


class Credentials(BaseModel):
    pass

    class Config:
        json_encoders = {
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


class SystemConfiguration(BaseModel):
    pass


class SystemSettings(BaseModel):
    name: str
    description: str
    configuration: SystemConfiguration | None = None
    credentials: Credentials | None = None


class Configurable(BaseModel, abc.ABC):
    """A base class for all configurable objects."""
    defaults: typing.ClassVar[SystemSettings]

    @classmethod
    def process_configuration(cls, configuration: dict) -> SystemSettings:
        # TODO: Robust validation.

        final_configuration = cls.defaults.dict()
        final_configuration.update(configuration)
        return cls.defaults.__class__.parse_obj(configuration)


