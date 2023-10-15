import abc
import uuid
from typing import Any, Callable, Generic, Optional, TypeVar, Union
import logging
import datetime

from pydantic import BaseModel, Field


def UserConfigurable(*args, **kwargs):
    return Field(*args, **kwargs, user_configurable=True)
    # TODO: use this to auto-generate docs for the application configuration


class SystemConfiguration(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True            
        json_encoders = {
                uuid.UUID: lambda v: str(v),
                float: lambda v: str(9999.99 if v == float("inf") or v == float("-inf") else v),
                datetime: lambda v: v.isoformat()
            } 
        # This is a list of Field to Exclude during serialization
        default_exclude = {
            "agent",
            "workspace",
            "prompt_manager",
            "chat_model_provider",
            "memory",
            "tool_registry",
            "prompt_settings",
            "systems",
            "configuration",
            "name",
            "description"
        }
        allow_inf_nan = False

    
    def json(self,**dumps_kwargs: Any) -> str:
        logging.Logger(__name__).warning(f'{__qualname__}.json()')
        return super().json(**dumps_kwargs)




class SystemSettings(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True            
        json_encoders = {
                uuid.UUID: lambda v: str(v),
                float: lambda v: str(9999.99 if v == float("inf") or v == float("-inf") else v),
                datetime: lambda v: v.isoformat()
            } 
        allow_inf_nan = False
        default_exclude = {
            "agent",
            "workspace",
            "prompt_manager",
            "chat_model_provider",
            "memory",
            "tool_registry",
            "prompt_settings",
            "systems",
            "configuration",
            "agent_setting_module",
            "agent_setting_class",
            "name",
            "description"
        }
    
    # TODO: #21 https://github.com/ph-ausseil/afaas/issues/21
    created_at : datetime.datetime  =  datetime.datetime.now()
    modified_at : datetime.datetime  = datetime.datetime.now()


    def json(self,**dumps_kwargs: Any) -> str:
        logging.Logger(__name__).warning(f'{__qualname__}.json()')
        return super().json(**dumps_kwargs)
    
S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    class SystemSettings(SystemSettings):
        """A base class for all system settings."""

        name: str
        description: str

        class Config:
            extra = "allow"
            use_enum_values = True
            allow_inf_nan = False

    
    def json(self,**dumps_kwargs: Any) -> str:
        logging.Logger(__name__).warning(f'{__qualname__}.json()')
        return super().json(**dumps_kwargs)


    def __init__(self, settings: S, logger: logging.Logger):
        self._settings = settings
        self._configuration = settings.configuration
        self._logger = logger
