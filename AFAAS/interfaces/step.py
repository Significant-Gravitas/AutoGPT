from AFAAS.core.lib.task import Task
from AFAAS.core.prompting import BasePromptStrategy


from pydantic import BaseModel, Field


import abc
import uuid
from typing import Callable, Type


class StepInterface(abc.ABC, BaseModel):
    class Config(BaseModel.Config):
        arbitrary_types_allowed = True
    strategy : Type[BasePromptStrategy]
    strategy_kwargs : dict
    response_post_process : Callable
    autocorrection : bool = False
    task_id: str = Field(default_factory=lambda: Task.generate_uuid())

    @staticmethod
    def generate_uuid():
        return "S" + str(uuid.uuid4())

    def __eq__(self, __value: object) -> bool:
        return self.strategy == __value.strategy and self.strategy_kwargs == __value.strategy_kwargs and self.response_post_process == __value.response_post_process and self.autocorrection == __value.autocorrection