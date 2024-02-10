from __future__ import annotations

import abc
import uuid
from typing import Callable, Type

from pydantic import BaseModel, Field, ConfigDict

from AFAAS.interfaces.prompts import AbstractPromptStrategy


class JobInterface(abc.ABC, BaseModel):

    model_config = ConfigDict( arbitrary_types_allowed = True)

    strategy: Type[AbstractPromptStrategy]
    strategy_kwargs: dict
    response_post_process: Callable
    autocorrection: bool = False
    job_id: str = Field(default_factory=lambda: JobInterface.generate_uuid())

    @staticmethod
    def generate_uuid():
        return "J" + str(uuid.uuid4())

    def __eq__(self, __value: object) -> bool:
        return (
            self.strategy == __value.strategy
            and self.strategy_kwargs == __value.strategy_kwargs
            and self.response_post_process == __value.response_post_process
            and self.autocorrection == __value.autocorrection
        )
