import typing

import pydantic


class LibraryAgent(pydantic.BaseModel):
    agent_id: str
    agent_version: int

    name: str
    description: str

    isCreatedByUser: bool

    input_schema: dict[str, typing.Any]
    output_schema: dict[str, typing.Any]
