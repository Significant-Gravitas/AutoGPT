"""What every model in a page design shares."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Ref = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )
