from typing import Any
from pydantic import BaseModel


class AbilityResult(BaseModel):
    success: bool
    message: Any
    data: Any
