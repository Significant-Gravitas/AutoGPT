from pydantic import BaseModel


class AbilityResult(BaseModel):
    success: bool
    message: str
    data: str
