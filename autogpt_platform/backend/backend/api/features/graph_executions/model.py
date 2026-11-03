import pydantic


class ExecutionShareRequest(pydantic.BaseModel):
    """Optional request body for share endpoint."""

    pass  # Empty body is fine


class ExecutionShareResponse(pydantic.BaseModel):
    """Response from share endpoints."""

    share_url: str
    share_token: str
