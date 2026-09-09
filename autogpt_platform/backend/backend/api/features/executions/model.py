import pydantic


class ShareRequest(pydantic.BaseModel):
    """Optional request body for share endpoint."""

    pass  # Empty body is fine


class ShareResponse(pydantic.BaseModel):
    """Response from share endpoints."""

    share_url: str
    share_token: str
