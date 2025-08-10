from typing import Optional

from pydantic import BaseModel, Field


class TurnstileVerifyRequest(BaseModel):
    """Request model for verifying a Turnstile token."""

    token: str = Field(description="The Turnstile token to verify")
    action: Optional[str] = Field(
        default=None, description="The action that the user is attempting to perform"
    )


class TurnstileVerifyResponse(BaseModel):
    """Response model for the Turnstile verification endpoint."""

    success: bool = Field(description="Whether the token verification was successful")
    error: Optional[str] = Field(
        default=None, description="Error message if verification failed"
    )
    challenge_timestamp: Optional[str] = Field(
        default=None, description="Timestamp of the challenge (ISO format)"
    )
    hostname: Optional[str] = Field(
        default=None, description="Hostname of the site where the challenge was solved"
    )
    action: Optional[str] = Field(
        default=None, description="The action associated with this verification"
    )
