"""Public request and response models for partner embedding."""

from pydantic import BaseModel, EmailStr, Field


class ProvisionPartnerIdentityRequest(BaseModel):
    partner_id: str = Field(min_length=1, max_length=80)
    external_subject: str = Field(min_length=1, max_length=255)
    external_account_id: str = Field(min_length=1, max_length=255)
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    account_name: str = Field(min_length=1, max_length=120)
    is_admin: bool = False


class ProvisionPartnerIdentityResponse(BaseModel):
    user_id: str
    organization_id: str
    team_id: str


class ShadowIdentityIDs(BaseModel):
    user_id: str
    organization_id: str
    team_id: str


class CreateEmbedSessionResponse(BaseModel):
    id: str
    created_at: str


class EmbedTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=64_000)
    message_id: str | None = Field(default=None, max_length=64)
