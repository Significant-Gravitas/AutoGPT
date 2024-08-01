import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserBase(BaseModel):
    id: str = Field(None, description="The unique Supabase ID of the user")


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    id: str = Field(..., description="The unique Supabase identifier of the user")


class UserResponse(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime


class UsersListResponse(BaseModel):
    users: List[UserResponse]
