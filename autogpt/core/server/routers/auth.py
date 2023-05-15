from fastapi import APIRouter, Depends, HTTPException, status

from autogpt.core.server.models.user import UserCreate
from autogpt.core.server.services import auth_service
from autogpt.core.server.services.auth_service import get_current_user

router = APIRouter()


@router.post("/register", response_model=UserCreate)
async def register_user(user: UserCreate):
    """Register a new user"""
    return await auth_service.register_user(user)


@router.post("/login", response_model=UserCreate)
async def login_user(user: UserCreate):
    """Login a user"""
    return await auth_service.login_user(user)


@router.post("/logout")
async def logout_user(current_user=Depends(get_current_user)):
    """Logout a user"""
    return await auth_service.logout_user(current_user)


@router.post("/password-reset")
async def initiate_password_reset(current_user=Depends(get_current_user)):
    """Initiate a password reset"""
    return await auth_service.initiate_password_reset(current_user)


@router.post("/password-reset/confirm")
async def confirm_password_reset(current_user=Depends(get_current_user)):
    """Confirm a password reset"""
    return await auth_service.confirm_password_reset(current_user)
