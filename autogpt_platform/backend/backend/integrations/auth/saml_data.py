"""
SAML Authentication Data Layer

Handles database operations for SAML providers, users, and authentication requests.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import uuid

from prisma import Json
from prisma.enums import APIKeyPermission
from prisma.models import SAMLProvider, SAMLUser, SAMLAuthRequest, User
from prisma.types import (
    SAMLProviderCreateInput,
    SAMLProviderUpdateInput,
    SAMLAuthRequestCreateInput,
)

from backend.data.auth.base import APIAuthorizationInfo
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class SAMLProviderData:
    """Data layer for SAML provider operations."""
    
    @staticmethod
    async def create_provider(data: SAMLProviderCreateInput) -> SAMLProvider:
        """Create a new SAML provider."""
        provider = await SAMLProvider.prisma().create(data=data)
        logger.info(f"Created SAML provider: {provider.providerName}")
        return provider
    
    @staticmethod
    async def get_provider(provider_id: str) -> Optional[SAMLProvider]:
        """Get a SAML provider by ID."""
        return await SAMLProvider.prisma().find_unique(where={"id": provider_id})
    
    @staticmethod
    async def get_provider_by_name(provider_name: str) -> Optional[SAMLProvider]:
        """Get a SAML provider by name."""
        return await SAMLProvider.prisma().find_first(
            where={"providerName": provider_name}
        )
    
    @staticmethod
    async def list_providers(enabled_only: bool = True) -> List[SAMLProvider]:
        """List all SAML providers."""
        where_clause = {"enabled": enabled_only} if enabled_only else {}
        return await SAMLProvider.prisma().find_many(
            where=where_clause,
            order={"createdAt": "asc"}
        )
    
    @staticmethod
    async def update_provider(
        provider_id: str,
        data: SAMLProviderUpdateInput
    ) -> Optional[SAMLProvider]:
        """Update a SAML provider."""
        provider = await SAMLProvider.prisma().update(
            where={"id": provider_id},
            data=data
        )
        if provider:
            logger.info(f"Updated SAML provider: {provider.providerName}")
        return provider
    
    @staticmethod
    async def delete_provider(provider_id: str) -> bool:
        """Delete a SAML provider."""
        try:
            await SAMLProvider.prisma().delete(where={"id": provider_id})
            logger.info(f"Deleted SAML provider: {provider_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete SAML provider {provider_id}: {e}")
            return False
    
    @staticmethod
    async def enable_provider(provider_id: str) -> Optional[SAMLProvider]:
        """Enable a SAML provider."""
        return await SAMLProviderData.update_provider(
            provider_id,
            data={"enabled": True}
        )
    
    @staticmethod
    async def disable_provider(provider_id: str) -> Optional[SAMLProvider]:
        """Disable a SAML provider."""
        return await SAMLProviderData.update_provider(
            provider_id,
            data={"enabled": False}
        )


class SAMLUserData:
    """Data layer for SAML user operations."""
    
    @staticmethod
    async def create_or_update_user(
        provider_id: str,
        name_id: str,
        user_id: str,
        attributes: Dict[str, Any],
        session_index: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> SAMLUser:
        """Create or update a SAML user record."""
        # Check if user already exists
        existing = await SAMLUser.prisma().find_first(
            where={
                "providerId": provider_id,
                "nameId": name_id
            }
        )
        
        user_data = {
            "providerId": provider_id,
            "nameId": name_id,
            "userId": user_id,
            "email": attributes.get("email"),
            "name": attributes.get("name"),
            "firstName": attributes.get("first_name"),
            "lastName": attributes.get("last_name"),
            "username": attributes.get("username"),
            "groups": attributes.get("groups", []),
            "rawAttributes": Json(attributes.get("raw_attributes", {})),
            "sessionIndex": session_index,
            "expiresAt": expires_at,
            "active": True,
            "lastLoginAt": datetime.now(timezone.utc),
        }
        
        if existing:
            # Update existing user
            user = await SAMLUser.prisma().update(
                where={"id": existing.id},
                data=user_data
            )
            logger.info(f"Updated SAML user: {user.email}")
        else:
            # Create new user
            user = await SAMLUser.prisma().create(data=user_data)
            logger.info(f"Created SAML user: {user.email}")
        
        return user
    
    @staticmethod
    async def get_user(user_id: str) -> Optional[SAMLUser]:
        """Get a SAML user by ID."""
        return await SAMLUser.prisma().find_unique(
            where={"id": user_id},
            include={"provider": True, "user": True}
        )
    
    @staticmethod
    async def get_user_by_name_id(
        provider_id: str,
        name_id: str
    ) -> Optional[SAMLUser]:
        """Get a SAML user by provider and NameID."""
        return await SAMLUser.prisma().find_first(
            where={
                "providerId": provider_id,
                "nameId": name_id
            },
            include={"provider": True, "user": True}
        )
    
    @staticmethod
    async def get_users_by_provider(
        provider_id: str,
        active_only: bool = True
    ) -> List[SAMLUser]:
        """Get all users for a provider."""
        where_clause = {
            "providerId": provider_id,
            "active": active_only
        } if active_only else {"providerId": provider_id}
        
        return await SAMLUser.prisma().find_many(
            where=where_clause,
            include={"provider": True, "user": True},
            order={"lastLoginAt": "desc"}
        )
    
    @staticmethod
    async def get_users_by_internal_user(
        user_id: str,
        active_only: bool = True
    ) -> List[SAMLUser]:
        """Get all SAML identities for an internal user."""
        where_clause = {
            "userId": user_id,
            "active": active_only
        } if active_only else {"userId": user_id}
        
        return await SAMLUser.prisma().find_many(
            where=where_clause,
            include={"provider": True},
            order={"lastLoginAt": "desc"}
        )
    
    @staticmethod
    async def deactivate_user(user_id: str) -> Optional[SAMLUser]:
        """Deactivate a SAML user."""
        user = await SAMLUser.prisma().update(
            where={"id": user_id},
            data={"active": False}
        )
        if user:
            logger.info(f"Deactivated SAML user: {user.email}")
        return user
    
    @staticmethod
    async def reactivate_user(user_id: str) -> Optional[SAMLUser]:
        """Reactivate a SAML user."""
        user = await SAMLUser.prisma().update(
            where={"id": user_id},
            data={"active": True}
        )
        if user:
            logger.info(f"Reactivated SAML user: {user.email}")
        return user
    
    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """Delete a SAML user."""
        try:
            await SAMLUser.prisma().delete(where={"id": user_id})
            logger.info(f"Deleted SAML user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete SAML user {user_id}: {e}")
            return False
    
    @staticmethod
    async def cleanup_expired_sessions() -> int:
        """Clean up expired SAML sessions."""
        now = datetime.now(timezone.utc)
        result = await SAMLUser.prisma().update_many(
            where={
                "expiresAt": {"lt": now},
                "active": True
            },
            data={"active": False}
        )
        
        if result.count > 0:
            logger.info(f"Deactivated {result.count} expired SAML sessions")
        
        return result.count


class SAMLAuthRequestData:
    """Data layer for SAML authentication request tracking."""
    
    @staticmethod
    async def create_request(
        provider_id: str,
        request_id: str,
        relay_state: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> SAMLAuthRequest:
        """Create a new SAML auth request record."""
        data = SAMLAuthRequestCreateInput(
            providerId=provider_id,
            requestId=request_id,
            relayState=relay_state,
            userId=user_id,
            status="pending"
        )
        
        request = await SAMLAuthRequest.prisma().create(data=data)
        logger.debug(f"Created SAML auth request: {request.id}")
        return request
    
    @staticmethod
    async def get_request(request_id: str) -> Optional[SAMLAuthRequest]:
        """Get a SAML auth request by ID."""
        return await SAMLAuthRequest.prisma().find_unique(
            where={"id": request_id},
            include={"provider": True, "user": True}
        )
    
    @staticmethod
    async def get_request_by_saml_id(saml_request_id: str) -> Optional[SAMLAuthRequest]:
        """Get a SAML auth request by SAML request ID."""
        return await SAMLAuthRequest.prisma().find_first(
            where={"requestId": saml_request_id},
            include={"provider": True, "user": True}
        )
    
    @staticmethod
    async def complete_request(
        request_id: str,
        user_id: Optional[str] = None
    ) -> Optional[SAMLAuthRequest]:
        """Mark a SAML auth request as completed."""
        data = {
            "status": "completed",
            "completedAt": datetime.now(timezone.utc)
        }
        
        if user_id:
            data["userId"] = user_id
        
        request = await SAMLAuthRequest.prisma().update(
            where={"id": request_id},
            data=data
        )
        
        if request:
            logger.debug(f"Completed SAML auth request: {request.id}")
        
        return request
    
    @staticmethod
    async def fail_request(request_id: str) -> Optional[SAMLAuthRequest]:
        """Mark a SAML auth request as failed."""
        request = await SAMLAuthRequest.prisma().update(
            where={"id": request_id},
            data={
                "status": "failed",
                "completedAt": datetime.now(timezone.utc)
            }
        )
        
        if request:
            logger.debug(f"Failed SAML auth request: {request.id}")
        
        return request
    
    @staticmethod
    async def cleanup_old_requests(hours: int = 24) -> int:
        """Clean up old SAML auth requests."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        result = await SAMLAuthRequest.prisma().delete_many(
            where={
                "createdAt": {"lt": cutoff},
                "status": {"in": ["completed", "failed"]}
            }
        )
        
        if result.count > 0:
            logger.info(f"Cleaned up {result.count} old SAML auth requests")
        
        return result.count


class SAMLAuthService:
    """High-level SAML authentication service."""
    
    @staticmethod
    async def authenticate_user(
        provider_name: str,
        name_id: str,
        attributes: Dict[str, Any],
        session_index: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> Tuple[Optional[User], APIAuthorizationInfo]:
        """
        Authenticate a user via SAML and create/update their account.
        
        Returns:
            Tuple of (user, auth_info)
        """
        # Get provider
        provider = await SAMLProviderData.get_provider_by_name(provider_name)
        if not provider or not provider.enabled:
            raise ValueError(f"SAML provider not found or disabled: {provider_name}")
        
        # Find or create user
        email = attributes.get("email")
        if not email:
            raise ValueError("Email is required from SAML attributes")
        
        # Try to find existing user by email
        user = await User.prisma().find_unique(where={"email": email})
        
        if not user:
            # Create new user
            user = await User.prisma().create(
                data={
                    "id": str(uuid.uuid4()),
                    "email": email,
                    "name": attributes.get("name"),
                    "metadata": Json({
                        "auth_method": "saml",
                        "provider": provider_name,
                        "created_via": f"saml:{provider_name}"
                    })
                }
            )
            logger.info(f"Created new user via SAML: {user.email}")
        
        # Create or update SAML user record
        await SAMLUserData.create_or_update_user(
            provider_id=provider.id,
            name_id=name_id,
            user_id=user.id,
            attributes=attributes,
            session_index=session_index,
            expires_at=expires_at
        )
        
        # Create authorization info
        auth_info = APIAuthorizationInfo(
            user_id=user.id,
            scopes=[APIKeyPermission.USER_READ, APIKeyPermission.USER_WRITE],
            type="oauth",  # Using oauth type for SAML as well
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at
        )
        
        return user, auth_info
    
    @staticmethod
    async def logout_user(
        provider_name: str,
        name_id: str
    ) -> bool:
        """Logout a SAML user by deactivating their session."""
        provider = await SAMLProviderData.get_provider_by_name(provider_name)
        if not provider:
            return False
        
        saml_user = await SAMLUserData.get_user_by_name_id(
            provider.id,
            name_id
        )
        
        if saml_user:
            await SAMLUserData.deactivate_user(saml_user.id)
            logger.info(f"Logged out SAML user: {saml_user.email}")
            return True
        
        return False
    
    @staticmethod
    async def get_user_providers(user_id: str) -> List[Dict[str, Any]]:
        """Get all SAML providers linked to a user."""
        saml_users = await SAMLUserData.get_users_by_internal_user(user_id)
        
        providers = []
        for saml_user in saml_users:
            providers.append({
                "provider_id": saml_user.provider.id,
                "provider_name": saml_user.provider.providerName,
                "display_name": saml_user.provider.displayName,
                "email": saml_user.email,
                "last_login": saml_user.lastLoginAt,
                "active": saml_user.active
            })
        
        return providers
