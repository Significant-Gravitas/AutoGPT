"""
SAML Authentication API Routes

Provides HTTP endpoints for SAML authentication flows including:
- Initiate login
- Process SAML response
- Initiate logout
- Process logout response
- Provider management
- Metadata endpoint
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from backend.integrations.auth.saml import SAMLAuthManager, get_saml_manager
from backend.integrations.auth.saml_data import (
    SAMLAuthService,
    SAMLAuthRequestData,
    SAMLProviderData,
    SAMLUserData,
)
from backend.server.utils import get_current_user, get_user_id
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter(prefix="/api/auth/saml", tags=["saml"])


# Request/Response Models
class LoginRequest(BaseModel):
    provider_name: str = Field(..., description="SAML provider name")
    relay_state: Optional[str] = Field(None, description="Optional relay state")


class LoginResponse(BaseModel):
    auth_url: str = Field(..., description="URL to redirect to for authentication")
    request_id: str = Field(..., description="SAML request ID for tracking")


class ProviderResponse(BaseModel):
    id: str
    provider_name: str
    display_name: str
    enabled: bool
    created_at: datetime


class CreateUserRequest(BaseModel):
    provider_name: str
    name_id: str
    attributes: Dict[str, Any]
    session_index: Optional[str] = None
    expires_at: Optional[datetime] = None


# Helper Functions
async def get_saml_manager() -> SAMLAuthManager:
    """Get the SAML manager instance."""
    return get_saml_manager()


# Routes
@router.get("/providers", response_model=list[ProviderResponse])
async def list_providers(
    enabled_only: bool = Query(True, description="List only enabled providers")
):
    """List all configured SAML providers."""
    providers = await SAMLProviderData.list_providers(enabled_only=enabled_only)
    
    return [
        ProviderResponse(
            id=p.id,
            provider_name=p.providerName,
            display_name=p.displayName,
            enabled=p.enabled,
            created_at=p.createdAt
        )
        for p in providers
    ]


@router.post("/login", response_model=LoginResponse)
async def initiate_login(
    request: LoginRequest,
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """Initiate SAML login flow."""
    try:
        # Check if provider exists and is enabled
        provider = await SAMLProviderData.get_provider_by_name(request.provider_name)
        if not provider or not provider.enabled:
            raise HTTPException(
                status_code=404,
                detail=f"SAML provider not found or disabled: {request.provider_name}"
            )
        
        # Initiate login
        auth_url, request_id = saml_manager.initiate_login(
            provider_name=request.provider_name,
            relay_state=request.relay_state
        )
        
        # Track the request
        await SAMLAuthRequestData.create_request(
            provider_id=provider.id,
            request_id=request_id,
            relay_state=request.relay_state
        )
        
        return LoginResponse(auth_url=auth_url, request_id=request_id)
        
    except Exception as e:
        logger.error(f"Failed to initiate SAML login: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate login")


@router.get("/login/{provider_name}")
async def initiate_login_get(
    provider_name: str,
    relay_state: Optional[str] = Query(None),
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """Initiate SAML login via GET request."""
    try:
        # Check if provider exists and is enabled
        provider = await SAMLProviderData.get_provider_by_name(provider_name)
        if not provider or not provider.enabled:
            raise HTTPException(
                status_code=404,
                detail=f"SAML provider not found or disabled: {provider_name}"
            )
        
        # Initiate login
        auth_url, request_id = saml_manager.initiate_login(
            provider_name=provider_name,
            relay_state=relay_state
        )
        
        # Track the request
        await SAMLAuthRequestData.create_request(
            provider_id=provider.id,
            request_id=request_id,
            relay_state=relay_state
        )
        
        # Redirect to IdP
        return RedirectResponse(url=auth_url)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate SAML login: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate login")


@router.post("/acs")
async def process_assertion(
    request: Request,
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """
    Process SAML Assertion (Assertion Consumer Service).
    This endpoint receives the SAML response from the IdP.
    """
    try:
        # Get SAML response from form data
        form_data = await request.form()
        saml_response = form_data.get("SAMLResponse")
        relay_state = form_data.get("RelayState")
        
        if not saml_response:
            raise HTTPException(status_code=400, detail="Missing SAMLResponse")
        
        # Extract provider from request ID (this would need to be tracked)
        # For now, we'll parse the response to get the issuer
        # In production, you should track the request ID to provider mapping
        
        # Process response (this is simplified - you'd need to determine the provider)
        # For now, let's assume we can extract the provider from the response
        provider_name = "okta"  # This should be determined from the response
        
        # Get provider
        provider = await SAMLProviderData.get_provider_by_name(provider_name)
        if not provider:
            raise HTTPException(status_code=400, detail="Invalid SAML provider")
        
        # Process SAML response
        attributes, auth_info = saml_manager.process_response(
            provider_name=provider_name,
            saml_response=saml_response,
            relay_state=relay_state
        )
        
        # Authenticate user
        user, auth_info = await SAMLAuthService.authenticate_user(
            provider_name=provider_name,
            name_id=auth_info["name_id"],
            session_index=auth_info.get("session_index"),
            expires_at=auth_info.get("not_on_or_after"),
            attributes=attributes.dict()
        )
        
        # Update auth request
        if auth_info.get("request_id"):
            await SAMLAuthRequestData.complete_request(
                request_id=auth_info["request_id"],
                user_id=user.id
            )
        
        # Create session/token (this would integrate with your auth system)
        # For now, return success
        return {
            "success": True,
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "relay_state": relay_state
        }
        
    except Exception as e:
        logger.error(f"Failed to process SAML assertion: {e}")
        raise HTTPException(status_code=400, detail="Failed to process SAML response")


@router.get("/logout/{provider_name}")
async def initiate_logout(
    provider_name: str,
    user_id: str = Depends(get_user_id),
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """Initiate SAML logout flow."""
    try:
        # Get user's SAML identity for this provider
        saml_users = await SAMLUserData.get_users_by_internal_user(user_id)
        saml_user = None
        
        for su in saml_users:
            if su.provider.providerName == provider_name:
                saml_user = su
                break
        
        if not saml_user or not saml_user.active:
            raise HTTPException(
                status_code=400,
                detail="No active SAML session found for this provider"
            )
        
        # Initiate logout
        logout_url = saml_manager.initiate_logout(
            provider_name=provider_name,
            name_id=saml_user.nameId,
            session_index=saml_user.sessionIndex
        )
        
        # Deactivate the session
        await SAMLUserData.deactivate_user(saml_user.id)
        
        # Redirect to IdP logout
        return RedirectResponse(url=logout_url)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate SAML logout: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate logout")


@router.post("/slo")
async def process_logout_response(
    request: Request,
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """
    Process SAML Logout Response (Single Logout Service).
    This endpoint receives the logout response from the IdP.
    """
    try:
        # Get SAML response from form data
        form_data = await request.form()
        saml_response = form_data.get("SAMLResponse")
        
        if not saml_response:
            raise HTTPException(status_code=400, detail="Missing SAMLResponse")
        
        # Process logout response
        # In production, you'd need to determine which provider this is from
        provider_name = "okta"  # This should be determined from the response
        
        success = saml_manager.process_logout_response(
            provider_name=provider_name,
            saml_response=saml_response
        )
        
        if success:
            return {"success": True, "message": "Logout successful"}
        else:
            return {"success": False, "message": "Logout failed"}
            
    except Exception as e:
        logger.error(f"Failed to process SAML logout response: {e}")
        raise HTTPException(status_code=400, detail="Failed to process logout response")


@router.get("/metadata/{provider_name}")
async def get_metadata(
    provider_name: str,
    saml_manager: SAMLAuthManager = Depends(get_saml_manager)
):
    """Get SP metadata for a SAML provider."""
    try:
        metadata = saml_manager.generate_metadata(provider_name)
        
        # Return as XML
        return Response(
            content=metadata,
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename=metadata-{provider_name}.xml"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metadata")


@router.get("/user/providers")
async def get_user_saml_providers(
    user_id: str = Depends(get_user_id)
):
    """Get all SAML providers linked to the current user."""
    providers = await SAMLAuthService.get_user_providers(user_id)
    return {"providers": providers}


@router.delete("/user/providers/{provider_name}")
async def unlink_saml_provider(
    provider_name: str,
    user_id: str = Depends(get_user_id)
):
    """Unlink a SAML provider from the current user."""
    try:
        # Get provider
        provider = await SAMLProviderData.get_provider_by_name(provider_name)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        # Find and deactivate user's SAML identity
        saml_users = await SAMLUserData.get_users_by_internal_user(user_id)
        saml_user = None
        
        for su in saml_users:
            if su.providerId == provider.id:
                saml_user = su
                break
        
        if not saml_user:
            raise HTTPException(
                status_code=404,
                detail="SAML identity not found for this provider"
            )
        
        # Deactivate the SAML user
        await SAMLUserData.deactivate_user(saml_user.id)
        
        return {"success": True, "message": "Provider unlinked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unlink SAML provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to unlink provider")


# Admin Routes (for managing providers)
@router.post("/admin/providers")
async def create_provider(
    provider_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Create a new SAML provider (admin only)."""
    # Check if user is admin (implement your own check)
    if not current_user.metadata.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Create provider
        provider = await SAMLProviderData.create_provider(provider_data)
        
        # Register with SAML manager
        saml_manager = get_saml_manager()
        from backend.integrations.auth.saml import SAMLProviderConfig
        
        config = SAMLProviderConfig(
            provider_name=provider.providerName,
            entity_id=provider.entityId,
            acs_url=provider.acsUrl,
            slo_url=provider.sloUrl,
            idp_entity_id=provider.idpEntityId,
            idp_sso_url=provider.idpSsoUrl,
            idp_slo_url=provider.idpSloUrl,
            idp_x509_cert=provider.idpX509Cert,
            want_assertions_signed=provider.wantAssertionsSigned,
            want_response_signed=provider.wantResponseSigned,
            want_assertions_encrypted=provider.wantAssertionsEncrypted,
            want_name_id_encrypted=provider.wantNameIdEncrypted,
            attribute_mapping=dict(provider.attributeMapping)
        )
        
        saml_manager.register_provider(config)
        
        return {"success": True, "provider_id": provider.id}
        
    except Exception as e:
        logger.error(f"Failed to create SAML provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to create provider")


@router.put("/admin/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    provider_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Update a SAML provider (admin only)."""
    # Check if user is admin
    if not current_user.metadata.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        provider = await SAMLProviderData.update_provider(provider_id, provider_data)
        
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update SAML provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to update provider")


@router.delete("/admin/providers/{provider_id}")
async def delete_provider(
    provider_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a SAML provider (admin only)."""
    # Check if user is admin
    if not current_user.metadata.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        success = await SAMLProviderData.delete_provider(provider_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SAML provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete provider")
