"""
SAML 2.0 Authentication Provider Integration

This module provides SAML 2.0 authentication support for AutoGPT,
integrating with enterprise identity providers like Okta, Azure AD, ADFS, etc.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.client import Saml2Client
from saml2.config import Config as Saml2Config
from saml2.metadata import entity_descriptor
from saml2.response import StatusResponse

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class SAMLProviderConfig(BaseModel):
    """Configuration for a SAML identity provider."""
    
    # Provider information
    provider_name: str = Field(description="Name of the SAML provider")
    entity_id: str = Field(description="Entity ID for the SP")
    acs_url: str = Field(description="Assertion Consumer Service URL")
    slo_url: Optional[str] = Field(default=None, description="Single Logout URL")
    
    # IdP metadata
    idp_entity_id: str = Field(description="IdP Entity ID")
    idp_sso_url: str = Field(description="IdP Single Sign-On URL")
    idp_slo_url: Optional[str] = Field(default=None, description="IdP Single Logout URL")
    idp_x509_cert: str = Field(description="IdP X.509 certificate")
    
    # Security settings
    want_assertions_signed: bool = True
    want_response_signed: bool = True
    want_assertions_encrypted: bool = False
    want_name_id_encrypted: bool = False
    
    # Attribute mapping
    attribute_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "email": "email",
            "name": "name",
            "first_name": "firstName",
            "last_name": "lastName",
            "username": "username",
            "groups": "groups",
        }
    )
    
    @validator('acs_url', 'slo_url', 'idp_sso_url', 'idp_slo_url')
    def validate_urls(cls, v):
        if v:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
        return v


class SAMLUserAttributes(BaseModel):
    """SAML user attributes extracted from assertion."""
    
    name_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    groups: List[str] = Field(default_factory=list)
    raw_attributes: Dict[str, List[str]] = Field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        """Get the display name for the user."""
        if self.name:
            return self.name
        elif self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.email:
            return self.email
        elif self.username:
            return self.username
        else:
            return self.name_id


class SAMLAuthManager:
    """Manages SAML authentication flows."""
    
    def __init__(self):
        self._providers: Dict[str, SAMLProviderConfig] = {}
        self._clients: Dict[str, Saml2Client] = {}
    
    def register_provider(self, config: SAMLProviderConfig) -> None:
        """Register a SAML provider configuration."""
        self._providers[config.provider_name] = config
        self._clients[config.provider_name] = self._create_saml_client(config)
        logger.info(f"Registered SAML provider: {config.provider_name}")
    
    def get_provider(self, provider_name: str) -> Optional[SAMLProviderConfig]:
        """Get a registered provider configuration."""
        return self._providers.get(provider_name)
    
    def get_client(self, provider_name: str) -> Optional[Saml2Client]:
        """Get a SAML client for a provider."""
        return self._clients.get(provider_name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def _create_saml_client(self, config: SAMLProviderConfig) -> Saml2Client:
        """Create a SAML2Client from configuration."""
        saml_config = {
            "entityid": config.entity_id,
            "description": "AutoGPT Platform",
            "service": {
                "sp": {
                    "name_id_policy_format": None,
                    "endpoints": {
                        "assertion_consumer_service": [
                            (config.acs_url, BINDING_HTTP_POST),
                        ],
                    },
                    "allow_unsolicited": True,
                    "authn_requests_signed": True,
                    "logout_requests_signed": True,
                    "want_assertions_signed": config.want_assertions_signed,
                    "want_response_signed": config.want_response_signed,
                },
            },
            "metadata": {
                "remote": [{
                    "entity_id": config.idp_entity_id,
                    "single_sign_on_service": {
                        "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect": config.idp_sso_url,
                    },
                    "single_logout_service": {
                        "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect": config.idp_slo_url,
                    } if config.idp_slo_url else None,
                    "x509cert": config.idp_x509_cert,
                }],
            },
            "security": {
                "want_assertions_encrypted": config.want_assertions_encrypted,
                "want_name_id_encrypted": config.want_name_id_encrypted,
                "metadata_signed": False,
                "logout_response_signed": True,
                "signature_algorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
                "digest_algorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
            },
            "organization": {
                "name": "AutoGPT Platform",
                "display_name": "AutoGPT",
                "url": "https://agpt.co",
            },
            "contact_person": [
                {
                    "given_name": "Support",
                    "sur_name": "Team",
                    "email_address": "support@agpt.co",
                    "contact_type": "technical",
                },
            ],
        }
        
        # Add SLO endpoint if configured
        if config.slo_url:
            saml_config["service"]["sp"]["endpoints"]["single_logout_service"] = [
                (config.slo_url, BINDING_HTTP_REDIRECT),
            ]
        
        # Remove None values
        saml_config["metadata"]["remote"][0]["single_logout_service"] = (
            saml_config["metadata"]["remote"][0]["single_logout_service"] or None
        )
        
        return Saml2Client(config=Saml2Config().load(saml_config))
    
    def initiate_login(
        self,
        provider_name: str,
        relay_state: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Initiate SAML login flow.
        
        Returns:
            Tuple of (auth_url, request_id)
        """
        client = self.get_client(provider_name)
        if not client:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        # Generate auth request
        req_id, auth_info = client.prepare_for_authenticate(relay_state=relay_state)
        
        # Get the URL to redirect to
        auth_url = auth_info["headers"][0][1].split("url=")[1]
        
        logger.info(f"Initiated SAML login for provider: {provider_name}")
        return auth_url, req_id
    
    def process_response(
        self,
        provider_name: str,
        saml_response: str,
        relay_state: Optional[str] = None
    ) -> Tuple[SAMLUserAttributes, Dict[str, Any]]:
        """
        Process SAML response from IdP.
        
        Returns:
            Tuple of (user_attributes, auth_info)
        """
        client = self.get_client(provider_name)
        if not client:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        # Parse the SAML response
        authn_response = client.parse_authn_request_response(
            saml_response,
            BINDING_HTTP_POST
        )
        
        # Validate the response
        if authn_response.name_id is None:
            raise ValueError("No NameID found in SAML response")
        
        # Extract user attributes
        attributes = self._extract_attributes(authn_response, provider_name)
        
        # Create auth info
        auth_info = {
            "provider": provider_name,
            "name_id": authn_response.name_id,
            "session_index": authn_response.session_index,
            "relay_state": relay_state,
            "not_on_or_after": authn_response.not_on_or_after,
            "authn_instant": authn_response.authn_instant,
        }
        
        logger.info(f"Processed SAML response for user: {attributes.email}")
        return attributes, auth_info
    
    def _extract_attributes(
        self,
        authn_response: StatusResponse,
        provider_name: str
    ) -> SAMLUserAttributes:
        """Extract user attributes from SAML response."""
        config = self.get_provider(provider_name)
        if not config:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        # Get all attributes
        avas = authn_response.ava
        if not avas:
            avas = {}
        
        # Map attributes using provider configuration
        mapped_attrs = {}
        for saml_attr, local_attr in config.attribute_mapping.items():
            if saml_attr in avas:
                values = avas[saml_attr]
                if values:
                    mapped_attrs[local_attr] = values[0] if len(values) == 1 else values
        
        # Create user attributes object
        return SAMLUserAttributes(
            name_id=authn_response.name_id,
            email=mapped_attrs.get("email"),
            name=mapped_attrs.get("name"),
            first_name=mapped_attrs.get("first_name"),
            last_name=mapped_attrs.get("last_name"),
            username=mapped_attrs.get("username"),
            groups=mapped_attrs.get("groups", []) if isinstance(mapped_attrs.get("groups"), list) else [mapped_attrs.get("groups")] if mapped_attrs.get("groups") else [],
            raw_attributes=avas
        )
    
    def initiate_logout(
        self,
        provider_name: str,
        name_id: str,
        session_index: Optional[str] = None,
        relay_state: Optional[str] = None
    ) -> str:
        """
        Initiate SAML logout flow.
        
        Returns:
            Logout URL to redirect to
        """
        client = self.get_client(provider_name)
        if not client:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        # Generate logout request
        logout_info = client.global_logout(name_id, session_index, relay_state)
        
        # Get the URL to redirect to
        logout_url = logout_info["headers"][0][1].split("url=")[1]
        
        logger.info(f"Initiated SAML logout for provider: {provider_name}")
        return logout_url
    
    def process_logout_response(
        self,
        provider_name: str,
        saml_response: str
    ) -> bool:
        """
        Process SAML logout response from IdP.
        
        Returns:
            True if logout was successful
        """
        client = self.get_client(provider_name)
        if not client:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        # Parse the logout response
        client.parse_logout_request_response(saml_response, BINDING_HTTP_POST)
        
        logger.info(f"Processed SAML logout response for provider: {provider_name}")
        return True
    
    def generate_metadata(self, provider_name: str) -> str:
        """Generate SP metadata for a provider."""
        config = self.get_provider(provider_name)
        if not config:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        client = self.get_client(provider_name)
        if not client:
            raise ValueError(f"Client not found for provider: {provider_name}")
        
        metadata = entity_descriptor(client.config)
        return str(metadata)


# Global SAML auth manager instance
_saml_manager: Optional[SAMLAuthManager] = None


def get_saml_manager() -> SAMLAuthManager:
    """Get the global SAML auth manager instance."""
    global _saml_manager
    if _saml_manager is None:
        _saml_manager = SAMLAuthManager()
    return _saml_manager


def configure_saml_providers() -> None:
    """Configure SAML providers from settings."""
    manager = get_saml_manager()
    
    # Example: Configure providers from environment variables or settings
    # This would typically be loaded from a configuration file or database
    
    # Example Okta configuration
    if settings.saml_okta_enabled:
        okta_config = SAMLProviderConfig(
            provider_name="okta",
            entity_id=settings.saml_sp_entity_id or "https://agpt.co/saml",
            acs_url=settings.saml_acs_url or "https://agpt.co/api/auth/saml/acs",
            slo_url=settings.saml_slo_url or "https://agpt.co/api/auth/saml/slo",
            idp_entity_id=settings.saml_okta_entity_id,
            idp_sso_url=settings.saml_okta_sso_url,
            idp_slo_url=settings.saml_okta_slo_url,
            idp_x509_cert=settings.saml_okta_x509_cert,
        )
        manager.register_provider(okta_config)
    
    # Example Azure AD configuration
    if settings.saml_azure_enabled:
        azure_config = SAMLProviderConfig(
            provider_name="azure",
            entity_id=settings.saml_sp_entity_id or "https://agpt.co/saml",
            acs_url=settings.saml_acs_url or "https://agpt.co/api/auth/saml/acs",
            slo_url=settings.saml_slo_url or "https://agpt.co/api/auth/saml/slo",
            idp_entity_id=settings.saml_azure_entity_id,
            idp_sso_url=settings.saml_azure_sso_url,
            idp_slo_url=settings.saml_azure_slo_url,
            idp_x509_cert=settings.saml_azure_x509_cert,
        )
        manager.register_provider(azure_config)


# Initialize SAML providers on module import
try:
    configure_saml_providers()
except Exception as e:
    logger.warning(f"Failed to configure SAML providers: {e}")
