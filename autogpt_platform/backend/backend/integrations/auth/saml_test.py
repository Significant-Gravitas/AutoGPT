"""
Tests for SAML Authentication Integration
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from backend.integrations.auth.saml import (
    SAMLAuthManager,
    SAMLProviderConfig,
    SAMLUserAttributes,
)
from backend.integrations.auth.saml_data import (
    SAMLAuthService,
    SAMLAuthRequestData,
    SAMLUserData,
)
from backend.data.auth.base import APIAuthorizationInfo


class TestSAMLProviderConfig:
    """Test SAMLProviderConfig validation."""
    
    def test_valid_config(self):
        """Test creating a valid provider configuration."""
        config = SAMLProviderConfig(
            provider_name="okta",
            entity_id="https://agpt.co/saml",
            acs_url="https://agpt.co/api/auth/saml/acs",
            idp_entity_id="https://okta.com/entity",
            idp_sso_url="https://okta.com/sso",
            idp_x509_cert="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
        )
        
        assert config.provider_name == "okta"
        assert config.want_assertions_signed is True
        assert "email" in config.attribute_mapping
    
    def test_invalid_url(self):
        """Test validation of URLs."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            SAMLProviderConfig(
                provider_name="test",
                entity_id="invalid-url",
                acs_url="https://agpt.co/api/auth/saml/acs",
                idp_entity_id="https://idp.com",
                idp_sso_url="https://idp.com/sso",
                idp_x509_cert="cert"
            )


class TestSAMLAuthManager:
    """Test SAMLAuthManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a SAMLAuthManager instance."""
        return SAMLAuthManager()
    
    @pytest.fixture
    def provider_config(self):
        """Create a test provider configuration."""
        return SAMLProviderConfig(
            provider_name="test",
            entity_id="https://test.com/saml",
            acs_url="https://test.com/api/auth/saml/acs",
            idp_entity_id="https://idp.com/entity",
            idp_sso_url="https://idp.com/sso",
            idp_x509_cert="-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----"
        )
    
    def test_register_provider(self, manager, provider_config):
        """Test registering a SAML provider."""
        manager.register_provider(provider_config)
        
        assert provider_config.provider_name in manager._providers
        assert provider_config.provider_name in manager._clients
        assert manager.get_provider("test") is provider_config
    
    def test_list_providers(self, manager, provider_config):
        """Test listing registered providers."""
        manager.register_provider(provider_config)
        
        providers = manager.list_providers()
        assert "test" in providers
    
    @patch('backend.integrations.auth.saml.Saml2Client')
    def test_initiate_login(self, mock_client_class, manager, provider_config):
        """Test initiating SAML login."""
        # Mock the SAML client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.prepare_for_authenticate.return_value = ("req123", {"headers": [("Location", "url=https://idp.com/sso")]})
        
        manager.register_provider(provider_config)
        
        auth_url, request_id = manager.initiate_login("test", relay_state="test-state")
        
        assert auth_url == "https://idp.com/sso"
        assert request_id == "req123"
        mock_client.prepare_for_authenticate.assert_called_once_with(relay_state="test-state")
    
    def test_initiate_login_unknown_provider(self, manager):
        """Test initiating login with unknown provider."""
        with pytest.raises(ValueError, match="Provider not registered"):
            manager.initiate_login("unknown")
    
    @patch('backend.integrations.auth.saml.Saml2Client')
    def test_process_response(self, mock_client_class, manager, provider_config):
        """Test processing SAML response."""
        # Mock the SAML client and response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.name_id = "user123"
        mock_response.session_index = "session123"
        mock_response.ava = {
            "email": ["user@test.com"],
            "name": ["Test User"],
            "firstName": ["Test"],
            "lastName": ["User"]
        }
        mock_client.parse_authn_request_response.return_value = mock_response
        
        manager.register_provider(provider_config)
        
        attributes, auth_info = manager.process_response(
            "test",
            "fake_saml_response",
            relay_state="test-state"
        )
        
        assert isinstance(attributes, SAMLUserAttributes)
        assert attributes.name_id == "user123"
        assert attributes.email == "user@test.com"
        assert attributes.name == "Test User"
        assert auth_info["provider"] == "test"
        assert auth_info["relay_state"] == "test-state"
    
    @patch('backend.integrations.auth.saml.Saml2Client')
    def test_initiate_logout(self, mock_client_class, manager, provider_config):
        """Test initiating SAML logout."""
        # Mock the SAML client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.global_logout.return_value = {"headers": [("Location", "url=https://idp.com/slo")]}
        
        manager.register_provider(provider_config)
        
        logout_url = manager.initiate_logout("test", "user123", "session123")
        
        assert logout_url == "https://idp.com/slo"
        mock_client.global_logout.assert_called_once_with("user123", "session123", None)
    
    @patch('backend.integrations.auth.saml.Saml2Client')
    def test_generate_metadata(self, mock_client_class, manager, provider_config):
        """Test generating SP metadata."""
        # Mock the SAML client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.config = MagicMock()
        
        from saml2.metadata import entity_descriptor as mock_entity_descriptor
        mock_metadata = MagicMock()
        mock_metadata.__str__ = lambda: "<EntityDescriptor>...</EntityDescriptor>"
        with patch('backend.integrations.auth.saml.entity_descriptor', mock_entity_descriptor):
            manager.register_provider(provider_config)
            
            metadata = manager.generate_metadata("test")
            
            assert "<EntityDescriptor>" in metadata
            mock_client.config.entityid = provider_config.entity_id


class TestSAMLUserData:
    """Test SAMLUserData operations."""
    
    @pytest.mark.asyncio
    async def test_create_user(self):
        """Test creating a SAML user."""
        # Mock the database operations
        with patch('backend.integrations.auth.saml_data.SAMLUser.prisma') as mock_prisma:
            mock_prisma.find_first.return_value = None  # User doesn't exist
            mock_user = MagicMock()
            mock_user.id = "user123"
            mock_user.email = "test@example.com"
            mock_prisma.create.return_value = mock_user
            
            result = await SAMLUserData.create_or_update_user(
                provider_id="provider123",
                name_id="name123",
                user_id="internal123",
                attributes={
                    "email": "test@example.com",
                    "name": "Test User",
                    "groups": ["users", "admins"]
                }
            )
            
            assert result == mock_user
            mock_prisma.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_existing_user(self):
        """Test updating an existing SAML user."""
        with patch('backend.integrations.auth.saml_data.SAMLUser.prisma') as mock_prisma:
            # Mock existing user
            mock_existing = MagicMock()
            mock_existing.id = "user123"
            mock_prisma.find_first.return_value = mock_existing
            
            # Mock updated user
            mock_updated = MagicMock()
            mock_updated.email = "updated@example.com"
            mock_prisma.update.return_value = mock_updated
            
            result = await SAMLUserData.create_or_update_user(
                provider_id="provider123",
                name_id="name123",
                user_id="internal123",
                attributes={"email": "updated@example.com"}
            )
            
            assert result == mock_updated
            mock_prisma.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_by_name_id(self):
        """Test getting user by provider and NameID."""
        with patch('backend.integrations.auth.saml_data.SAMLUser.prisma') as mock_prisma:
            mock_user = MagicMock()
            mock_user.nameId = "name123"
            mock_user.email = "test@example.com"
            mock_prisma.find_first.return_value = mock_user
            
            result = await SAMLUserData.get_user_by_name_id("provider123", "name123")
            
            assert result == mock_user
            mock_prisma.find_first.assert_called_once_with(
                where={
                    "providerId": "provider123",
                    "nameId": "name123"
                },
                include={"provider": True, "user": True}
            )
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """Test cleaning up expired sessions."""
        with patch('backend.integrations.auth.saml_data.SAMLUser.prisma') as mock_prisma:
            mock_result = MagicMock()
            mock_result.count = 5
            mock_prisma.update_many.return_value = mock_result
            
            count = await SAMLUserData.cleanup_expired_sessions()
            
            assert count == 5
            mock_prisma.update_many.assert_called_once()


class TestSAMLAuthService:
    """Test SAMLAuthService high-level operations."""
    
    @pytest.mark.asyncio
    async def test_authenticate_user_new_user(self):
        """Test authenticating a new user via SAML."""
        with patch('backend.integrations.auth.saml_data.SAMLProviderData.get_provider_by_name') as mock_get_provider, \
             patch('backend.integrations.auth.saml_data.User.prisma') as mock_user_prisma, \
             patch('backend.integrations.auth.saml_data.SAMLUserData.create_or_update_user') as mock_create_saml:
            
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.id = "provider123"
            mock_provider.enabled = True
            mock_get_provider.return_value = mock_provider
            
            # Mock user lookup (not found)
            mock_user_prisma.find_unique.return_value = None
            
            # Mock user creation
            mock_new_user = MagicMock()
            mock_new_user.id = "user123"
            mock_new_user.email = "test@example.com"
            mock_user_prisma.create.return_value = mock_new_user
            
            # Mock SAML user creation
            mock_saml_user = MagicMock()
            mock_create_saml.return_value = mock_saml_user
            
            user, auth_info = await SAMLAuthService.authenticate_user(
                provider_name="okta",
                name_id="name123",
                attributes={
                    "email": "test@example.com",
                    "name": "Test User"
                }
            )
            
            assert user == mock_new_user
            assert isinstance(auth_info, APIAuthorizationInfo)
            assert auth_info.user_id == "user123"
            assert auth_info.type == "oauth"
    
    @pytest.mark.asyncio
    async def test_authenticate_user_existing_user(self):
        """Test authenticating an existing user via SAML."""
        with patch('backend.integrations.auth.saml_data.SAMLProviderData.get_provider_by_name') as mock_get_provider, \
             patch('backend.integrations.auth.saml_data.User.prisma') as mock_user_prisma, \
             patch('backend.integrations.auth.saml_data.SAMLUserData.create_or_update_user') as mock_create_saml:
            
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.id = "provider123"
            mock_provider.enabled = True
            mock_get_provider.return_value = mock_provider
            
            # Mock existing user
            mock_existing_user = MagicMock()
            mock_existing_user.id = "user123"
            mock_existing_user.email = "test@example.com"
            mock_user_prisma.find_unique.return_value = mock_existing_user
            
            # Mock SAML user update
            mock_saml_user = MagicMock()
            mock_create_saml.return_value = mock_saml_user
            
            user, auth_info = await SAMLAuthService.authenticate_user(
                provider_name="okta",
                name_id="name123",
                attributes={
                    "email": "test@example.com",
                    "name": "Test User"
                }
            )
            
            assert user == mock_existing_user
            assert auth_info.user_id == "user123"
            # Should not create a new user
            mock_user_prisma.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_authenticate_user_missing_email(self):
        """Test authentication failure when email is missing."""
        with pytest.raises(ValueError, match="Email is required"):
            await SAMLAuthService.authenticate_user(
                provider_name="okta",
                name_id="name123",
                attributes={"name": "Test User"}  # No email
            )
    
    @pytest.mark.asyncio
    async def test_logout_user(self):
        """Test logging out a SAML user."""
        with patch('backend.integrations.auth.saml_data.SAMLProviderData.get_provider_by_name') as mock_get_provider, \
             patch('backend.integrations.auth.saml_data.SAMLUserData.get_user_by_name_id') as mock_get_user, \
             patch('backend.integrations.auth.saml_data.SAMLUserData.deactivate_user') as mock_deactivate:
            
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.id = "provider123"
            mock_get_provider.return_value = mock_provider
            
            # Mock SAML user
            mock_saml_user = MagicMock()
            mock_saml_user.id = "saml123"
            mock_saml_user.email = "test@example.com"
            mock_get_user.return_value = mock_saml_user
            
            # Mock deactivation
            mock_deactivate.return_value = mock_saml_user
            
            result = await SAMLAuthService.logout_user("okta", "name123")
            
            assert result is True
            mock_deactivate.assert_called_once_with("saml123")
    
    @pytest.mark.asyncio
    async def test_get_user_providers(self):
        """Test getting all SAML providers for a user."""
        with patch('backend.integrations.auth.saml_data.SAMLUserData.get_users_by_internal_user') as mock_get_users:
            # Mock SAML users
            mock_saml1 = MagicMock()
            mock_saml1.provider.id = "provider1"
            mock_saml1.provider.providerName = "okta"
            mock_saml1.provider.displayName = "Okta"
            mock_saml1.email = "user@example.com"
            mock_saml1.lastLoginAt = datetime.now(timezone.utc)
            mock_saml1.active = True
            
            mock_saml2 = MagicMock()
            mock_saml2.provider.id = "provider2"
            mock_saml2.provider.providerName = "azure"
            mock_saml2.provider.displayName = "Azure AD"
            mock_saml2.email = "user@example.com"
            mock_saml2.lastLoginAt = datetime.now(timezone.utc)
            mock_saml2.active = False
            
            mock_get_users.return_value = [mock_saml1, mock_saml2]
            
            providers = await SAMLAuthService.get_user_providers("user123")
            
            assert len(providers) == 2
            assert providers[0]["provider_name"] == "okta"
            assert providers[0]["display_name"] == "Okta"
            assert providers[0]["active"] is True
            assert providers[1]["provider_name"] == "azure"
            assert providers[1]["active"] is False


class TestSAMLAuthRequestData:
    """Test SAMLAuthRequestData operations."""
    
    @pytest.mark.asyncio
    async def test_create_request(self):
        """Test creating an auth request."""
        with patch('backend.integrations.auth.saml_data.SAMLAuthRequest.prisma') as mock_prisma:
            mock_request = MagicMock()
            mock_request.id = "req123"
            mock_prisma.create.return_value = mock_request
            
            result = await SAMLAuthRequestData.create_request(
                provider_id="provider123",
                request_id="saml123",
                relay_state="test-state"
            )
            
            assert result == mock_request
            mock_prisma.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complete_request(self):
        """Test completing an auth request."""
        with patch('backend.integrations.auth.saml_data.SAMLAuthRequest.prisma') as mock_prisma:
            mock_request = MagicMock()
            mock_request.status = "completed"
            mock_prisma.update.return_value = mock_request
            
            result = await SAMLAuthRequestData.complete_request("req123", "user123")
            
            assert result == mock_request
            mock_prisma.update.assert_called_once_with(
                where={"id": "req123"},
                data={
                    "status": "completed",
                    "completedAt": pytest.approx(datetime.now(timezone.utc), rel=timedelta(seconds=1))
                }
            )
    
    @pytest.mark.asyncio
    async def test_cleanup_old_requests(self):
        """Test cleaning up old auth requests."""
        with patch('backend.integrations.auth.saml_data.SAMLAuthRequest.prisma') as mock_prisma:
            mock_result = MagicMock()
            mock_result.count = 10
            mock_prisma.delete_many.return_value = mock_result
            
            count = await SAMLAuthRequestData.cleanup_old_requests(hours=24)
            
            assert count == 10
            mock_prisma.delete_many.assert_called_once()


# Integration-style tests
class TestSAMLIntegration:
    """Integration tests for SAML authentication flow."""
    
    @pytest.mark.asyncio
    async def test_full_auth_flow(self):
        """Test a complete SAML authentication flow."""
        # This would be a more complex integration test
        # that tests the entire flow from login initiation to user creation
        pass
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self):
        """Test the lifecycle of a SAML provider."""
        # Test creating, updating, enabling, disabling, and deleting a provider
        pass
