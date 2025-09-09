"""
Tests for the SDK auto-registration system via AutoRegistry.

This test suite verifies:
1. Provider registration and retrieval
2. OAuth handler registration via patches
3. Webhook manager registration via patches
4. Credential registration and management
5. Block configuration association
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from backend.integrations.providers import ProviderName
from backend.sdk import (
    APIKeyCredentials,
    AutoRegistry,
    BaseOAuthHandler,
    BaseWebhooksManager,
    Block,
    BlockConfiguration,
    Provider,
    ProviderBuilder,
)


class TestAutoRegistry:
    """Test the AutoRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        AutoRegistry.clear()

    def test_provider_registration(self):
        """Test that providers can be registered and retrieved."""
        # Create a test provider
        provider = Provider(
            name="test_provider",
            oauth_handler=None,
            webhook_manager=None,
            default_credentials=[],
            base_costs=[],
            supported_auth_types={"api_key"},
        )

        # Register it
        AutoRegistry.register_provider(provider)

        # Verify it's registered
        assert "test_provider" in AutoRegistry._providers
        assert AutoRegistry.get_provider("test_provider") == provider

    def test_provider_with_oauth(self):
        """Test provider registration with OAuth handler."""

        # Create a mock OAuth handler
        class TestOAuthHandler(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GITHUB

        from backend.sdk.provider import OAuthConfig

        # Set environment variables so OAuth handler gets registered
        with patch.dict(
            os.environ,
            {"TEST_CLIENT_ID": "test_id", "TEST_CLIENT_SECRET": "test_secret"},
        ):
            provider = Provider(
                name="oauth_provider",
                oauth_config=OAuthConfig(
                    oauth_handler=TestOAuthHandler,
                    client_id_env_var="TEST_CLIENT_ID",
                    client_secret_env_var="TEST_CLIENT_SECRET",
                ),
                webhook_manager=None,
                default_credentials=[],
                base_costs=[],
                supported_auth_types={"oauth2"},
            )

            AutoRegistry.register_provider(provider)

            # Verify OAuth handler is registered
            assert "oauth_provider" in AutoRegistry._oauth_handlers
            assert AutoRegistry._oauth_handlers["oauth_provider"] == TestOAuthHandler

    def test_provider_with_webhook_manager(self):
        """Test provider registration with webhook manager."""

        # Create a mock webhook manager
        class TestWebhookManager(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GITHUB

        provider = Provider(
            name="webhook_provider",
            oauth_handler=None,
            webhook_manager=TestWebhookManager,
            default_credentials=[],
            base_costs=[],
            supported_auth_types={"api_key"},
        )

        AutoRegistry.register_provider(provider)

        # Verify webhook manager is registered
        assert "webhook_provider" in AutoRegistry._webhook_managers
        assert AutoRegistry._webhook_managers["webhook_provider"] == TestWebhookManager

    def test_default_credentials_registration(self):
        """Test that default credentials are registered."""
        # Create test credentials
        from backend.sdk import SecretStr

        cred1 = APIKeyCredentials(
            id="test-cred-1",
            provider="test_provider",
            api_key=SecretStr("test-key-1"),
            title="Test Credential 1",
        )
        cred2 = APIKeyCredentials(
            id="test-cred-2",
            provider="test_provider",
            api_key=SecretStr("test-key-2"),
            title="Test Credential 2",
        )

        provider = Provider(
            name="test_provider",
            oauth_handler=None,
            webhook_manager=None,
            default_credentials=[cred1, cred2],
            base_costs=[],
            supported_auth_types={"api_key"},
        )

        AutoRegistry.register_provider(provider)

        # Verify credentials are registered
        all_creds = AutoRegistry.get_all_credentials()
        assert cred1 in all_creds
        assert cred2 in all_creds

    def test_api_key_registration(self):
        """Test API key environment variable registration."""
        import os

        # Set up a test environment variable
        os.environ["TEST_API_KEY"] = "test-api-key-value"

        try:
            AutoRegistry.register_api_key("test_provider", "TEST_API_KEY")

            # Verify the mapping is stored
            assert AutoRegistry._api_key_mappings["test_provider"] == "TEST_API_KEY"

            # Verify a credential was created
            all_creds = AutoRegistry.get_all_credentials()
            test_cred = next(
                (c for c in all_creds if c.id == "test_provider-default"), None
            )
            assert test_cred is not None
            assert test_cred.provider == "test_provider"
            assert test_cred.api_key.get_secret_value() == "test-api-key-value"  # type: ignore

        finally:
            # Clean up
            del os.environ["TEST_API_KEY"]

    def test_get_oauth_handlers(self):
        """Test retrieving all OAuth handlers."""

        # Register multiple providers with OAuth
        class TestOAuth1(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GITHUB

        class TestOAuth2(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GOOGLE

        from backend.sdk.provider import OAuthConfig

        # Set environment variables so OAuth handlers get registered
        with patch.dict(
            os.environ,
            {"TEST_CLIENT_ID": "test_id", "TEST_CLIENT_SECRET": "test_secret"},
        ):
            provider1 = Provider(
                name="provider1",
                oauth_config=OAuthConfig(
                    oauth_handler=TestOAuth1,
                    client_id_env_var="TEST_CLIENT_ID",
                    client_secret_env_var="TEST_CLIENT_SECRET",
                ),
                webhook_manager=None,
                default_credentials=[],
                base_costs=[],
                supported_auth_types={"oauth2"},
            )

            provider2 = Provider(
                name="provider2",
                oauth_config=OAuthConfig(
                    oauth_handler=TestOAuth2,
                    client_id_env_var="TEST_CLIENT_ID",
                    client_secret_env_var="TEST_CLIENT_SECRET",
                ),
                webhook_manager=None,
                default_credentials=[],
                base_costs=[],
                supported_auth_types={"oauth2"},
            )

            AutoRegistry.register_provider(provider1)
            AutoRegistry.register_provider(provider2)

            handlers = AutoRegistry.get_oauth_handlers()
            assert "provider1" in handlers
            assert "provider2" in handlers
            assert handlers["provider1"] == TestOAuth1
            assert handlers["provider2"] == TestOAuth2

    def test_block_configuration_registration(self):
        """Test registering block configuration."""

        # Create a test block class
        class TestBlock(Block):
            pass

        config = BlockConfiguration(
            provider="test_provider",
            costs=[],
            default_credentials=[],
            webhook_manager=None,
            oauth_handler=None,
        )

        AutoRegistry.register_block_configuration(TestBlock, config)

        # Verify it's registered
        assert TestBlock in AutoRegistry._block_configurations
        assert AutoRegistry._block_configurations[TestBlock] == config

    def test_clear_registry(self):
        """Test clearing all registrations."""
        # Add some registrations
        provider = Provider(
            name="test_provider",
            oauth_handler=None,
            webhook_manager=None,
            default_credentials=[],
            base_costs=[],
            supported_auth_types={"api_key"},
        )
        AutoRegistry.register_provider(provider)
        AutoRegistry.register_api_key("test", "TEST_KEY")

        # Clear everything
        AutoRegistry.clear()

        # Verify everything is cleared
        assert len(AutoRegistry._providers) == 0
        assert len(AutoRegistry._default_credentials) == 0
        assert len(AutoRegistry._oauth_handlers) == 0
        assert len(AutoRegistry._webhook_managers) == 0
        assert len(AutoRegistry._block_configurations) == 0
        assert len(AutoRegistry._api_key_mappings) == 0


class TestAutoRegistryPatching:
    """Test the integration patching functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        AutoRegistry.clear()

    @patch("backend.integrations.webhooks.load_webhook_managers")
    def test_webhook_manager_patching(self, mock_load_managers):
        """Test that webhook managers are patched into the system."""
        # Set up the mock to return an empty dict
        mock_load_managers.return_value = {}

        # Create a test webhook manager
        class TestWebhookManager(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GITHUB

        # Register a provider with webhooks
        provider = Provider(
            name="webhook_provider",
            oauth_handler=None,
            webhook_manager=TestWebhookManager,
            default_credentials=[],
            base_costs=[],
            supported_auth_types={"api_key"},
        )

        AutoRegistry.register_provider(provider)

        # Mock the webhooks module
        mock_webhooks = MagicMock()
        mock_webhooks.load_webhook_managers = mock_load_managers

        with patch.dict(
            "sys.modules", {"backend.integrations.webhooks": mock_webhooks}
        ):
            # Apply patches
            AutoRegistry.patch_integrations()

            # Call the patched function
            result = mock_webhooks.load_webhook_managers()

            # Verify our webhook manager is included
            assert "webhook_provider" in result
            assert result["webhook_provider"] == TestWebhookManager


class TestProviderBuilder:
    """Test the ProviderBuilder fluent API."""

    def setup_method(self):
        """Clear registry before each test."""
        AutoRegistry.clear()

    def test_basic_provider_builder(self):
        """Test building a basic provider."""
        provider = (
            ProviderBuilder("test_provider")
            .with_api_key("TEST_API_KEY", "Test API Key")
            .build()
        )

        assert provider.name == "test_provider"
        assert "api_key" in provider.supported_auth_types
        assert AutoRegistry.get_provider("test_provider") == provider

    def test_provider_builder_with_oauth(self):
        """Test building a provider with OAuth."""

        class TestOAuth(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GITHUB

        with patch.dict(
            os.environ,
            {
                "OAUTH_TEST_CLIENT_ID": "test_id",
                "OAUTH_TEST_CLIENT_SECRET": "test_secret",
            },
        ):
            provider = (
                ProviderBuilder("oauth_test")
                .with_oauth(TestOAuth, scopes=["read", "write"])
                .build()
            )

            assert provider.oauth_config is not None
            assert provider.oauth_config.oauth_handler == TestOAuth
            assert "oauth2" in provider.supported_auth_types

    def test_provider_builder_with_webhook(self):
        """Test building a provider with webhook manager."""

        class TestWebhook(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GITHUB

        provider = (
            ProviderBuilder("webhook_test").with_webhook_manager(TestWebhook).build()
        )

        assert provider.webhook_manager == TestWebhook

    def test_provider_builder_with_base_cost(self):
        """Test building a provider with base costs."""
        from backend.data.block import BlockCostType

        provider = (
            ProviderBuilder("cost_test")
            .with_base_cost(10, BlockCostType.RUN)
            .with_base_cost(5, BlockCostType.BYTE)
            .build()
        )

        assert len(provider.base_costs) == 2
        assert provider.base_costs[0].cost_amount == 10
        assert provider.base_costs[0].cost_type == BlockCostType.RUN
        assert provider.base_costs[1].cost_amount == 5
        assert provider.base_costs[1].cost_type == BlockCostType.BYTE

    def test_provider_builder_with_api_client(self):
        """Test building a provider with API client factory."""

        def mock_client_factory():
            return Mock()

        provider = (
            ProviderBuilder("client_test").with_api_client(mock_client_factory).build()
        )

        assert provider._api_client_factory == mock_client_factory

    def test_provider_builder_with_error_handler(self):
        """Test building a provider with error handler."""

        def mock_error_handler(exc: Exception) -> str:
            return f"Error: {str(exc)}"

        provider = (
            ProviderBuilder("error_test").with_error_handler(mock_error_handler).build()
        )

        assert provider._error_handler == mock_error_handler

    def test_provider_builder_complete_example(self):
        """Test building a complete provider with all features."""
        from backend.data.block import BlockCostType

        class TestOAuth(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GITHUB

        class TestWebhook(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GITHUB

        def client_factory():
            return Mock()

        def error_handler(exc):
            return str(exc)

        # Set environment variables for OAuth to be registered
        with patch.dict(
            os.environ,
            {
                "COMPLETE_TEST_CLIENT_ID": "test_id",
                "COMPLETE_TEST_CLIENT_SECRET": "test_secret",
                "COMPLETE_API_KEY": "test_api_key",
            },
        ):
            provider = (
                ProviderBuilder("complete_test")
                .with_api_key("COMPLETE_API_KEY", "Complete API Key")
                .with_oauth(TestOAuth, scopes=["read"])
                .with_webhook_manager(TestWebhook)
                .with_base_cost(100, BlockCostType.RUN)
                .with_api_client(client_factory)
                .with_error_handler(error_handler)
                .with_config(custom_setting="value")
                .build()
            )

            # Verify all settings
            assert provider.name == "complete_test"
            assert "api_key" in provider.supported_auth_types
            assert "oauth2" in provider.supported_auth_types
            assert provider.oauth_config is not None
            assert provider.oauth_config.oauth_handler == TestOAuth
            assert provider.webhook_manager == TestWebhook
            assert len(provider.base_costs) == 1
            assert provider._api_client_factory == client_factory
            assert provider._error_handler == error_handler
            assert provider.get_config("custom_setting") == "value"  # from with_config

            # Verify it's registered
            assert AutoRegistry.get_provider("complete_test") == provider
            assert "complete_test" in AutoRegistry._oauth_handlers
            assert "complete_test" in AutoRegistry._webhook_managers


class TestSDKImports:
    """Test that all expected exports are available from the SDK."""

    def test_core_block_imports(self):
        """Test core block system imports."""
        from backend.sdk import Block, BlockCategory

        # Just verify they're importable
        assert Block is not None
        assert BlockCategory is not None

    def test_schema_imports(self):
        """Test schema and model imports."""
        from backend.sdk import APIKeyCredentials, SchemaField

        assert SchemaField is not None
        assert APIKeyCredentials is not None

    def test_type_alias_imports(self):
        """Test type alias imports are removed."""
        # Type aliases have been removed from SDK
        # Users should import from typing or use built-in types directly
        pass

    def test_cost_system_imports(self):
        """Test cost system imports."""
        from backend.sdk import BlockCost, BlockCostType

        assert BlockCost is not None
        assert BlockCostType is not None

    def test_utility_imports(self):
        """Test utility imports."""
        from backend.sdk import BaseModel, Requests, json

        assert json is not None
        assert BaseModel is not None
        assert Requests is not None

    def test_integration_imports(self):
        """Test integration imports."""
        from backend.sdk import ProviderName

        assert ProviderName is not None

    def test_sdk_component_imports(self):
        """Test SDK-specific component imports."""
        from backend.sdk import AutoRegistry, ProviderBuilder

        assert AutoRegistry is not None
        assert ProviderBuilder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
