"""
Tests for the SDK's integration patching mechanism.

This test suite verifies that the AutoRegistry correctly patches
existing integration points to include SDK-registered components.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from backend.integrations.providers import ProviderName
from backend.sdk import (
    AutoRegistry,
    BaseOAuthHandler,
    BaseWebhooksManager,
    Credentials,
    ProviderBuilder,
)


class MockOAuthHandler(BaseOAuthHandler):
    """Mock OAuth handler for testing."""

    PROVIDER_NAME = ProviderName.GITHUB

    @classmethod
    async def authorize(cls, *args, **kwargs):
        return "mock_auth"


class MockWebhookManager(BaseWebhooksManager):
    """Mock webhook manager for testing."""

    PROVIDER_NAME = ProviderName.GITHUB

    @classmethod
    async def validate_payload(cls, webhook, request, credentials: Credentials | None):
        return {}, "test_event"

    async def _register_webhook(self, *args, **kwargs):
        return "mock_webhook_id", {}

    async def _deregister_webhook(self, *args, **kwargs):
        pass


class TestWebhookPatching:
    """Test webhook manager patching functionality."""

    def setup_method(self):
        """Clear registry."""
        AutoRegistry.clear()

    def test_webhook_manager_patching(self):
        """Test that webhook managers are correctly patched."""

        # Mock the original load_webhook_managers function
        def mock_load_webhook_managers():
            return {
                "existing_webhook": Mock(spec=BaseWebhooksManager),
            }

        # Register a provider with webhooks
        (
            ProviderBuilder("webhook_provider")
            .with_webhook_manager(MockWebhookManager)
            .build()
        )

        # Mock the webhooks module
        mock_webhooks_module = MagicMock()
        mock_webhooks_module.load_webhook_managers = mock_load_webhook_managers

        with patch.dict(
            "sys.modules", {"backend.integrations.webhooks": mock_webhooks_module}
        ):
            AutoRegistry.patch_integrations()

            # Call the patched function
            result = mock_webhooks_module.load_webhook_managers()

            # Original webhook should still exist
            assert "existing_webhook" in result

            # New webhook should be added
            assert "webhook_provider" in result
            assert result["webhook_provider"] == MockWebhookManager

    def test_webhook_patching_no_original_function(self):
        """Test webhook patching when load_webhook_managers doesn't exist."""
        # Mock webhooks module without load_webhook_managers
        mock_webhooks_module = MagicMock(spec=[])

        # Register a provider
        (
            ProviderBuilder("test_provider")
            .with_webhook_manager(MockWebhookManager)
            .build()
        )

        with patch.dict(
            "sys.modules", {"backend.integrations.webhooks": mock_webhooks_module}
        ):
            # Should not raise an error
            AutoRegistry.patch_integrations()

            # Function should not be added if it didn't exist
            assert not hasattr(mock_webhooks_module, "load_webhook_managers")


class TestPatchingIntegration:
    """Test the complete patching integration flow."""

    def setup_method(self):
        """Clear registry."""
        AutoRegistry.clear()

    def test_complete_provider_registration_and_patching(self):
        """Test the complete flow from provider registration to patching."""
        # Mock webhooks module
        mock_webhooks = MagicMock()
        mock_webhooks.load_webhook_managers = lambda: {"original": Mock()}

        # Create a fully featured provider
        (
            ProviderBuilder("complete_provider")
            .with_api_key("COMPLETE_KEY", "Complete API Key")
            .with_oauth(MockOAuthHandler, scopes=["read", "write"])
            .with_webhook_manager(MockWebhookManager)
            .build()
        )

        # Apply patches
        with patch.dict(
            "sys.modules",
            {
                "backend.integrations.webhooks": mock_webhooks,
            },
        ):
            AutoRegistry.patch_integrations()

            # Verify webhook patching
            webhook_result = mock_webhooks.load_webhook_managers()
            assert "complete_provider" in webhook_result
            assert webhook_result["complete_provider"] == MockWebhookManager
            assert "original" in webhook_result  # Original preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
