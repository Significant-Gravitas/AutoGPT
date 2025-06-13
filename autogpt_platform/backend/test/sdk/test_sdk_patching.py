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
    async def validate_payload(cls, webhook, request):
        return {}, "test_event"

    async def _register_webhook(self, *args, **kwargs):
        return "mock_webhook_id", {}

    async def _deregister_webhook(self, *args, **kwargs):
        pass


class TestOAuthPatching:
    """Test OAuth handler patching functionality."""

    def setup_method(self):
        """Clear registry and set up mocks."""
        AutoRegistry.clear()

    def test_oauth_handler_dictionary_patching(self):
        """Test that OAuth handlers are correctly patched into HANDLERS_BY_NAME."""
        # Create original handlers
        original_handlers = {
            "existing_provider": Mock(spec=BaseOAuthHandler),
        }

        # Create a mock oauth module
        mock_oauth_module = MagicMock()
        mock_oauth_module.HANDLERS_BY_NAME = original_handlers.copy()

        # Register a new provider with OAuth
        (ProviderBuilder("new_oauth_provider").with_oauth(MockOAuthHandler).build())

        # Apply patches
        with patch("backend.integrations.oauth", mock_oauth_module):
            AutoRegistry.patch_integrations()

            patched_dict = mock_oauth_module.HANDLERS_BY_NAME

            # Test that original handler still exists
            assert "existing_provider" in patched_dict
            assert (
                patched_dict["existing_provider"]
                == original_handlers["existing_provider"]
            )

            # Test that new handler is accessible
            assert "new_oauth_provider" in patched_dict
            assert patched_dict["new_oauth_provider"] == MockOAuthHandler

            # Test dict methods
            assert "existing_provider" in patched_dict.keys()
            assert "new_oauth_provider" in patched_dict.keys()

            # Test .get() method
            assert patched_dict.get("new_oauth_provider") == MockOAuthHandler
            assert patched_dict.get("nonexistent", "default") == "default"

            # Test __contains__
            assert "new_oauth_provider" in patched_dict
            assert "nonexistent" not in patched_dict

    def test_oauth_patching_with_multiple_providers(self):
        """Test patching with multiple OAuth providers."""

        # Create another OAuth handler
        class AnotherOAuthHandler(BaseOAuthHandler):
            PROVIDER_NAME = ProviderName.GOOGLE

        # Register multiple providers
        (ProviderBuilder("oauth_provider_1").with_oauth(MockOAuthHandler).build())

        (ProviderBuilder("oauth_provider_2").with_oauth(AnotherOAuthHandler).build())

        # Mock the oauth module
        mock_oauth_module = MagicMock()
        mock_oauth_module.HANDLERS_BY_NAME = {}

        with patch("backend.integrations.oauth", mock_oauth_module):
            AutoRegistry.patch_integrations()

            patched_dict = mock_oauth_module.HANDLERS_BY_NAME

            # Both providers should be accessible
            assert patched_dict["oauth_provider_1"] == MockOAuthHandler
            assert patched_dict["oauth_provider_2"] == AnotherOAuthHandler

            # Check values() method
            values = list(patched_dict.values())
            assert MockOAuthHandler in values
            assert AnotherOAuthHandler in values

            # Check items() method
            items = dict(patched_dict.items())
            assert items["oauth_provider_1"] == MockOAuthHandler
            assert items["oauth_provider_2"] == AnotherOAuthHandler


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

        with patch("backend.integrations.webhooks", mock_webhooks_module):
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

        with patch("backend.integrations.webhooks", mock_webhooks_module):
            # Should not raise an error
            AutoRegistry.patch_integrations()

            # Function should not be added if it didn't exist
            assert not hasattr(mock_webhooks_module, "load_webhook_managers")


class TestPatchingEdgeCases:
    """Test edge cases and error handling in patching."""

    def setup_method(self):
        """Clear registry."""
        AutoRegistry.clear()

    def test_patching_with_import_errors(self):
        """Test that patching handles import errors gracefully."""
        # Register a provider
        (ProviderBuilder("test_provider").with_oauth(MockOAuthHandler).build())

        # Make the oauth module import fail
        with patch("builtins.__import__", side_effect=ImportError("Mock import error")):
            # Should not raise an error
            AutoRegistry.patch_integrations()

    def test_patching_with_attribute_errors(self):
        """Test handling of missing attributes."""
        # Mock oauth module without HANDLERS_BY_NAME
        mock_oauth_module = MagicMock(spec=[])

        (ProviderBuilder("test_provider").with_oauth(MockOAuthHandler).build())

        with patch("backend.integrations.oauth", mock_oauth_module):
            # Should not raise an error
            AutoRegistry.patch_integrations()

    def test_patching_preserves_thread_safety(self):
        """Test that patching maintains thread safety."""
        import threading
        import time

        results = []
        errors = []

        def register_provider(name, delay=0):
            try:
                time.sleep(delay)
                (ProviderBuilder(name).with_oauth(MockOAuthHandler).build())
                results.append(name)
            except Exception as e:
                errors.append((name, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=register_provider, args=(f"provider_{i}", i * 0.01)
            )
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 5
        assert len(AutoRegistry._providers) == 5

        # Verify all providers are registered
        for i in range(5):
            assert f"provider_{i}" in AutoRegistry._providers


class TestPatchingIntegration:
    """Test the complete patching integration flow."""

    def setup_method(self):
        """Clear registry."""
        AutoRegistry.clear()

    def test_complete_provider_registration_and_patching(self):
        """Test the complete flow from provider registration to patching."""
        # Mock both oauth and webhooks modules
        mock_oauth = MagicMock()
        mock_oauth.HANDLERS_BY_NAME = {"original": Mock()}

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
        with patch("backend.integrations.oauth", mock_oauth):
            with patch("backend.integrations.webhooks", mock_webhooks):
                AutoRegistry.patch_integrations()

                # Verify OAuth patching
                oauth_dict = mock_oauth.HANDLERS_BY_NAME
                assert "complete_provider" in oauth_dict
                assert oauth_dict["complete_provider"] == MockOAuthHandler
                assert "original" in oauth_dict  # Original preserved

                # Verify webhook patching
                webhook_result = mock_webhooks.load_webhook_managers()
                assert "complete_provider" in webhook_result
                assert webhook_result["complete_provider"] == MockWebhookManager
                assert "original" in webhook_result  # Original preserved

    def test_patching_is_idempotent(self):
        """Test that calling patch_integrations multiple times is safe."""
        mock_oauth = MagicMock()
        mock_oauth.HANDLERS_BY_NAME = {}

        # Register a provider
        (ProviderBuilder("test_provider").with_oauth(MockOAuthHandler).build())

        with patch("backend.integrations.oauth", mock_oauth):
            # Patch multiple times
            AutoRegistry.patch_integrations()
            AutoRegistry.patch_integrations()
            AutoRegistry.patch_integrations()

            # Should still work correctly
            oauth_dict = mock_oauth.HANDLERS_BY_NAME
            assert "test_provider" in oauth_dict
            assert oauth_dict["test_provider"] == MockOAuthHandler

            # Should not have duplicates or errors
            assert len([k for k in oauth_dict.keys() if k == "test_provider"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
