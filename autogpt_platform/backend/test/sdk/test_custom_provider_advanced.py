"""
Advanced tests for custom provider functionality including OAuth and Webhooks.

This test suite demonstrates how custom providers can integrate with all
aspects of the SDK including OAuth authentication and webhook handling.
"""

import time
from enum import Enum
from typing import Any, Optional

from backend.integrations.providers import ProviderName
from backend.sdk import (
    APIKeyCredentials,
    BaseModel,
    BaseOAuthHandler,
    BaseWebhooksManager,
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    Boolean,
    CredentialsField,
    CredentialsMetaInput,
    Dict,
    Float,
    List,
    OAuth2Credentials,
    SchemaField,
    SecretStr,
    String,
    cost_config,
    default_credentials,
    oauth_config,
    provider,
    webhook_config,
)
from backend.util.test import execute_block_test


# Custom OAuth Handler for testing
class CustomServiceOAuthHandler(BaseOAuthHandler):
    """OAuth handler for our custom service."""

    PROVIDER_NAME = ProviderName("custom-oauth-service")
    DEFAULT_SCOPES = ["read", "write", "admin"]

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        """Generate OAuth login URL."""
        scope_str = " ".join(scopes)
        return f"https://custom-oauth-service.com/oauth/authorize?client_id=test&scope={scope_str}&state={state}"

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        """Exchange authorization code for tokens."""
        # Mock token exchange
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            access_token=SecretStr("mock-access-token"),
            refresh_token=SecretStr("mock-refresh-token"),
            scopes=scopes,
            access_token_expires_at=int(time.time() + 3600),
            title="Custom OAuth Service",
            id="custom-oauth-creds",
        )


# Custom Webhook Manager for testing
class CustomWebhookManager(BaseWebhooksManager):
    """Webhook manager for our custom service."""

    PROVIDER_NAME = ProviderName("custom-webhook-service")

    class WebhookType(str, Enum):
        DATA_RECEIVED = "data_received"
        STATUS_CHANGED = "status_changed"

    @classmethod
    async def validate_payload(cls, webhook: Any, request: Any) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        # Mock payload validation
        payload = {"data": "test data", "timestamp": time.time()}
        event_type = "data_received"
        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Any,
        webhook_type: Any,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register webhook with external service."""
        # Mock webhook registration
        webhook_id = "custom-webhook-12345"
        config = {"url": ingress_url, "events": events, "resource": resource}
        return webhook_id, config

    async def _deregister_webhook(self, webhook: Any, credentials: Any) -> None:
        """Deregister webhook from external service."""
        # Mock webhook deregistration
        pass


# Test OAuth-enabled block
@provider("custom-oauth-service")
@oauth_config("custom-oauth-service", CustomServiceOAuthHandler)
@cost_config(
    BlockCost(cost_amount=15, cost_type=BlockCostType.RUN),
)
class CustomOAuthBlock(Block):
    """Block that uses OAuth authentication with a custom provider."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="custom-oauth-service",
            supported_credential_types={"oauth2"},
            required_scopes={"read", "write"},
            description="OAuth credentials for custom service",
        )
        action: String = SchemaField(
            description="Action to perform", default="fetch_data"
        )

    class Output(BlockSchema):
        data: Dict = SchemaField(description="Retrieved data")
        token_valid: Boolean = SchemaField(description="Whether OAuth token was valid")
        scopes: List[String] = SchemaField(description="Available scopes")

    def __init__(self):
        super().__init__(
            id="f3456789-abcd-ef01-2345-6789abcdef01",
            description="Custom OAuth provider test block",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CustomOAuthBlock.Input,
            output_schema=CustomOAuthBlock.Output,
            test_input={
                "credentials": {
                    "provider": "custom-oauth-service",
                    "id": "oauth-test-creds",
                    "type": "oauth2",
                    "title": "Test OAuth Creds",
                },
                "action": "test_action",
            },
            test_output=[
                ("data", {"status": "success", "action": "test_action"}),
                ("token_valid", True),
                ("scopes", ["read", "write"]),
            ],
            test_credentials=OAuth2Credentials(
                id="oauth-test-creds",
                provider="custom-oauth-service",
                access_token=SecretStr("test-access-token"),
                refresh_token=SecretStr("test-refresh-token"),
                scopes=["read", "write"],
                access_token_expires_at=int(time.time() + 3600),
                title="Test OAuth Credentials",
            ),
        )

    def run(
        self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
    ) -> BlockOutput:
        # Simulate OAuth API call
        token = credentials.access_token.get_secret_value()

        yield "data", {"status": "success", "action": input_data.action}
        yield "token_valid", bool(token)
        yield "scopes", credentials.scopes


# Event filter model for webhook
class WebhookEventFilter(BaseModel):
    data_received: bool = True
    status_changed: bool = False


# Test Webhook-enabled block
@provider("custom-webhook-service")
@webhook_config("custom-webhook-service", CustomWebhookManager)
class CustomWebhookBlock(Block):
    """Block that receives webhooks from a custom provider."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="custom-webhook-service",
            supported_credential_types={"api_key"},
            description="Credentials for webhook service",
        )
        events: WebhookEventFilter = SchemaField(
            description="Events to listen for", default_factory=WebhookEventFilter
        )
        payload: Dict = SchemaField(
            description="Webhook payload", default={}, hidden=True
        )

    class Output(BlockSchema):
        event_type: String = SchemaField(description="Type of event received")
        event_data: Dict = SchemaField(description="Event data")
        timestamp: Float = SchemaField(description="Event timestamp")

    def __init__(self):
        super().__init__(
            id="a4567890-bcde-f012-3456-7890bcdef012",
            description="Custom webhook provider test block",
            categories={BlockCategory.INPUT},
            input_schema=CustomWebhookBlock.Input,
            output_schema=CustomWebhookBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("custom-webhook-service"),
                webhook_type="data_received",
                event_filter_input="events",
                resource_format="webhook/{webhook_id}",
            ),
            test_input={
                "credentials": {
                    "provider": "custom-webhook-service",
                    "id": "webhook-test-creds",
                    "type": "api_key",
                    "title": "Test Webhook Creds",
                },
                "events": {"data_received": True, "status_changed": False},
                "payload": {
                    "type": "data_received",
                    "data": "test",
                    "timestamp": 1234567890.0,
                },
            },
            test_output=[
                ("event_type", "data_received"),
                (
                    "event_data",
                    {
                        "type": "data_received",
                        "data": "test",
                        "timestamp": 1234567890.0,
                    },
                ),
                ("timestamp", 1234567890.0),
            ],
            test_credentials=APIKeyCredentials(
                id="webhook-test-creds",
                provider="custom-webhook-service",
                api_key=SecretStr("webhook-api-key"),
                title="Webhook API Key",
                expires_at=None,
            ),
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload

        yield "event_type", payload.get("type", "unknown")
        yield "event_data", payload
        yield "timestamp", payload.get("timestamp", 0.0)


# Combined block using multiple custom features
@provider("custom-full-service")
@oauth_config("custom-full-service", CustomServiceOAuthHandler)
@webhook_config("custom-full-service", CustomWebhookManager)
@cost_config(
    BlockCost(cost_amount=20, cost_type=BlockCostType.RUN),
    BlockCost(cost_amount=5, cost_type=BlockCostType.BYTE),
)
@default_credentials(
    APIKeyCredentials(
        id="custom-full-service-default",
        provider="custom-full-service",
        api_key=SecretStr("default-full-service-key"),
        title="Custom Full Service Default Key",
        expires_at=None,
    )
)
class CustomFullServiceBlock(Block):
    """Block demonstrating all custom provider features."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="custom-full-service",
            supported_credential_types={"api_key", "oauth2"},
            description="Credentials for full service",
        )
        mode: String = SchemaField(description="Operation mode", default="standard")

    class Output(BlockSchema):
        result: String = SchemaField(description="Operation result")
        features_used: List[String] = SchemaField(description="Features utilized")

    def __init__(self):
        super().__init__(
            id="b5678901-cdef-0123-4567-8901cdef0123",
            description="Full-featured custom provider block",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=CustomFullServiceBlock.Input,
            output_schema=CustomFullServiceBlock.Output,
            test_input={
                "credentials": {
                    "provider": "custom-full-service",
                    "id": "full-test-creds",
                    "type": "api_key",
                    "title": "Full Service Test Creds",
                },
                "mode": "test",
            },
            test_output=[
                ("result", "SUCCESS: test mode"),
                ("features_used", ["provider", "cost_config", "default_credentials"]),
            ],
            test_credentials=APIKeyCredentials(
                id="full-test-creds",
                provider="custom-full-service",
                api_key=SecretStr("full-service-test-key"),
                title="Full Service Test Key",
                expires_at=None,
            ),
        )

    def run(self, input_data: Input, *, credentials: Any, **kwargs) -> BlockOutput:
        features = ["provider", "cost_config", "default_credentials"]

        if isinstance(credentials, OAuth2Credentials):
            features.append("oauth")

        yield "result", f"SUCCESS: {input_data.mode} mode"
        yield "features_used", features


class TestCustomProviderAdvanced:
    """Advanced test suite for custom provider functionality."""

    def test_oauth_handler_registration(self):
        """Test that custom OAuth handlers are registered."""
        from backend.sdk.auto_registry import get_registry

        registry = get_registry()
        oauth_handlers = registry.get_oauth_handlers_dict()

        # Check if our custom OAuth handler is registered
        assert "custom-oauth-service" in oauth_handlers
        assert oauth_handlers["custom-oauth-service"] == CustomServiceOAuthHandler

    def test_webhook_manager_registration(self):
        """Test that custom webhook managers are registered."""
        from backend.sdk.auto_registry import get_registry

        registry = get_registry()
        webhook_managers = registry.get_webhook_managers_dict()

        # Check if our custom webhook manager is registered
        assert "custom-webhook-service" in webhook_managers
        assert webhook_managers["custom-webhook-service"] == CustomWebhookManager

    def test_oauth_block_execution(self):
        """Test OAuth-enabled block execution."""
        block = CustomOAuthBlock()
        execute_block_test(block)

    def test_webhook_block_execution(self):
        """Test webhook-enabled block execution."""
        block = CustomWebhookBlock()
        execute_block_test(block)

    def test_full_service_block_execution(self):
        """Test full-featured block execution."""
        block = CustomFullServiceBlock()
        execute_block_test(block)

    def test_multiple_decorators_on_same_provider(self):
        """Test that a single provider can have multiple features."""
        from backend.sdk.auto_registry import get_registry

        registry = get_registry()

        # Check OAuth handler
        oauth_handlers = registry.get_oauth_handlers_dict()
        assert "custom-full-service" in oauth_handlers

        # Check webhook manager
        webhook_managers = registry.get_webhook_managers_dict()
        assert "custom-full-service" in webhook_managers

        # Check default credentials
        default_creds = registry.get_default_credentials_list()
        full_service_creds = [
            cred for cred in default_creds if cred.provider == "custom-full-service"
        ]
        assert len(full_service_creds) >= 1

        # Check cost config
        block_costs = registry.get_block_costs_dict()
        assert CustomFullServiceBlock in block_costs


# Main test function
def test_custom_provider_advanced_functionality():
    """Run all advanced custom provider tests."""
    test_instance = TestCustomProviderAdvanced()

    test_instance.test_oauth_handler_registration()
    test_instance.test_webhook_manager_registration()
    test_instance.test_oauth_block_execution()
    test_instance.test_webhook_block_execution()
    test_instance.test_full_service_block_execution()
    test_instance.test_multiple_decorators_on_same_provider()
