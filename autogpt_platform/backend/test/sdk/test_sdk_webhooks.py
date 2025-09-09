"""
Tests for SDK webhook functionality.

This test suite verifies webhook blocks and webhook manager integration.
"""

from enum import Enum

import pytest

from backend.integrations.providers import ProviderName
from backend.sdk import (
    APIKeyCredentials,
    AutoRegistry,
    BaseModel,
    BaseWebhooksManager,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockWebhookConfig,
    Credentials,
    CredentialsField,
    CredentialsMetaInput,
    Field,
    ProviderBuilder,
    SchemaField,
    SecretStr,
)


class TestWebhookTypes(str, Enum):
    """Test webhook event types."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class TestWebhooksManager(BaseWebhooksManager):
    """Test webhook manager implementation."""

    PROVIDER_NAME = ProviderName.GITHUB  # Reuse for testing

    class WebhookType(str, Enum):
        TEST = "test"

    @classmethod
    async def validate_payload(
        cls, webhook, request, credentials: Credentials | None = None
    ):
        """Validate incoming webhook payload."""
        # Mock implementation
        payload = {"test": "data"}
        event_type = "test_event"
        return payload, event_type

    async def _register_webhook(
        self,
        credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register webhook with external service."""
        # Mock implementation
        webhook_id = f"test_webhook_{resource}"
        config = {
            "webhook_type": webhook_type,
            "resource": resource,
            "events": events,
            "url": ingress_url,
        }
        return webhook_id, config

    async def _deregister_webhook(self, webhook, credentials) -> None:
        """Deregister webhook from external service."""
        # Mock implementation
        pass


class TestWebhookBlock(Block):
    """Test webhook block implementation."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="test_webhooks",
            supported_credential_types={"api_key"},
            description="Webhook service credentials",
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks",
        )
        resource_id: str = SchemaField(
            description="Resource to monitor",
        )
        events: list[TestWebhookTypes] = SchemaField(
            description="Events to listen for",
            default=[TestWebhookTypes.CREATED],
        )
        payload: dict = SchemaField(
            description="Webhook payload",
            default={},
        )

    class Output(BlockSchema):
        webhook_id: str = SchemaField(description="Registered webhook ID")
        is_active: bool = SchemaField(description="Webhook is active")
        event_count: int = SchemaField(description="Number of events configured")

    def __init__(self):
        super().__init__(
            id="test-webhook-block",
            description="Test webhook block",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=TestWebhookBlock.Input,
            output_schema=TestWebhookBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider="test_webhooks",  # type: ignore
                webhook_type="test",
                resource_format="{resource_id}",
            ),
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Simulate webhook registration
        webhook_id = f"webhook_{input_data.resource_id}"

        yield "webhook_id", webhook_id
        yield "is_active", True
        yield "event_count", len(input_data.events)


class TestWebhookBlockCreation:
    """Test creating webhook blocks with the SDK."""

    def setup_method(self):
        """Set up test environment."""
        AutoRegistry.clear()

        # Register a provider with webhook support
        self.provider = (
            ProviderBuilder("test_webhooks")
            .with_api_key("TEST_WEBHOOK_KEY", "Test Webhook API Key")
            .with_webhook_manager(TestWebhooksManager)
            .build()
        )

    @pytest.mark.asyncio
    async def test_basic_webhook_block(self):
        """Test creating a basic webhook block."""
        block = TestWebhookBlock()

        # Verify block configuration
        assert block.webhook_config is not None
        assert block.webhook_config.provider == "test_webhooks"
        assert block.webhook_config.webhook_type == "test"
        assert "{resource_id}" in block.webhook_config.resource_format  # type: ignore

        # Test block execution
        test_creds = APIKeyCredentials(
            id="test-webhook-creds",
            provider="test_webhooks",
            api_key=SecretStr("test-key"),
            title="Test Webhook Key",
        )

        outputs = {}
        async for name, value in block.run(
            TestWebhookBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_webhooks",
                    "id": "test-webhook-creds",
                    "type": "api_key",
                },
                webhook_url="https://example.com/webhook",
                resource_id="resource_123",
                events=[TestWebhookTypes.CREATED, TestWebhookTypes.UPDATED],
            ),
            credentials=test_creds,
        ):
            outputs[name] = value

        assert outputs["webhook_id"] == "webhook_resource_123"
        assert outputs["is_active"] is True
        assert outputs["event_count"] == 2

    @pytest.mark.asyncio
    async def test_webhook_block_with_filters(self):
        """Test webhook block with event filters."""

        class EventFilterModel(BaseModel):
            include_system: bool = Field(default=False)
            severity_levels: list[str] = Field(
                default_factory=lambda: ["info", "warning"]
            )

        class FilteredWebhookBlock(Block):
            """Webhook block with filtering."""

            class Input(BlockSchema):
                credentials: CredentialsMetaInput = CredentialsField(
                    provider="test_webhooks",
                    supported_credential_types={"api_key"},
                )
                resource: str = SchemaField(description="Resource to monitor")
                filters: EventFilterModel = SchemaField(
                    description="Event filters",
                    default_factory=EventFilterModel,
                )
                payload: dict = SchemaField(
                    description="Webhook payload",
                    default={},
                )

            class Output(BlockSchema):
                webhook_active: bool = SchemaField(description="Webhook active")
                filter_summary: str = SchemaField(description="Active filters")

            def __init__(self):
                super().__init__(
                    id="filtered-webhook-block",
                    description="Webhook with filters",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=FilteredWebhookBlock.Input,
                    output_schema=FilteredWebhookBlock.Output,
                    webhook_config=BlockWebhookConfig(
                        provider="test_webhooks",  # type: ignore
                        webhook_type="filtered",
                        resource_format="{resource}",
                    ),
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                filters = input_data.filters
                filter_parts = []

                if filters.include_system:
                    filter_parts.append("system events")

                filter_parts.append(f"{len(filters.severity_levels)} severity levels")

                yield "webhook_active", True
                yield "filter_summary", ", ".join(filter_parts)

        # Test the block
        block = FilteredWebhookBlock()

        test_creds = APIKeyCredentials(
            id="test-creds",
            provider="test_webhooks",
            api_key=SecretStr("key"),
            title="Test Key",
        )

        # Test with default filters
        outputs = {}
        async for name, value in block.run(
            FilteredWebhookBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_webhooks",
                    "id": "test-creds",
                    "type": "api_key",
                },
                resource="test_resource",
            ),
            credentials=test_creds,
        ):
            outputs[name] = value

        assert outputs["webhook_active"] is True
        assert "2 severity levels" in outputs["filter_summary"]

        # Test with custom filters
        custom_filters = EventFilterModel(
            include_system=True,
            severity_levels=["error", "critical"],
        )

        outputs = {}
        async for name, value in block.run(
            FilteredWebhookBlock.Input(
                credentials={  # type: ignore
                    "provider": "test_webhooks",
                    "id": "test-creds",
                    "type": "api_key",
                },
                resource="test_resource",
                filters=custom_filters,
            ),
            credentials=test_creds,
        ):
            outputs[name] = value

        assert "system events" in outputs["filter_summary"]
        assert "2 severity levels" in outputs["filter_summary"]


class TestWebhookManagerIntegration:
    """Test webhook manager integration with AutoRegistry."""

    def setup_method(self):
        """Clear registry."""
        AutoRegistry.clear()

    def test_webhook_manager_registration(self):
        """Test that webhook managers are properly registered."""

        # Create multiple webhook managers
        class WebhookManager1(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GITHUB

        class WebhookManager2(BaseWebhooksManager):
            PROVIDER_NAME = ProviderName.GOOGLE

        # Register providers with webhook managers
        (
            ProviderBuilder("webhook_service_1")
            .with_webhook_manager(WebhookManager1)
            .build()
        )

        (
            ProviderBuilder("webhook_service_2")
            .with_webhook_manager(WebhookManager2)
            .build()
        )

        # Verify registration
        managers = AutoRegistry.get_webhook_managers()
        assert "webhook_service_1" in managers
        assert "webhook_service_2" in managers
        assert managers["webhook_service_1"] == WebhookManager1
        assert managers["webhook_service_2"] == WebhookManager2

    @pytest.mark.asyncio
    async def test_webhook_block_with_provider_manager(self):
        """Test webhook block using a provider's webhook manager."""
        # Register provider with webhook manager
        (
            ProviderBuilder("integrated_webhooks")
            .with_api_key("INTEGRATED_KEY", "Integrated Webhook Key")
            .with_webhook_manager(TestWebhooksManager)
            .build()
        )

        # Create a block that uses this provider
        class IntegratedWebhookBlock(Block):
            """Block using integrated webhook manager."""

            class Input(BlockSchema):
                credentials: CredentialsMetaInput = CredentialsField(
                    provider="integrated_webhooks",
                    supported_credential_types={"api_key"},
                )
                target: str = SchemaField(description="Webhook target")
                payload: dict = SchemaField(
                    description="Webhook payload",
                    default={},
                )

            class Output(BlockSchema):
                status: str = SchemaField(description="Webhook status")
                manager_type: str = SchemaField(description="Manager type used")

            def __init__(self):
                super().__init__(
                    id="integrated-webhook-block",
                    description="Uses integrated webhook manager",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=IntegratedWebhookBlock.Input,
                    output_schema=IntegratedWebhookBlock.Output,
                    webhook_config=BlockWebhookConfig(
                        provider="integrated_webhooks",  # type: ignore
                        webhook_type=TestWebhooksManager.WebhookType.TEST,
                        resource_format="{target}",
                    ),
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                # Get the webhook manager for this provider
                managers = AutoRegistry.get_webhook_managers()
                manager_class = managers.get("integrated_webhooks")

                yield "status", "configured"
                yield "manager_type", (
                    manager_class.__name__ if manager_class else "none"
                )

        # Test the block
        block = IntegratedWebhookBlock()

        test_creds = APIKeyCredentials(
            id="integrated-creds",
            provider="integrated_webhooks",
            api_key=SecretStr("key"),
            title="Integrated Key",
        )

        outputs = {}
        async for name, value in block.run(
            IntegratedWebhookBlock.Input(
                credentials={  # type: ignore
                    "provider": "integrated_webhooks",
                    "id": "integrated-creds",
                    "type": "api_key",
                },
                target="test_target",
            ),
            credentials=test_creds,
        ):
            outputs[name] = value

        assert outputs["status"] == "configured"
        assert outputs["manager_type"] == "TestWebhooksManager"


class TestWebhookEventHandling:
    """Test webhook event handling in blocks."""

    @pytest.mark.asyncio
    async def test_webhook_event_processing_block(self):
        """Test a block that processes webhook events."""

        class WebhookEventBlock(Block):
            """Block that processes webhook events."""

            class Input(BlockSchema):
                event_type: str = SchemaField(description="Type of webhook event")
                payload: dict = SchemaField(description="Webhook payload")
                verify_signature: bool = SchemaField(
                    description="Whether to verify webhook signature",
                    default=True,
                )

            class Output(BlockSchema):
                processed: bool = SchemaField(description="Event was processed")
                event_summary: str = SchemaField(description="Summary of event")
                action_required: bool = SchemaField(description="Action required")

            def __init__(self):
                super().__init__(
                    id="webhook-event-processor",
                    description="Processes incoming webhook events",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=WebhookEventBlock.Input,
                    output_schema=WebhookEventBlock.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                # Process based on event type
                event_type = input_data.event_type
                payload = input_data.payload

                if event_type == "created":
                    summary = f"New item created: {payload.get('id', 'unknown')}"
                    action_required = True
                elif event_type == "updated":
                    summary = f"Item updated: {payload.get('id', 'unknown')}"
                    action_required = False
                elif event_type == "deleted":
                    summary = f"Item deleted: {payload.get('id', 'unknown')}"
                    action_required = True
                else:
                    summary = f"Unknown event: {event_type}"
                    action_required = False

                yield "processed", True
                yield "event_summary", summary
                yield "action_required", action_required

        # Test the block with different events
        block = WebhookEventBlock()

        # Test created event
        outputs = {}
        async for name, value in block.run(
            WebhookEventBlock.Input(
                event_type="created",
                payload={"id": "123", "name": "Test Item"},
            )
        ):
            outputs[name] = value

        assert outputs["processed"] is True
        assert "New item created: 123" in outputs["event_summary"]
        assert outputs["action_required"] is True

        # Test updated event
        outputs = {}
        async for name, value in block.run(
            WebhookEventBlock.Input(
                event_type="updated",
                payload={"id": "456", "changes": ["name", "status"]},
            )
        ):
            outputs[name] = value

        assert outputs["processed"] is True
        assert "Item updated: 456" in outputs["event_summary"]
        assert outputs["action_required"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
