"""
Integration test demonstrating the complete SDK workflow.
This shows how a developer would create a new block with zero external configuration.
"""

import sys
from enum import Enum
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# ruff: noqa: E402
# Import SDK at module level for testing (after sys.path modification)
from backend.integrations.providers import ProviderName
from backend.sdk import (
    APIKeyCredentials,
    BaseWebhooksManager,
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    CredentialsField,
    CredentialsMetaInput,
    Dict,
    Float,
    Integer,
    List,
    SchemaField,
    SecretStr,
    String,
    cost_config,
    default_credentials,
    provider,
    webhook_config,
)


def test_complete_sdk_workflow():
    """
    Demonstrate the complete workflow of creating a new block with the SDK.
    This test shows:
    1. Single import statement
    2. Custom provider registration
    3. Cost configuration
    4. Default credentials
    5. Zero external configuration needed
    """

    print("\n" + "=" * 60)
    print("🚀 SDK Integration Test - Complete Workflow")
    print("=" * 60 + "\n")

    # Step 1: Import everything needed with a single statement
    print("Step 1: Import SDK")
    # SDK already imported at module level
    print("✅ Imported all components with 'from backend.sdk import *'")

    # Step 2: Create a custom AI service block
    print("\nStep 2: Create a custom AI service block")

    @provider("custom-ai-vision-service")
    @cost_config(
        BlockCost(cost_amount=10, cost_type=BlockCostType.RUN),
        BlockCost(cost_amount=5, cost_type=BlockCostType.BYTE),
    )
    @default_credentials(
        APIKeyCredentials(
            id="custom-ai-vision-default",
            provider="custom-ai-vision-service",
            api_key=SecretStr("vision-service-default-api-key"),
            title="Custom AI Vision Service Default API Key",
            expires_at=None,
        )
    )
    class CustomAIVisionBlock(Block):
        """
        Custom AI Vision Analysis Block

        This block demonstrates:
        - Custom provider name (not in the original enum)
        - Automatic cost registration
        - Default credentials setup
        - Complex input/output schemas
        """

        class Input(BlockSchema):
            credentials: CredentialsMetaInput = CredentialsField(
                provider="custom-ai-vision-service",
                supported_credential_types={"api_key"},
                description="API credentials for Custom AI Vision Service",
            )
            image_url: String = SchemaField(
                description="URL of the image to analyze",
                placeholder="https://example.com/image.jpg",
            )
            analysis_type: String = SchemaField(
                description="Type of analysis to perform",
                default="general",
            )
            confidence_threshold: Float = SchemaField(
                description="Minimum confidence threshold for detections",
                default=0.7,
                ge=0.0,
                le=1.0,
            )
            max_results: Integer = SchemaField(
                description="Maximum number of results to return",
                default=10,
                ge=1,
                le=100,
            )

        class Output(BlockSchema):
            detections: List[Dict] = SchemaField(
                description="List of detected items with confidence scores", default=[]
            )
            analysis_type: String = SchemaField(
                description="Type of analysis performed"
            )
            processing_time: Float = SchemaField(
                description="Time taken to process the image in seconds"
            )
            total_detections: Integer = SchemaField(
                description="Total number of detections found"
            )
            error: String = SchemaField(
                description="Error message if analysis failed", default=""
            )

        def __init__(self):
            super().__init__(
                id="303d9bd3-f2a5-41ca-bb9c-e347af8ef72f",
                description="Analyze images using Custom AI Vision Service with configurable detection types",
                categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
                input_schema=CustomAIVisionBlock.Input,
                output_schema=CustomAIVisionBlock.Output,
                test_input={
                    "image_url": "https://example.com/test-image.jpg",
                    "analysis_type": "objects",
                    "confidence_threshold": 0.8,
                    "max_results": 5,
                },
                test_output=[
                    (
                        "detections",
                        [
                            {"object": "car", "confidence": 0.95},
                            {"object": "person", "confidence": 0.87},
                        ],
                    ),
                    ("analysis_type", "objects"),
                    ("processing_time", 1.23),
                    ("total_detections", 2),
                    ("error", ""),
                ],
                static_output=False,
            )

        def run(
            self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
        ) -> BlockOutput:
            import time

            start_time = time.time()

            try:
                # Get API key
                api_key = credentials.api_key.get_secret_value()

                # Simulate API call to vision service
                print(f"  - Using API key: {api_key[:10]}...")
                print(f"  - Analyzing image: {input_data.image_url}")
                print(f"  - Analysis type: {input_data.analysis_type}")

                # Mock detection results based on analysis type
                mock_results = {
                    "general": [
                        {"category": "indoor", "confidence": 0.92},
                        {"category": "office", "confidence": 0.88},
                    ],
                    "faces": [
                        {"face_id": 1, "confidence": 0.95, "age": "25-35"},
                        {"face_id": 2, "confidence": 0.91, "age": "40-50"},
                    ],
                    "objects": [
                        {"object": "laptop", "confidence": 0.94},
                        {"object": "coffee_cup", "confidence": 0.89},
                        {"object": "notebook", "confidence": 0.85},
                    ],
                    "text": [
                        {"text": "Hello World", "confidence": 0.97},
                        {"text": "SDK Demo", "confidence": 0.93},
                    ],
                    "scene": [
                        {"scene": "office_workspace", "confidence": 0.91},
                        {"scene": "indoor_lighting", "confidence": 0.87},
                    ],
                }

                # Get results for the requested analysis type
                detections = mock_results.get(
                    input_data.analysis_type,
                    [{"error": "Unknown analysis type", "confidence": 0.0}],
                )

                # Filter by confidence threshold
                filtered_detections = [
                    d
                    for d in detections
                    if d.get("confidence", 0) >= input_data.confidence_threshold
                ]

                # Limit results
                final_detections = filtered_detections[: input_data.max_results]

                # Calculate processing time
                processing_time = time.time() - start_time

                # Yield results
                yield "detections", final_detections
                yield "analysis_type", input_data.analysis_type
                yield "processing_time", round(processing_time, 3)
                yield "total_detections", len(final_detections)
                yield "error", ""

            except Exception as e:
                yield "detections", []
                yield "analysis_type", input_data.analysis_type
                yield "processing_time", time.time() - start_time
                yield "total_detections", 0
                yield "error", str(e)

    print("✅ Block class created with all decorators")

    # Step 3: Verify auto-registration worked
    print("\nStep 3: Verify auto-registration")
    from backend.sdk.auto_registry import get_registry

    registry = get_registry()

    # Check provider registration
    assert "custom-ai-vision-service" in registry.providers
    print("✅ Custom provider 'custom-ai-vision-service' auto-registered")

    # Check cost registration
    assert CustomAIVisionBlock in registry.block_costs
    costs = registry.block_costs[CustomAIVisionBlock]
    assert len(costs) == 2
    assert costs[0].cost_amount == 10
    assert costs[0].cost_type == BlockCostType.RUN
    print("✅ Block costs auto-registered (10 credits per run, 5 per byte)")

    # Check credential registration
    creds = registry.get_default_credentials_list()
    vision_cred = next((c for c in creds if c.id == "custom-ai-vision-default"), None)
    assert vision_cred is not None
    assert vision_cred.provider == "custom-ai-vision-service"
    print("✅ Default credentials auto-registered")

    # Step 4: Test dynamic provider enum
    print("\nStep 4: Test dynamic provider support")
    provider_instance = ProviderName("custom-ai-vision-service")
    assert provider_instance.value == "custom-ai-vision-service"
    assert isinstance(provider_instance, ProviderName)
    print("✅ ProviderName enum accepts custom provider dynamically")

    # Step 5: Instantiate and test the block
    print("\nStep 5: Test block instantiation and execution")
    block = CustomAIVisionBlock()

    # Verify block properties
    assert block.id == "303d9bd3-f2a5-41ca-bb9c-e347af8ef72f"
    assert BlockCategory.AI in block.categories
    assert BlockCategory.MULTIMEDIA in block.categories
    print("✅ Block instantiated successfully")

    # Test block execution
    test_credentials = APIKeyCredentials(
        id="test-cred",
        provider="custom-ai-vision-service",
        api_key=SecretStr("test-api-key-12345"),
        title="Test API Key",
    )

    test_input = CustomAIVisionBlock.Input(
        credentials={
            "provider": "custom-ai-vision-service",
            "id": "test",
            "type": "api_key",
        },
        image_url="https://example.com/test.jpg",
        analysis_type="objects",
        confidence_threshold=0.8,
        max_results=3,
    )

    print("\n  Running block with test data...")
    results = list(block.run(test_input, credentials=test_credentials))

    # Verify outputs
    output_dict = {key: value for key, value in results}
    assert "detections" in output_dict
    assert "analysis_type" in output_dict
    assert output_dict["analysis_type"] == "objects"
    assert "total_detections" in output_dict
    assert output_dict["error"] == ""
    print("✅ Block execution successful")

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("🎉 SDK Integration Test Complete!")
    print("=" * 60)
    print("\nKey achievements demonstrated:")
    print("✅ Single import: from backend.sdk import *")
    print("✅ Custom provider registered automatically")
    print("✅ Costs configured via decorator")
    print("✅ Default credentials set via decorator")
    print("✅ Block works without ANY external configuration")
    print("✅ Dynamic provider name accepted by enum")
    print("\nThe SDK successfully enables zero-configuration block development!")

    return True


def test_webhook_block_workflow():
    """Test creating a webhook block with the SDK"""

    print("\n\n" + "=" * 60)
    print("🔔 Webhook Block Integration Test")
    print("=" * 60 + "\n")

    # SDK already imported at module level

    # Create a simple webhook manager
    class CustomWebhookManager(BaseWebhooksManager):
        PROVIDER_NAME = ProviderName("custom-webhook-service")

        class WebhookType(str, Enum):
            DATA_UPDATE = "data_update"
            STATUS_CHANGE = "status_change"

        @classmethod
        async def validate_payload(cls, webhook, request) -> tuple[dict, str]:
            payload = await request.json()
            event_type = request.headers.get("X-Custom-Event", "unknown")
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
            # Mock registration
            return "webhook-12345", {"status": "registered"}

        async def _deregister_webhook(self, webhook, credentials) -> None:
            pass

    # Create webhook block
    @provider("custom-webhook-service")
    @webhook_config("custom-webhook-service", CustomWebhookManager)
    class CustomWebhookBlock(Block):
        class Input(BlockSchema):
            webhook_events: Dict = SchemaField(
                description="Events to listen for",
                default={"data_update": True, "status_change": False},
            )
            payload: Dict = SchemaField(
                description="Webhook payload", default={}, hidden=True
            )

        class Output(BlockSchema):
            event_type: String = SchemaField(description="Type of event")
            event_data: Dict = SchemaField(description="Event data")
            timestamp: String = SchemaField(description="Event timestamp")

        def __init__(self):
            super().__init__(
                id="3e730ed4-6eb2-4b89-b5ae-001860c88aef",
                description="Listen for custom webhook events",
                categories={BlockCategory.INPUT},
                input_schema=CustomWebhookBlock.Input,
                output_schema=CustomWebhookBlock.Output,
                block_type=BlockType.WEBHOOK,
                webhook_config=BlockWebhookConfig(
                    provider=ProviderName("custom-webhook-service"),
                    webhook_type="data_update",
                    event_filter_input="webhook_events",
                    resource_format="{resource}",
                ),
            )

        def run(self, input_data: Input, **kwargs) -> BlockOutput:
            payload = input_data.payload
            yield "event_type", payload.get("type", "unknown")
            yield "event_data", payload
            yield "timestamp", payload.get("timestamp", "")

    # Verify registration
    from backend.sdk.auto_registry import get_registry

    registry = get_registry()

    assert "custom-webhook-service" in registry.webhook_managers
    assert registry.webhook_managers["custom-webhook-service"] == CustomWebhookManager
    print("✅ Webhook manager auto-registered")
    print("✅ Webhook block created successfully")

    return True


if __name__ == "__main__":
    try:
        # Run main integration test
        success1 = test_complete_sdk_workflow()

        # Run webhook integration test
        success2 = test_webhook_block_workflow()

        if success1 and success2:
            print("\n\n🌟 All integration tests passed successfully!")
            sys.exit(0)
        else:
            print("\n\n❌ Some integration tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Integration test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
