#!/usr/bin/env python3
"""
Standalone SDK tests that can run without Redis/PostgreSQL/RabbitMQ.
Run with: python test/sdk/test_sdk_standalone.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


def test_sdk_imports():
    """Test that SDK imports work correctly"""
    print("\n=== Testing SDK Imports ===")

    # Import SDK
    import backend.sdk as sdk

    # Verify imports
    assert hasattr(sdk, "Block")
    assert hasattr(sdk, "BlockSchema")
    assert hasattr(sdk, "SchemaField")
    assert hasattr(sdk, "provider")
    assert hasattr(sdk, "cost_config")
    print("✅ SDK imports work correctly")


def test_dynamic_provider():
    """Test dynamic provider enum"""
    print("\n=== Testing Dynamic Provider ===")

    from backend.sdk import ProviderName

    # Test existing provider
    github = ProviderName.GITHUB
    assert github.value == "github"

    # Test dynamic provider
    custom = ProviderName("my-custom-provider")
    assert custom.value == "my-custom-provider"
    assert isinstance(custom, ProviderName)
    print("✅ Dynamic provider enum works")


def test_auto_registry():
    """Test auto-registration system"""
    print("\n=== Testing Auto-Registry ===")

    from backend.sdk import BlockCost, BlockCostType, cost_config, provider
    from backend.sdk.auto_registry import get_registry

    registry = get_registry()
    initial_count = len(registry.providers)

    # Register a test provider
    @provider("test-provider-xyz")
    class TestBlock:
        pass

    assert "test-provider-xyz" in registry.providers
    assert len(registry.providers) == initial_count + 1

    # Register costs
    @cost_config(BlockCost(cost_amount=5, cost_type=BlockCostType.RUN))
    class TestBlock2:
        pass

    assert TestBlock2 in registry.block_costs
    assert registry.block_costs[TestBlock2][0].cost_amount == 5

    print("✅ Auto-registry works correctly")


def test_complete_block_creation():
    """Test creating a complete block with SDK"""
    print("\n=== Testing Complete Block Creation ===")

    from backend.sdk import (
        APIKeyCredentials,
        Block,
        BlockCategory,
        BlockCost,
        BlockCostType,
        BlockOutput,
        BlockSchema,
        CredentialsField,
        CredentialsMetaInput,
        Integer,
        SchemaField,
        SecretStr,
        String,
        cost_config,
        default_credentials,
        provider,
    )

    @provider("test-ai-service")
    @cost_config(
        BlockCost(cost_amount=10, cost_type=BlockCostType.RUN),
        BlockCost(cost_amount=2, cost_type=BlockCostType.BYTE),
    )
    @default_credentials(
        APIKeyCredentials(
            id="test-ai-default",
            provider="test-ai-service",
            api_key=SecretStr("test-default-key"),
            title="Test AI Service Default Key",
        )
    )
    class TestAIBlock(Block):
        class Input(BlockSchema):
            credentials: CredentialsMetaInput = CredentialsField(
                provider="test-ai-service",
                supported_credential_types={"api_key"},
                description="API credentials",
            )
            prompt: String = SchemaField(description="AI prompt")

        class Output(BlockSchema):
            response: String = SchemaField(description="AI response")
            tokens: Integer = SchemaField(description="Token count")

        def __init__(self):
            super().__init__(
                id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                description="Test AI Service Block",
                categories={BlockCategory.AI, BlockCategory.TEXT},
                input_schema=TestAIBlock.Input,
                output_schema=TestAIBlock.Output,
            )

        def run(self, input_data: Input, **kwargs) -> BlockOutput:
            yield "response", f"AI says: {input_data.prompt}"
            yield "tokens", len(input_data.prompt.split())

    # Verify block creation
    block = TestAIBlock()
    assert block.id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert BlockCategory.AI in block.categories

    # Verify auto-registration
    from backend.sdk.auto_registry import get_registry

    registry = get_registry()

    assert "test-ai-service" in registry.providers
    assert TestAIBlock in registry.block_costs
    assert len(registry.block_costs[TestAIBlock]) == 2

    print("✅ Complete block creation works")


def run_all_tests():
    """Run all standalone tests"""
    print("\n" + "=" * 60)
    print("🧪 Running Standalone SDK Tests")
    print("=" * 60)

    tests = [
        test_sdk_imports,
        test_dynamic_provider,
        test_auto_registry,
        test_complete_block_creation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All standalone SDK tests passed!")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
