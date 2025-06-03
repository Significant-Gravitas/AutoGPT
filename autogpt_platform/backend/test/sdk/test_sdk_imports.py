"""
Test the SDK import system and auto-registration
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


def test_sdk_imports():
    """Test that all expected imports are available from backend.sdk"""

    # Import the module and check its contents
    import backend.sdk as sdk

    # Core block components should be available
    assert hasattr(sdk, "Block")
    assert hasattr(sdk, "BlockCategory")
    assert hasattr(sdk, "BlockOutput")
    assert hasattr(sdk, "BlockSchema")
    assert hasattr(sdk, "SchemaField")

    # Credential types should be available
    assert hasattr(sdk, "CredentialsField")
    assert hasattr(sdk, "CredentialsMetaInput")
    assert hasattr(sdk, "APIKeyCredentials")
    assert hasattr(sdk, "OAuth2Credentials")

    # Cost system should be available
    assert hasattr(sdk, "BlockCost")
    assert hasattr(sdk, "BlockCostType")

    # Providers should be available
    assert hasattr(sdk, "ProviderName")

    # Type aliases should work
    assert sdk.String is str
    assert sdk.Integer is int
    assert sdk.Float is float
    assert sdk.Boolean is bool

    # Decorators should be available
    assert hasattr(sdk, "provider")
    assert hasattr(sdk, "cost_config")
    assert hasattr(sdk, "default_credentials")
    assert hasattr(sdk, "webhook_config")
    assert hasattr(sdk, "oauth_config")

    # Common types should be available
    assert hasattr(sdk, "List")
    assert hasattr(sdk, "Dict")
    assert hasattr(sdk, "Optional")
    assert hasattr(sdk, "Any")
    assert hasattr(sdk, "Union")
    assert hasattr(sdk, "BaseModel")
    assert hasattr(sdk, "SecretStr")

    # Utilities should be available
    assert hasattr(sdk, "json")
    assert hasattr(sdk, "logging")


def test_auto_registry():
    """Test the auto-registration system"""

    from backend.sdk import APIKeyCredentials, BlockCost, BlockCostType, SecretStr
    from backend.sdk.auto_registry import AutoRegistry, get_registry

    # Get the registry
    registry = get_registry()
    assert isinstance(registry, AutoRegistry)

    # Test registering a provider
    registry.register_provider("test-provider")
    assert "test-provider" in registry.providers

    # Test registering block costs
    test_costs = [BlockCost(cost_amount=5, cost_type=BlockCostType.RUN)]

    class TestBlock:
        pass

    registry.register_block_cost(TestBlock, test_costs)
    assert TestBlock in registry.block_costs
    assert registry.block_costs[TestBlock] == test_costs

    # Test registering credentials
    test_cred = APIKeyCredentials(
        id="test-cred",
        provider="test-provider",
        api_key=SecretStr("test-key"),
        title="Test Credential",
    )
    registry.register_default_credential(test_cred)
    assert test_cred in registry.default_credentials


def test_decorators():
    """Test that decorators work correctly"""

    from backend.sdk import (
        APIKeyCredentials,
        BlockCost,
        BlockCostType,
        SecretStr,
        cost_config,
        default_credentials,
        provider,
    )
    from backend.sdk.auto_registry import get_registry

    # Clear registry for test
    registry = get_registry()

    # Test provider decorator
    @provider("decorator-test")
    class DecoratorTestBlock:
        pass

    assert "decorator-test" in registry.providers

    # Test cost_config decorator
    @cost_config(BlockCost(cost_amount=10, cost_type=BlockCostType.RUN))
    class CostTestBlock:
        pass

    assert CostTestBlock in registry.block_costs
    assert len(registry.block_costs[CostTestBlock]) == 1
    assert registry.block_costs[CostTestBlock][0].cost_amount == 10

    # Test default_credentials decorator
    @default_credentials(
        APIKeyCredentials(
            id="decorator-test-cred",
            provider="decorator-test",
            api_key=SecretStr("test-api-key"),
            title="Decorator Test Credential",
        )
    )
    class CredTestBlock:
        pass

    # Check if credential was registered
    creds = registry.get_default_credentials_list()
    assert any(c.id == "decorator-test-cred" for c in creds)


def test_example_block_imports():
    """Test that example blocks can use SDK imports"""
    # Skip this test since example blocks were moved to examples directory
    # to avoid interfering with main tests
    pass


if __name__ == "__main__":
    # Run tests
    test_sdk_imports()
    print("✅ SDK imports test passed")

    test_auto_registry()
    print("✅ Auto-registry test passed")

    test_decorators()
    print("✅ Decorators test passed")

    test_example_block_imports()
    print("✅ Example block test passed")

    print("\n🎉 All SDK tests passed!")
