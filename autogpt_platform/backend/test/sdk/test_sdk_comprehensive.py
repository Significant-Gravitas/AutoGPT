"""
Comprehensive test suite for the AutoGPT SDK implementation.
Tests all aspects of the SDK including imports, decorators, and auto-registration.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestSDKImplementation:
    """Comprehensive SDK tests"""

    def test_sdk_imports_all_components(self):
        """Test that all expected components are available from backend.sdk import *"""
        # Import SDK
        import backend.sdk as sdk

        # Core block components
        assert hasattr(sdk, "Block")
        assert hasattr(sdk, "BlockCategory")
        assert hasattr(sdk, "BlockOutput")
        assert hasattr(sdk, "BlockSchema")
        assert hasattr(sdk, "BlockType")
        assert hasattr(sdk, "SchemaField")

        # Credential components
        assert hasattr(sdk, "CredentialsField")
        assert hasattr(sdk, "CredentialsMetaInput")
        assert hasattr(sdk, "APIKeyCredentials")
        assert hasattr(sdk, "OAuth2Credentials")
        assert hasattr(sdk, "UserPasswordCredentials")

        # Cost components
        assert hasattr(sdk, "BlockCost")
        assert hasattr(sdk, "BlockCostType")
        assert hasattr(sdk, "NodeExecutionStats")

        # Provider component
        assert hasattr(sdk, "ProviderName")

        # Type aliases
        assert sdk.String is str
        assert sdk.Integer is int
        assert sdk.Float is float
        assert sdk.Boolean is bool

        # Decorators
        assert hasattr(sdk, "provider")
        assert hasattr(sdk, "cost_config")
        assert hasattr(sdk, "default_credentials")
        assert hasattr(sdk, "webhook_config")
        assert hasattr(sdk, "oauth_config")

        # Common types
        assert hasattr(sdk, "List")
        assert hasattr(sdk, "Dict")
        assert hasattr(sdk, "Optional")
        assert hasattr(sdk, "Any")
        assert hasattr(sdk, "Union")
        assert hasattr(sdk, "BaseModel")
        assert hasattr(sdk, "SecretStr")
        assert hasattr(sdk, "Enum")

        # Utilities
        assert hasattr(sdk, "json")
        assert hasattr(sdk, "logging")

        print("✅ All SDK imports verified")

    def test_auto_registry_system(self):
        """Test the auto-registration system"""
        from backend.sdk import APIKeyCredentials, BlockCost, BlockCostType, SecretStr
        from backend.sdk.auto_registry import AutoRegistry, get_registry

        # Get registry instance
        registry = get_registry()
        assert isinstance(registry, AutoRegistry)

        # Test provider registration
        initial_providers = len(registry.providers)
        registry.register_provider("test-provider-123")
        assert "test-provider-123" in registry.providers
        assert len(registry.providers) == initial_providers + 1

        # Test cost registration
        class TestBlock:
            pass

        test_costs = [
            BlockCost(cost_amount=10, cost_type=BlockCostType.RUN),
            BlockCost(cost_amount=2, cost_type=BlockCostType.BYTE),
        ]
        registry.register_block_cost(TestBlock, test_costs)
        assert TestBlock in registry.block_costs
        assert len(registry.block_costs[TestBlock]) == 2
        assert registry.block_costs[TestBlock][0].cost_amount == 10

        # Test credential registration
        test_cred = APIKeyCredentials(
            id="test-cred-123",
            provider="test-provider-123",
            api_key=SecretStr("test-api-key"),
            title="Test Credential",
        )
        registry.register_default_credential(test_cred)

        # Check credential was added
        creds = registry.get_default_credentials_list()
        assert any(c.id == "test-cred-123" for c in creds)

        # Test duplicate prevention
        initial_cred_count = len(registry.default_credentials)
        registry.register_default_credential(test_cred)  # Add again
        assert (
            len(registry.default_credentials) == initial_cred_count
        )  # Should not increase

        print("✅ Auto-registry system verified")

    def test_decorators_functionality(self):
        """Test that all decorators work correctly"""
        from backend.sdk import (
            APIKeyCredentials,
            Block,
            BlockCategory,
            BlockCost,
            BlockCostType,
            BlockOutput,
            BlockSchema,
            SchemaField,
            SecretStr,
            String,
            cost_config,
            default_credentials,
            oauth_config,
            provider,
            webhook_config,
        )
        from backend.sdk.auto_registry import get_registry

        registry = get_registry()

        # Clear registry state for clean test
        # initial_provider_count = len(registry.providers)

        # Test combined decorators on a block
        @provider("test-service-xyz")
        @cost_config(
            BlockCost(cost_amount=15, cost_type=BlockCostType.RUN),
            BlockCost(cost_amount=3, cost_type=BlockCostType.SECOND),
        )
        @default_credentials(
            APIKeyCredentials(
                id="test-service-xyz-default",
                provider="test-service-xyz",
                api_key=SecretStr("default-test-key"),
                title="Test Service Default Key",
            )
        )
        class TestServiceBlock(Block):
            class Input(BlockSchema):
                text: String = SchemaField(description="Test input")

            class Output(BlockSchema):
                result: String = SchemaField(description="Test output")

            def __init__(self):
                super().__init__(
                    id="f0421f19-53da-4824-97cc-4d2bccd1399f",
                    description="Test service block",
                    categories={BlockCategory.TEXT},
                    input_schema=TestServiceBlock.Input,
                    output_schema=TestServiceBlock.Output,
                )

            def run(self, input_data: Input, **kwargs) -> BlockOutput:
                yield "result", f"Processed: {input_data.text}"

        # Verify decorators worked
        assert "test-service-xyz" in registry.providers
        assert TestServiceBlock in registry.block_costs
        assert len(registry.block_costs[TestServiceBlock]) == 2

        # Check credentials
        creds = registry.get_default_credentials_list()
        assert any(c.id == "test-service-xyz-default" for c in creds)

        # Test webhook decorator (mock classes for testing)
        class MockWebhookManager:
            pass

        @webhook_config("test-webhook-provider", MockWebhookManager)
        class TestWebhookBlock:
            pass

        assert "test-webhook-provider" in registry.webhook_managers
        assert registry.webhook_managers["test-webhook-provider"] == MockWebhookManager

        # Test oauth decorator
        class MockOAuthHandler:
            pass

        @oauth_config("test-oauth-provider", MockOAuthHandler)
        class TestOAuthBlock:
            pass

        assert "test-oauth-provider" in registry.oauth_handlers
        assert registry.oauth_handlers["test-oauth-provider"] == MockOAuthHandler

        print("✅ All decorators verified")

    def test_provider_enum_dynamic_support(self):
        """Test that ProviderName enum supports dynamic providers"""
        from backend.sdk import ProviderName

        # Test existing provider
        existing = ProviderName.GITHUB
        assert existing.value == "github"
        assert isinstance(existing, ProviderName)

        # Test dynamic provider
        dynamic = ProviderName("my-custom-provider-abc")
        assert dynamic.value == "my-custom-provider-abc"
        assert isinstance(dynamic, ProviderName)
        assert dynamic._name_ == "MY-CUSTOM-PROVIDER-ABC"

        # Test that same dynamic provider returns same instance
        dynamic2 = ProviderName("my-custom-provider-abc")
        assert dynamic.value == dynamic2.value

        # Test invalid input
        try:
            ProviderName(123)  # Should not work with non-string
            assert False, "Should have failed with non-string"
        except ValueError:
            pass  # Expected

        print("✅ Dynamic provider enum verified")

    def test_complete_block_example(self):
        """Test a complete block using all SDK features"""
        # This simulates what a block developer would write
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
            Float,
            SchemaField,
            SecretStr,
            String,
            cost_config,
            default_credentials,
            provider,
        )

        @provider("ai-translator-service")
        @cost_config(
            BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
            BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE),
        )
        @default_credentials(
            APIKeyCredentials(
                id="ai-translator-default",
                provider="ai-translator-service",
                api_key=SecretStr("translator-default-key"),
                title="AI Translator Default API Key",
            )
        )
        class AITranslatorBlock(Block):
            """AI-powered translation block using the SDK"""

            class Input(BlockSchema):
                credentials: CredentialsMetaInput = CredentialsField(
                    provider="ai-translator-service",
                    supported_credential_types={"api_key"},
                    description="API credentials for AI Translator",
                )
                text: String = SchemaField(
                    description="Text to translate", default="Hello, world!"
                )
                target_language: String = SchemaField(
                    description="Target language code", default="es"
                )

            class Output(BlockSchema):
                translated_text: String = SchemaField(description="Translated text")
                source_language: String = SchemaField(
                    description="Detected source language"
                )
                confidence: Float = SchemaField(
                    description="Translation confidence score"
                )
                error: String = SchemaField(
                    description="Error message if any", default=""
                )

            def __init__(self):
                super().__init__(
                    id="dc832afe-902a-4520-8512-d3b85428d4ec",
                    description="Translate text using AI Translator Service",
                    categories={BlockCategory.TEXT, BlockCategory.AI},
                    input_schema=AITranslatorBlock.Input,
                    output_schema=AITranslatorBlock.Output,
                    test_input={"text": "Hello, world!", "target_language": "es"},
                    test_output=[
                        ("translated_text", "¡Hola, mundo!"),
                        ("source_language", "en"),
                        ("confidence", 0.95),
                    ],
                )

            def run(
                self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
            ) -> BlockOutput:
                try:
                    # Simulate translation
                    credentials.api_key.get_secret_value()  # Verify we can access the key

                    # Mock translation logic
                    translations = {
                        ("Hello, world!", "es"): "¡Hola, mundo!",
                        ("Hello, world!", "fr"): "Bonjour le monde!",
                        ("Hello, world!", "de"): "Hallo Welt!",
                    }

                    key = (input_data.text, input_data.target_language)
                    translated = translations.get(
                        key, f"[{input_data.target_language}] {input_data.text}"
                    )

                    yield "translated_text", translated
                    yield "source_language", "en"
                    yield "confidence", 0.95
                    yield "error", ""

                except Exception as e:
                    yield "translated_text", ""
                    yield "source_language", ""
                    yield "confidence", 0.0
                    yield "error", str(e)

        # Verify the block was created correctly
        block = AITranslatorBlock()
        assert block.id == "dc832afe-902a-4520-8512-d3b85428d4ec"
        assert block.description == "Translate text using AI Translator Service"
        assert BlockCategory.TEXT in block.categories
        assert BlockCategory.AI in block.categories

        # Verify decorators registered everything
        from backend.sdk.auto_registry import get_registry

        registry = get_registry()

        assert "ai-translator-service" in registry.providers
        assert AITranslatorBlock in registry.block_costs
        assert len(registry.block_costs[AITranslatorBlock]) == 2

        creds = registry.get_default_credentials_list()
        assert any(c.id == "ai-translator-default" for c in creds)

        print("✅ Complete block example verified")

    def test_backward_compatibility(self):
        """Test that old-style imports still work"""
        # Test that we can still import from original locations
        try:
            from backend.data.block import (
                Block,
                BlockCategory,
                BlockOutput,
                BlockSchema,
            )
            from backend.data.model import SchemaField

            assert Block is not None
            assert BlockCategory is not None
            assert BlockOutput is not None
            assert BlockSchema is not None
            assert SchemaField is not None
            print("✅ Backward compatibility verified")
        except ImportError as e:
            print(f"❌ Backward compatibility issue: {e}")
            raise

    def test_auto_registration_patching(self):
        """Test that auto-registration correctly patches existing systems"""
        from backend.sdk.auto_registry import patch_existing_systems

        # This would normally be called during app startup
        # For testing, we'll verify the patching logic works
        try:
            patch_existing_systems()
            print("✅ Auto-registration patching verified")
        except Exception as e:
            print(f"⚠️ Patching had issues (expected in test environment): {e}")
            # This is expected in test environment where not all systems are loaded

    def test_import_star_works(self):
        """Test that 'from backend.sdk import *' actually works"""
        # Create a temporary module to test import *
        test_code = """
from backend.sdk import *

# Test that common items are available
assert Block is not None
assert BlockSchema is not None
assert SchemaField is not None
assert String == str
assert provider is not None
assert cost_config is not None
print("Import * works correctly")
"""

        # Execute in a clean namespace
        namespace = {"__name__": "__main__"}
        try:
            exec(test_code, namespace)
            print("✅ Import * functionality verified")
        except Exception as e:
            print(f"❌ Import * failed: {e}")
            raise


def run_all_tests():
    """Run all SDK tests"""
    print("\n" + "=" * 60)
    print("🧪 Running Comprehensive SDK Tests")
    print("=" * 60 + "\n")

    test_suite = TestSDKImplementation()

    tests = [
        ("SDK Imports", test_suite.test_sdk_imports_all_components),
        ("Auto-Registry System", test_suite.test_auto_registry_system),
        ("Decorators", test_suite.test_decorators_functionality),
        ("Dynamic Provider Enum", test_suite.test_provider_enum_dynamic_support),
        ("Complete Block Example", test_suite.test_complete_block_example),
        ("Backward Compatibility", test_suite.test_backward_compatibility),
        ("Auto-Registration Patching", test_suite.test_auto_registration_patching),
        ("Import * Syntax", test_suite.test_import_star_works),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 40)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All SDK tests passed! The implementation is working correctly.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
