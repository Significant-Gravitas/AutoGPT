"""Test that custom providers work with validation."""

from pydantic import BaseModel

from backend.integrations.providers import ProviderName


class TestModel(BaseModel):
    provider: ProviderName


def test_custom_provider_validation():
    """Test that custom provider names are accepted."""
    # Test with existing provider
    model1 = TestModel(provider=ProviderName("openai"))
    assert model1.provider == ProviderName.OPENAI
    assert model1.provider.value == "openai"

    # Test with custom provider
    model2 = TestModel(provider=ProviderName("my-custom-provider"))
    assert model2.provider.value == "my-custom-provider"

    # Test JSON schema
    schema = TestModel.model_json_schema()
    provider_schema = schema["properties"]["provider"]

    # Should not have enum constraint
    assert "enum" not in provider_schema
    assert provider_schema["type"] == "string"

    print("✅ Custom provider validation works!")


if __name__ == "__main__":
    test_custom_provider_validation()
