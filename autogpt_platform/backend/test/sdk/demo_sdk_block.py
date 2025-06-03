"""
Demo: Creating a new block with the SDK using 'from backend.sdk import *'

This file demonstrates the simplified block creation process.
"""

from backend.sdk import *  # noqa: F403, F405


# Create a custom translation service block with full auto-registration
@provider("ultra-translate-ai")
@cost_config(
    BlockCost(cost_amount=3, cost_type=BlockCostType.RUN),
    BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE),
)
@default_credentials(
    APIKeyCredentials(
        id="ultra-translate-default",
        provider="ultra-translate-ai",
        api_key=SecretStr("ultra-translate-default-api-key"),
        title="Ultra Translate AI Default API Key",
    )
)
class UltraTranslateBlock(Block):
    """
    Ultra Translate AI - Advanced Translation Service

    This block demonstrates how easy it is to create a new service integration
    with the SDK. No external configuration files need to be modified!
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="ultra-translate-ai",
            supported_credential_types={"api_key"},
            description="API credentials for Ultra Translate AI",
        )
        text: String = SchemaField(
            description="Text to translate", placeholder="Enter text to translate..."
        )
        source_language: String = SchemaField(
            description="Source language code (auto-detect if empty)",
            default="",
            placeholder="en, es, fr, de, ja, zh",
        )
        target_language: String = SchemaField(
            description="Target language code",
            default="es",
            placeholder="en, es, fr, de, ja, zh",
        )
        formality: String = SchemaField(
            description="Translation formality level (formal, neutral, informal)",
            default="neutral",
        )

    class Output(BlockSchema):
        translated_text: String = SchemaField(description="The translated text")
        detected_language: String = SchemaField(
            description="Auto-detected source language (if applicable)"
        )
        confidence: Float = SchemaField(
            description="Translation confidence score (0-1)"
        )
        alternatives: List[String] = SchemaField(
            description="Alternative translations", default=[]
        )
        error: String = SchemaField(
            description="Error message if translation failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="ultra-translate-block-aabbccdd-1122-3344-5566-778899aabbcc",
            description="Translate text between languages using Ultra Translate AI",
            categories={BlockCategory.TEXT, BlockCategory.AI},
            input_schema=UltraTranslateBlock.Input,
            output_schema=UltraTranslateBlock.Output,
            test_input={
                "text": "Hello, how are you?",
                "target_language": "es",
                "formality": "informal",
            },
            test_output=[
                ("translated_text", "¡Hola! ¿Cómo estás?"),
                ("detected_language", "en"),
                ("confidence", 0.98),
                ("alternatives", ["¡Hola! ¿Qué tal?", "¡Hola! ¿Cómo te va?"]),
                ("error", ""),
            ],
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            # Get API key
            api_key = credentials.api_key.get_secret_value()  # noqa: F841

            # Simulate translation based on input
            translations = {
                ("Hello, how are you?", "es", "informal"): {
                    "text": "¡Hola! ¿Cómo estás?",
                    "alternatives": ["¡Hola! ¿Qué tal?", "¡Hola! ¿Cómo te va?"],
                },
                ("Hello, how are you?", "es", "formal"): {
                    "text": "Hola, ¿cómo está usted?",
                    "alternatives": ["Buenos días, ¿cómo se encuentra?"],
                },
                ("Hello, how are you?", "fr", "neutral"): {
                    "text": "Bonjour, comment allez-vous ?",
                    "alternatives": ["Salut, comment ça va ?"],
                },
                ("Hello, how are you?", "de", "neutral"): {
                    "text": "Hallo, wie geht es dir?",
                    "alternatives": ["Hallo, wie geht's?"],
                },
            }

            # Get translation
            key = (input_data.text, input_data.target_language, input_data.formality)
            result = translations.get(
                key,
                {
                    "text": f"[{input_data.target_language}] {input_data.text}",
                    "alternatives": [],
                },
            )

            # Detect source language if not provided
            detected_lang = input_data.source_language or "en"

            yield "translated_text", result["text"]
            yield "detected_language", detected_lang
            yield "confidence", 0.95
            yield "alternatives", result["alternatives"]
            yield "error", ""

        except Exception as e:
            yield "translated_text", ""
            yield "detected_language", ""
            yield "confidence", 0.0
            yield "alternatives", []
            yield "error", str(e)


# This function demonstrates testing the block
def demo_block_usage():
    """Demonstrate using the block"""
    print("=" * 60)
    print("🌐 Ultra Translate AI Block Demo")
    print("=" * 60)

    # Create block instance
    block = UltraTranslateBlock()
    print(f"\n✅ Created block: {block.name}")
    print(f"   ID: {block.id}")
    print(f"   Categories: {block.categories}")

    # Check auto-registration
    from backend.sdk.auto_registry import get_registry

    registry = get_registry()

    print("\n📋 Auto-Registration Status:")
    print(f"   ✅ Provider registered: {'ultra-translate-ai' in registry.providers}")
    print(f"   ✅ Costs registered: {UltraTranslateBlock in registry.block_costs}")
    if UltraTranslateBlock in registry.block_costs:
        costs = registry.block_costs[UltraTranslateBlock]
        print(f"      - Per run: {costs[0].cost_amount} credits")
        print(f"      - Per byte: {costs[1].cost_amount} credits")

    creds = registry.get_default_credentials_list()
    has_default_cred = any(c.id == "ultra-translate-default" for c in creds)
    print(f"   ✅ Default credentials: {has_default_cred}")

    # Test dynamic provider enum
    print("\n🔧 Dynamic Provider Test:")
    provider = ProviderName("ultra-translate-ai")
    print(f"   ✅ Custom provider accepted: {provider.value}")
    print(f"   ✅ Is ProviderName instance: {isinstance(provider, ProviderName)}")

    # Test block execution
    print("\n🚀 Test Block Execution:")
    test_creds = APIKeyCredentials(
        id="test",
        provider="ultra-translate-ai",
        api_key=SecretStr("test-api-key"),
        title="Test",
    )

    # Create test input with credentials meta
    test_input = UltraTranslateBlock.Input(
        credentials={"provider": "ultra-translate-ai", "id": "test", "type": "api_key"},
        text="Hello, how are you?",
        target_language="es",
        formality="informal",
    )

    results = list(block.run(test_input, credentials=test_creds))
    output = {k: v for k, v in results}

    print(f"   Input: '{test_input.text}'")
    print(f"   Target: {test_input.target_language} ({test_input.formality})")
    print(f"   Output: '{output['translated_text']}'")
    print(f"   Confidence: {output['confidence']}")
    print(f"   Alternatives: {output['alternatives']}")

    print("\n✨ Block works perfectly with zero external configuration!")


if __name__ == "__main__":
    demo_block_usage()
