from enum import Enum

from openai import OpenAI

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


# Model name enum
class MultimodalAIModelName(str, Enum):
    LLAMA_VISION_11B = "meta-llama/llama-3.2-11b-vision-instruct:free"
    LLAMA_VISION_90B = "meta-llama/llama-3.2-90b-vision-instruct"
    PIXTRAL_12B = "mistralai/pixtral-12b"
    CHATGPT_4O_LATEST = "openai/chatgpt-4o-latest"


class MultimodalAIBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="openrouter_api_key",
            description="OpenRouter API Key",
            advanced=False,
        )
        prompt: str = SchemaField(
            description="Text prompt for multimodal AI response",
            placeholder="e.g., 'Describe the contents of the image'",
            title="Prompt",
        )
        image_url: str = SchemaField(
            description="URL of the image to analyze",
            placeholder="e.g., 'https://example.com/image.jpg'",
            title="Image URL",
        )
        model_name: MultimodalAIModelName = SchemaField(
            description="The name of the multimodal AI model",
            default=MultimodalAIModelName.LLAMA_VISION_11B,
            title="Multimodal AI Model",
            advanced=False,
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Generated output")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="62f5be31-9896-40ed-bd71-2dda74459e20",
            description="This block runs multimodal AI models on OpenRouter with advanced settings.",
            categories={BlockCategory.AI},
            input_schema=MultimodalAIBlock.Input,
            output_schema=MultimodalAIBlock.Output,
            test_input={
                "api_key": "test_api_key",
                "model_name": MultimodalAIModelName.LLAMA_VISION_11B,
                "prompt": "Describe the contents of the image",
                "image_url": "https://example.com/image.jpg",
            },
            test_output=[
                (
                    "result",
                    "The image depicts a serene boardwalk surrounded by lush greenery.",
                ),
            ],
            test_mock={
                "run_model": lambda api_key, model_name, prompt, image_url: "The image depicts a serene boardwalk surrounded by lush greenery.",
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Call the separated model execution logic
            result = self.run_model(
                api_key=input_data.api_key.get_secret_value(),
                model_name=input_data.model_name,
                prompt=input_data.prompt,
                image_url=input_data.image_url,
            )
            yield "result", result
        except Exception as e:
            yield "error", str(e)

    def run_model(self, api_key, model_name, prompt, image_url):
        # Initialize OpenAI client with the API key
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Call the API to create a completion based on the input data
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )

        # Extract and return the content from the API response
        return completion.choices[0].message.content
