import os
from enum import Enum
import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


# Model name enum
class IdeogramModelName(str, Enum):
    V2 = "V_2"
    V1 = "V_1"

    @property
    def api_name(self):
        return self.value


# Image type Enum
class ImageType(str, Enum):
    WEBP = "webp"
    JPG = "jpg"
    PNG = "png"


class IdeogramModelBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="ideogram_api_key",
            description="Ideogram API Key",
        )
        prompt: str = SchemaField(
            description="Text prompt for image generation",
            placeholder="e.g., 'A futuristic cityscape at sunset'",
            title="Prompt",
        )
        ideogram_model_name: IdeogramModelName = SchemaField(
            description="The name of the Image Generation Model, e.g., V_2",
            default=IdeogramModelName.V2,
            title="Image Generation Model",
            advanced=False,
        )
        seed: int | None = SchemaField(
            description="Random seed. Set for reproducible generation",
            default=None,
            title="Seed",
        )
        guidance: float = SchemaField(
            description=(
                "Controls the balance between adherence to the text prompt and image quality. "
                "Higher values make the output more closely match the prompt."
            ),
            default=3.0,
            title="Guidance",
        )
        aspect_ratio: str = SchemaField(
            description="Aspect ratio for the generated image",
            default="ASPECT_1_1",
            title="Aspect Ratio",
            placeholder="Choose from: ASPECT_1_1, ASPECT_16_9, etc.",
        )
        output_format: ImageType = SchemaField(
            description="File format of the output image",
            default=ImageType.PNG,
            title="Output Format",
        )
        safety_tolerance: int = SchemaField(
            description="Safety tolerance, 1 is most strict and 5 is most permissive",
            default=2,
            title="Safety Tolerance",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Generated image URL")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="6ab085e2-20b3-4055-bc3e-08036e01eca6",
            description="This block runs Ideogram models with advanced settings.",
            categories={BlockCategory.AI},
            input_schema=IdeogramModelBlock.Input,
            output_schema=IdeogramModelBlock.Output,
            test_input={
                "api_key": "test_api_key",
                "ideogram_model_name": IdeogramModelName.V2,
                "prompt": "A beautiful futuristic cityscape at dusk",
                "seed": None,
                "guidance": 3.0,
                "aspect_ratio": "ASPECT_1_1",
                "output_format": ImageType.PNG,
                "safety_tolerance": 2,
            },
            test_output=[
                (
                    "result",
                    "https://ideogram.ai/api/images/generated-image-url.png",
                ),
            ],
            test_mock={
                "run_model": lambda api_key, model_name, prompt, seed, guidance, aspect_ratio, output_format, safety_tolerance: "https://ideogram.ai/api/images/generated-image-url.png",
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        seed = input_data.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")

        try:
            result = self.run_model(
                api_key=input_data.api_key.get_secret_value(),
                model_name=input_data.ideogram_model_name.api_name,
                prompt=input_data.prompt,
                seed=seed,
                guidance=input_data.guidance,
                aspect_ratio=input_data.aspect_ratio,
                output_format=input_data.output_format,
                safety_tolerance=input_data.safety_tolerance,
            )
            yield "result", result
        except Exception as e:
            yield "error", str(e)

    def run_model(
        self,
        api_key,
        model_name,
        prompt,
        seed,
        guidance,
        aspect_ratio,
        output_format,
        safety_tolerance,
    ):
        url = "https://api.ideogram.ai/generate"
        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "image_request": {
                "prompt": prompt,
                "model": model_name,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "guidance": guidance,
                "output_format": output_format,
                "safety_tolerance": safety_tolerance,
                "magic_prompt_option": "AUTO"
            }
        }

        # Log the request data for debugging
        print("Sending Request Data:", data)

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status() 
            return response.json()['data'][0]['url']
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")  
            raise Exception(f"Failed to fetch image: {str(e)}")
