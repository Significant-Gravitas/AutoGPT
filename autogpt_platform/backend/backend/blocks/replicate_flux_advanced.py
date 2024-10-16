import os
from enum import Enum

import replicate

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


# Model name enum
class ReplicateFluxModelName(str, Enum):
    FLUX_SCHNELL = ("Flux Schnell",)
    FLUX_PRO = ("Flux Pro",)
    FLUX_PRO1_1 = ("Flux Pro 1.1",)

    @property
    def api_name(self):
        api_names = {
            ReplicateFluxModelName.FLUX_SCHNELL: "black-forest-labs/flux-schnell",
            ReplicateFluxModelName.FLUX_PRO: "black-forest-labs/flux-pro",
            ReplicateFluxModelName.FLUX_PRO1_1: "black-forest-labs/flux-1.1-pro",
        }
        return api_names[self]


# Image type Enum
class ImageType(str, Enum):
    WEBP = "webp"
    JPG = "jpg"
    PNG = "png"


class ReplicateFluxAdvancedModelBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="replicate_api_key",
            description="Replicate API Key",
        )
        prompt: str = SchemaField(
            description="Text prompt for image generation",
            placeholder="e.g., 'A futuristic cityscape at sunset'",
            title="Prompt",
        )
        replicate_model_name: ReplicateFluxModelName = SchemaField(
            description="The name of the Image Generation Model, i.e Flux Schnell",
            default=ReplicateFluxModelName.FLUX_SCHNELL,
            title="Image Generation Model",
            advanced=False,
        )
        seed: int | None = SchemaField(
            description="Random seed. Set for reproducible generation",
            default=None,
            title="Seed",
        )
        steps: int = SchemaField(
            description="Number of diffusion steps",
            default=25,
            title="Steps",
        )
        guidance: float = SchemaField(
            description=(
                "Controls the balance between adherence to the text prompt and image quality/diversity. "
                "Higher values make the output more closely match the prompt but may reduce overall image quality."
            ),
            default=3,
            title="Guidance",
        )
        interval: float = SchemaField(
            description=(
                "Interval is a setting that increases the variance in possible outputs. "
                "Setting this value low will ensure strong prompt following with more consistent outputs."
            ),
            default=2,
            title="Interval",
        )
        aspect_ratio: str = SchemaField(
            description="Aspect ratio for the generated image",
            default="1:1",
            title="Aspect Ratio",
            placeholder="Choose from: 1:1, 16:9, 2:3, 3:2, 4:5, 5:4, 9:16",
        )
        output_format: ImageType = SchemaField(
            description="File format of the output image",
            default=ImageType.WEBP,
            title="Output Format",
        )
        output_quality: int = SchemaField(
            description=(
                "Quality when saving the output images, from 0 to 100. "
                "Not relevant for .png outputs"
            ),
            default=80,
            title="Output Quality",
        )
        safety_tolerance: int = SchemaField(
            description="Safety tolerance, 1 is most strict and 5 is most permissive",
            default=2,
            title="Safety Tolerance",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Generated output")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="90f8c45e-e983-4644-aa0b-b4ebe2f531bc",
            description="This block runs Flux models on Replicate with advanced settings.",
            categories={BlockCategory.AI},
            input_schema=ReplicateFluxAdvancedModelBlock.Input,
            output_schema=ReplicateFluxAdvancedModelBlock.Output,
            test_input={
                "api_key": "test_api_key",
                "replicate_model_name": ReplicateFluxModelName.FLUX_SCHNELL,
                "prompt": "A beautiful landscape painting of a serene lake at sunrise",
                "seed": None,
                "steps": 25,
                "guidance": 3.0,
                "interval": 2.0,
                "aspect_ratio": "1:1",
                "output_format": ImageType.PNG,
                "output_quality": 80,
                "safety_tolerance": 2,
            },
            test_output=[
                (
                    "result",
                    "https://replicate.com/output/generated-image-url.jpg",
                ),
            ],
            test_mock={
                "run_model": lambda api_key, model_name, prompt, seed, steps, guidance, interval, aspect_ratio, output_format, output_quality, safety_tolerance: "https://replicate.com/output/generated-image-url.jpg",
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # If the seed is not provided, generate a random seed
        seed = input_data.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")

        # Run the model using the provided inputs
        result = self.run_model(
            api_key=input_data.api_key.get_secret_value(),
            model_name=input_data.replicate_model_name.api_name,
            prompt=input_data.prompt,
            seed=seed,
            steps=input_data.steps,
            guidance=input_data.guidance,
            interval=input_data.interval,
            aspect_ratio=input_data.aspect_ratio,
            output_format=input_data.output_format,
            output_quality=input_data.output_quality,
            safety_tolerance=input_data.safety_tolerance,
        )
        yield "result", result

    def run_model(
        self,
        api_key,
        model_name,
        prompt,
        seed,
        steps,
        guidance,
        interval,
        aspect_ratio,
        output_format,
        output_quality,
        safety_tolerance,
    ):
        # Initialize Replicate client with the API key
        client = replicate.Client(api_token=api_key)

        # Run the model with additional parameters
        output = client.run(
            f"{model_name}",
            input={
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "interval": interval,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "output_quality": output_quality,
                "safety_tolerance": safety_tolerance,
            },
        )

        # Check if output is a list or a string and extract accordingly; otherwise, assign a default message
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]  # If output is a list, get the first element
        elif isinstance(output, str):
            result_url = output  # If output is a string, use it directly
        else:
            result_url = (
                "No output received"  # Fallback message if output is not as expected
            )

        return result_url
