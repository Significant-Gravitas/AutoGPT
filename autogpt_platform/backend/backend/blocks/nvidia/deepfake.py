from backend.blocks.nvidia._auth import (
    NvidiaCredentials,
    NvidiaCredentialsField,
    NvidiaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class NvidiaDeepfakeDetectBlock(Block):
    class Input(BlockSchema):
        credentials: NvidiaCredentialsInput = NvidiaCredentialsField()
        image_base64: str = SchemaField(
            description="Image to analyze for deepfakes", image_upload=True
        )
        return_image: bool = SchemaField(
            description="Whether to return the processed image with markings",
            default=False,
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Detection status (SUCCESS, ERROR, CONTENT_FILTERED)",
            default="",
        )
        image: str = SchemaField(
            description="Processed image with detection markings (if return_image=True)",
            default="",
            image_output=True,
        )
        is_deepfake: float = SchemaField(
            description="Probability that the image is a deepfake (0-1)",
            default=0.0,
        )

    def __init__(self):
        super().__init__(
            id="8c7d0d67-e79c-44f6-92a1-c2600c8aac7f",
            description="Detects potential deepfakes in images using Nvidia's AI API",
            categories={BlockCategory.SAFETY},
            input_schema=NvidiaDeepfakeDetectBlock.Input,
            output_schema=NvidiaDeepfakeDetectBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: NvidiaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://ai.api.nvidia.com/v1/cv/hive/deepfake-image-detection"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }

        image_data = f"data:image/jpeg;base64,{input_data.image_base64}"

        payload = {
            "input": [image_data],
            "return_image": input_data.return_image,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            result = data.get("data", [{}])[0]

            # Get deepfake probability from first bounding box if any
            deepfake_prob = 0.0
            if result.get("bounding_boxes"):
                deepfake_prob = result["bounding_boxes"][0].get("is_deepfake", 0.0)

            yield "status", result.get("status", "ERROR")
            yield "is_deepfake", deepfake_prob

            if input_data.return_image:
                image_data = result.get("image", "")
                output_data = f"data:image/jpeg;base64,{image_data}"
                yield "image", output_data
            else:
                yield "image", ""

        except Exception as e:
            yield "error", str(e)
            yield "status", "ERROR"
            yield "is_deepfake", 0.0
            yield "image", ""
