import replicate
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField

class ReplicateFluxBasicModelBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="replicate_api_key",
            description="Replicate API Key",
        )
        replicate_model_name: str = SchemaField(
            description="The name of the Flux model on Replicate (e.g., 'black-forest-labs/flux-schnell')",
            placeholder="e.g., 'black-forest-labs/flux-schnell'",
        )
        prompt: str = SchemaField(
            description="Prompt for the model",
            placeholder="e.g., 'A futuristic cityscape at sunset'",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Generated output")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="c45f25b3-ca4d-4b92-abcf-8c010400df04",
            description="This block runs flux models on Replicate with basic settings.",
            categories={BlockCategory.AI},
            input_schema=ReplicateFluxBasicModelBlock.Input,
            output_schema=ReplicateFluxBasicModelBlock.Output,
            test_input={
                "api_key": "your_test_api_key",
                "model_name": "black-forest-labs/flux-schnell",
                "prompt": "A beautiful landscape painting of a serene lake at sunrise",
            },
            test_output=[
                (
                    "result",
                    {
                        "result_url": "https://replicate.com/output/generated-image-url.jpg",
                    },
                ),
            ],
            test_mock={
                "run_model": lambda *args, **kwargs: {
                    "result_url": "https://replicate.com/output/generated-image-url.jpg",
                }
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Run the model using the provided inputs
            result = self.run_model(
                input_data.api_key.get_secret_value(),
                input_data.replicate_model_name,
                input_data.prompt,
            )
            yield "result", result
        except Exception as e:
            yield "error", str(e)

    def run_model(self, api_key, model_name, prompt):
        # Initialize Replicate client with the API key
        client = replicate.Client(api_token=api_key)

        # Run the model
        output = client.run(
            f"{model_name}",
            input={"prompt": prompt,"output_format": "png"},
        )

        # Check if output is a list or a string and extract accordingly; otherwise, assign a default message
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]  # If output is a list, get the first element
        elif isinstance(output, str):
            result_url = output  # If output is a string, use it directly
        else:
            result_url = "No output received"  # Fallback message if output is not as expected
        return result_url
