import logging
from typing import Optional

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient

from backend.blocks.replicate._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ReplicateCredentialsInput,
)
from backend.blocks.replicate._helper import ReplicateOutputs, extract_result
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import APIKeyCredentials, CredentialsField, SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError

logger = logging.getLogger(__name__)


class ReplicateModelBlock(Block):
    """
    Block for running any Replicate model with custom inputs.

    This block allows you to:
    - Use any public Replicate model
    - Pass custom inputs as a dictionary
    - Specify model versions
    - Get structured outputs with prediction metadata
    """

    class Input(BlockSchemaInput):
        credentials: ReplicateCredentialsInput = CredentialsField(
            description="Enter your Replicate API key to access the model API. You can obtain an API key from https://replicate.com/account/api-tokens.",
        )
        model_name: str = SchemaField(
            description="The Replicate model name (format: 'owner/model-name')",
            placeholder="stability-ai/stable-diffusion",
            advanced=False,
        )
        model_inputs: dict[str, str | int] = SchemaField(
            default={},
            description="Dictionary of inputs to pass to the model",
            placeholder='{"prompt": "a beautiful landscape", "num_outputs": 1}',
            advanced=False,
        )
        version: Optional[str] = SchemaField(
            default=None,
            description="Specific version hash of the model (optional)",
            placeholder="db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: str = SchemaField(description="The output from the Replicate model")
        status: str = SchemaField(description="Status of the prediction")
        model_name: str = SchemaField(description="Name of the model used")

    def __init__(self):
        super().__init__(
            id="c40d75a2-d0ea-44c9-a4f6-634bb3bdab1a",
            description="Run Replicate models synchronously",
            categories={BlockCategory.AI},
            input_schema=ReplicateModelBlock.Input,
            output_schema=ReplicateModelBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "model_name": "meta/llama-2-7b-chat",
                "model_inputs": {"prompt": "Hello, world!", "max_new_tokens": 50},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", str),
                ("status", str),
                ("model_name", str),
            ],
            test_mock={
                "run_model": lambda model_ref, model_inputs, api_key: (
                    "Mock response from Replicate model"
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Execute the Replicate model with the provided inputs.

        Args:
            input_data: The input data containing model name and inputs
            credentials: The API credentials

        Yields:
            BlockOutput containing the model results and metadata
        """
        try:
            if input_data.version:
                model_ref = f"{input_data.model_name}:{input_data.version}"
            else:
                model_ref = input_data.model_name
            logger.debug(f"Running Replicate model: {model_ref}")
            result = await self.run_model(
                model_ref, input_data.model_inputs, credentials.api_key
            )
            yield "result", result
            yield "status", "succeeded"
            yield "model_name", input_data.model_name
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error running Replicate model: {error_msg}")

            # Input validation errors (422, 400) → BlockInputError
            if (
                "422" in error_msg
                or "Input validation failed" in error_msg
                or "400" in error_msg
            ):
                raise BlockInputError(
                    message=f"Invalid model inputs: {error_msg}",
                    block_name=self.name,
                    block_id=self.id,
                ) from e
            # Everything else → BlockExecutionError
            else:
                raise BlockExecutionError(
                    message=f"Replicate model error: {error_msg}",
                    block_name=self.name,
                    block_id=self.id,
                ) from e

    async def run_model(self, model_ref: str, model_inputs: dict, api_key: SecretStr):
        """
        Run the Replicate model. This method can be mocked for testing.

        Args:
            model_ref: The model reference (e.g., "owner/model-name:version")
            model_inputs: The inputs to pass to the model
            api_key: The Replicate API key as SecretStr

        Returns:
            Tuple of (result, prediction_id)
        """
        api_key_str = api_key.get_secret_value()
        client = ReplicateClient(api_token=api_key_str)
        output: ReplicateOutputs = await client.async_run(
            model_ref, input=model_inputs, wait=False
        )  # type: ignore they suck at typing

        result = extract_result(output)

        return result
