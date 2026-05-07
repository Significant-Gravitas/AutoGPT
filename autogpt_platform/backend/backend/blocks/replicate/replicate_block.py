import logging
from typing import Optional

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.replicate._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ReplicateCredentialsInput,
)
from backend.blocks.replicate._helper import extract_result
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    NodeExecutionStats,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

logger = logging.getLogger(__name__)

# Replicate $/sec varies by hardware tier (CPU $0.000100 → 8×H100 $0.005600).
# The API doesn't return which tier ran the prediction, so we pick a single
# rate that covers up to A100-80GB ($0.001875/sec) without under-billing.
# Cheaper hardware (L40S/L4/T4) is over-billed slightly; multi-GPU configs
# are still under-billed but are rare in user-facing community models.
_REPLICATE_USD_PER_SEC = 0.002000


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
        model_inputs: dict[str, str | int | float | bool] = SchemaField(
            default={},
            description=(
                "Dictionary of inputs to pass to the model. Values may be "
                "strings, integers, floats, or booleans — Replicate model "
                "schemas commonly require booleans (e.g. ``generate_audio``, "
                "``safety_checker``) and floats (e.g. ``temperature``, "
                "``guidance_scale``)."
            ),
            placeholder='{"prompt": "a beautiful landscape", "num_outputs": 1, "generate_audio": false}',
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
                "model_inputs": {
                    "prompt": "Hello, world!",
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "stream": False,
                },
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

        Uses predictions.async_create + async_wait instead of async_run so
        we can read ``prediction.metrics.predict_time`` after completion
        and emit it as ``provider_cost`` for the COST_USD resolver.

        Args:
            model_ref: The model reference (e.g., "owner/model-name:version")
            model_inputs: The inputs to pass to the model
            api_key: The Replicate API key as SecretStr

        Returns:
            Model output (same shape as previous async_run path)
        """
        api_key_str = api_key.get_secret_value()
        client = ReplicateClient(api_token=api_key_str)

        # Replicate SDK: version-pinned refs use `version=`; unpinned use
        # `model=`. Matches the `owner/name[:version]` contract above.
        if ":" in model_ref:
            model_name, version = model_ref.split(":", 1)
            prediction = await client.predictions.async_create(
                version=version, input=model_inputs
            )
        else:
            prediction = await client.predictions.async_create(
                model=model_ref, input=model_inputs
            )

        await prediction.async_wait()

        # async_wait returns normally on "failed"/"canceled" — only async_run
        # raises. Without this check we'd bill partial compute time on a
        # failed run and silently yield empty output.
        if prediction.status == "failed":
            raise RuntimeError(
                f"Replicate prediction failed: {prediction.error or 'unknown error'}"
            )
        if prediction.status == "canceled":
            raise RuntimeError("Replicate prediction was canceled")

        if prediction.metrics and prediction.metrics.get("predict_time"):
            predict_time = float(prediction.metrics["predict_time"])
            self.merge_stats(
                NodeExecutionStats(
                    provider_cost=predict_time * _REPLICATE_USD_PER_SEC,
                    provider_cost_type="cost_usd",
                )
            )

        if prediction.output is None:
            raise RuntimeError("Replicate prediction returned no output")
        return extract_result(prediction.output)
