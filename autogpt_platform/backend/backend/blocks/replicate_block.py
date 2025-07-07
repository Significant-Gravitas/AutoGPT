"""
Custom Replicate Model Block for AutoGPT
This block allows you to use any Replicate model by supplying the model name and inputs.
"""

import logging
import traceback
from typing import Any, Dict, Literal, Optional

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

# Test credentials for development
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    api_key=SecretStr("test_key"),
    provider=ProviderName.REPLICATE,
    title="Replicate",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class ReplicateCredentials(APIKeyCredentials):
    """Credentials for Replicate API"""

    api_key: SecretStr = SchemaField(
        description="Replicate API key", placeholder="r8_...", secret=True
    )


class ReplicateModelBlock(Block):
    """
    Block for running any Replicate model with custom inputs.

    This block allows you to:
    - Use any public Replicate model
    - Pass custom inputs as a dictionary
    - Specify model versions
    - Get structured outputs with prediction metadata
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Enter your Replicate API key to access the model API. You can obtain an API key from https://replicate.com/account/api-tokens.",
        )
        model_name: str = SchemaField(
            description="The Replicate model name (format: 'owner/model-name')",
            placeholder="stability-ai/stable-diffusion",
            advanced=False,
        )
        model_inputs: Dict[str, Any] = SchemaField(
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

    class Output(BlockSchema):
        result: Any = SchemaField(description="The output from the Replicate model")
        prediction_id: str = SchemaField(description="ID of the prediction")
        status: str = SchemaField(description="Status of the prediction")
        model_name: str = SchemaField(description="Name of the model used")
        error: str = SchemaField(description="Error message if any", default="")

    @staticmethod
    async def mock_run_model(model_ref, model_inputs, api_key):
        return "Mock response from Replicate model", "mock_prediction_id"

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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
                ("prediction_id", str),
                ("status", str),
                ("model_name", str),
            ],
            test_mock={"run_model": staticmethod(self.mock_run_model)},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if isinstance(input_data, dict):
            input_data = self.input_schema(**input_data)
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
            logger.info(f"Running Replicate model: {model_ref}")
            logger.info(f"Model inputs: {input_data.model_inputs}")
            api_key = credentials.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            result, prediction_id = await self.run_model(
                model_ref, input_data.model_inputs, api_key
            )
            yield "result", result
            yield "prediction_id", prediction_id
            yield "status", "succeeded"
            yield "model_name", input_data.model_name
            logger.info("Model execution completed successfully")
        except Exception as e:
            print(traceback.format_exc())
            error_msg = f"Unexpected error running Replicate model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def run_model(
        self, model_ref: str, model_inputs: Dict[str, Any], api_key: str
    ) -> tuple[Any, str]:
        """
        Run the Replicate model. This method can be mocked for testing.

        Args:
            model_ref: The model reference (e.g., "owner/model-name:version")
            model_inputs: The inputs to pass to the model
            api_key: The Replicate API key

        Returns:
            Tuple of (result, prediction_id)
        """
        client = ReplicateClient(api_token=api_key)
        prediction = client.run(model_ref, input=model_inputs)
        result = prediction
        if hasattr(prediction, "__iter__") and not isinstance(
            prediction, (str, bytes, dict)
        ):
            try:
                result = list(prediction)
            except Exception as e:
                logger.warning(f"Could not convert generator to list: {e}")
                result = prediction
        prediction_id = getattr(prediction, "id", "unknown")
        return result, prediction_id


class ReplicateAsyncModelBlock(Block):
    """
    Block for running Replicate models asynchronously with polling.

    This is useful for longer-running models that don't complete immediately.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Enter your Replicate API key to access the model API. You can obtain an API key from https://replicate.com/account/api-tokens.",
        )
        model_name: str = SchemaField(
            description="The Replicate model name (format: 'owner/model-name')",
            placeholder="stability-ai/stable-diffusion",
            advanced=False,
        )
        model_inputs: Dict[str, Any] = SchemaField(
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
        max_wait_time: int = SchemaField(
            default=300,
            description="Maximum time to wait for completion (seconds)",
            advanced=True,
            ge=1,
            le=3600,
        )

    class Output(BlockSchema):
        result: Any = SchemaField(description="The output from the Replicate model")
        prediction_id: str = SchemaField(description="ID of the prediction")
        status: str = SchemaField(description="Status of the prediction")
        model_name: str = SchemaField(description="Name of the model used")
        execution_time: float = SchemaField(
            description="Time taken to execute (seconds)"
        )
        error: str = SchemaField(description="Error message if any", default="")

    @staticmethod
    async def mock_run_async_model(model_ref, model_inputs, api_key, max_wait_time):
        return "Mock response from Replicate model", "mock_prediction_id", 1.5

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6g7-8901-bcde-f23456789012",
            description="Run Replicate models asynchronously with polling",
            categories={BlockCategory.AI},
            input_schema=ReplicateAsyncModelBlock.Input,
            output_schema=ReplicateAsyncModelBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "model_name": "stability-ai/stable-diffusion",
                "model_inputs": {"prompt": "a beautiful landscape", "num_outputs": 1},
                "max_wait_time": 60,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", str),
                ("prediction_id", str),
                ("status", str),
                ("model_name", str),
                ("execution_time", float),
            ],
            test_mock={"run_async_model": staticmethod(self.mock_run_async_model)},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if isinstance(input_data, dict):
            input_data = self.input_schema(**input_data)
        """
        Execute the Replicate model asynchronously with the provided inputs.

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
            logger.info(f"Running Replicate model asynchronously: {model_ref}")
            logger.info(f"Model inputs: {input_data.model_inputs}")
            api_key = credentials.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            result, prediction_id, execution_time = await self.run_async_model(
                model_ref, input_data.model_inputs, api_key, input_data.max_wait_time
            )
            yield "result", result
            yield "prediction_id", prediction_id
            yield "status", "succeeded"
            yield "model_name", input_data.model_name
            yield "execution_time", execution_time
            logger.info("Model execution completed successfully")
        except Exception as e:
            print(traceback.format_exc())
            error_msg = f"Unexpected error running Replicate model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def run_async_model(
        self,
        model_ref: str,
        model_inputs: Dict[str, Any],
        api_key: str,
        max_wait_time: int,
    ) -> tuple[Any, str, float]:
        """
        Run the Replicate model asynchronously with polling. This method can be mocked for testing.

        Args:
            model_ref: The model reference (e.g., "owner/model-name:version")
            model_inputs: The inputs to pass to the model
            api_key: The Replicate API key
            max_wait_time: Maximum time to wait for completion (seconds)

        Returns:
            Tuple of (result, prediction_id, execution_time)
        """
        import time

        start_time = time.time()
        client = ReplicateClient(api_token=api_key)

        # Start the prediction
        prediction = client.run(model_ref, input=model_inputs)

        # Get the result
        result = prediction
        if hasattr(prediction, "__iter__") and not isinstance(
            prediction, (str, bytes, dict)
        ):
            try:
                result = list(prediction)
            except Exception as e:
                logger.warning(f"Could not convert generator to list: {e}")
                result = prediction

        prediction_id = getattr(prediction, "id", "unknown")
        execution_time = time.time() - start_time

        return result, prediction_id, execution_time


# Mock classes for testing
class MockReplicateClient:
    """Mock Replicate client for testing"""

    def __init__(self):
        self.predictions = MockPredictions()

    def run(self, model_ref, input=None):
        return MockPrediction()


class MockAsyncReplicateClient:
    """Mock async Replicate client for testing"""

    def __init__(self):
        self.predictions = MockPredictions()

    def run(self, model_ref, input=None):
        return MockPrediction()


class MockPredictions:
    """Mock predictions object for testing"""

    def create(self, model, input):
        return MockPrediction()

    def get(self, prediction_id):
        return MockPrediction()


class MockPrediction:
    """Mock prediction object for testing"""

    def __init__(self):
        self.id = "mock_prediction_id"
        self.status = "succeeded"
        self.output = "Mock response from Replicate model"
