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
    def mock_run_model(model_ref, model_inputs, api_key):
        # api_key is now a SecretStr, but we don't need to extract it for mocking
        return "Mock response from Replicate model", "mock_prediction_id"

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
                ("prediction_id", str),
                ("status", str),
                ("model_name", str),
            ],
            test_mock={"run_model": staticmethod(self.mock_run_model)},
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
            logger.info(f"Running Replicate model: {model_ref}")
            logger.info(f"Model inputs: {input_data.model_inputs}")
            result, prediction_id = await self.run_model(
                model_ref, input_data.model_inputs, credentials.api_key
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
        self, model_ref: str, model_inputs: Dict[str, Any], api_key: SecretStr
    ) -> tuple[Any, str]:
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
    def mock_run_async_model(model_ref, model_inputs, api_key, max_wait_time):
        # api_key is now a SecretStr, but we don't need to extract it for mocking
        return "Mock response from Replicate model", "mock_prediction_id", 1.5

    def __init__(self):
        super().__init__(
            id="04e6de42-99ee-4fec-972a-b56bf871608c",
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
            result, prediction_id, execution_time = await self.run_async_model(
                model_ref,
                input_data.model_inputs,
                credentials.api_key,
                input_data.max_wait_time,
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

    def process_replicate_result(self, result):
        """
        Process Replicate model results, handling various output types safely.

        Args:
            result: The result from Replicate model (could be various types)

        Returns:
            Processed result with URLs extracted as strings
        """
        # Handle AsyncIterator or Generator
        if hasattr(result, "__aiter__") or hasattr(result, "__iter__"):
            # Check if it's an async iterator
            if hasattr(result, "__aiter__"):
                # For async iterators, we need to consume them in an async context
                # This should be handled by the caller in an async function
                raise RuntimeError(
                    "AsyncIterator results need to be consumed in async context"
                )

            # Handle regular iterators/lists
            try:
                result_list = (
                    list(result) if not isinstance(result, (list, tuple)) else result
                )
                processed_items = []

                for item in result_list:
                    processed_items.append(self.extract_url_or_convert(item))

                return processed_items
            except Exception:
                # If we can't iterate, treat as single item
                return self.extract_url_or_convert(result)

        # Handle single item
        return self.extract_url_or_convert(result)

    def extract_url_or_convert(self, item):
        """
        Extract URL from Replicate output objects or convert to string.

        Args:
            item: Individual result item from Replicate

        Returns:
            URL string if available, otherwise string representation
        """
        # Check for URL attribute (common in FileOutput objects)
        if hasattr(item, "url") and item.url is not None:
            return str(item.url)

        # Check for other common attributes
        if hasattr(item, "read"):
            # File-like object, might need special handling
            return str(item)

        # For primitive types or unknown objects
        return str(item)

    async def consume_async_iterator(self, async_result):
        """
        Safely consume an async iterator from Replicate.

        Args:
            async_result: AsyncIterator from Replicate model

        Returns:
            List of processed results
        """
        results = []
        try:
            async for item in async_result:
                results.append(self.extract_url_or_convert(item))
        except Exception:
            # If async iteration fails, try to convert directly
            results.append(self.extract_url_or_convert(async_result))

        return results

    async def run_async_model(
        self,
        model_ref: str,
        model_inputs: Dict[str, Any],
        api_key: SecretStr,
        max_wait_time: int,
    ) -> tuple[Any, str, float]:
        """
        Run the Replicate model asynchronously with proper result handling.
        """
        import asyncio
        import time

        start_time = time.time()
        api_key_str = api_key.get_secret_value()
        client = ReplicateClient(api_token=api_key_str)

        try:
            # Use asyncio.wait_for to implement timeout
            raw_result = await asyncio.wait_for(
                client.async_run(model_ref, input=model_inputs), timeout=max_wait_time
            )

            # Process the result based on its type
            if hasattr(raw_result, "__aiter__"):
                # Handle async iterator
                result = await self.consume_async_iterator(raw_result)
            else:
                # Handle regular result
                result = self.process_replicate_result(raw_result)

            execution_time = time.time() - start_time

            logger.info(f"Model completed in {execution_time:.2f} seconds")
            return result, "async_result", execution_time

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Model execution timed out after {max_wait_time} seconds"
            )
        except Exception as e:
            logger.error(f"Error running async model: {e}")
            raise RuntimeError(f"Error during model execution: {e}")


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
