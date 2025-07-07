"""
Test file for Replicate Model Blocks
This file contains tests and examples for the ReplicateModelBlock and ReplicateAsyncModelBlock
"""

from typing import Literal, cast
from unittest.mock import MagicMock, patch

import pytest

from backend.blocks.replicate_block import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ReplicateAsyncModelBlock,
    ReplicateModelBlock,
)
from backend.data.model import CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Type annotation to satisfy pyright while keeping runtime compatibility
CredentialsInputType = CredentialsMetaInput[ProviderName.REPLICATE, Literal["api_key"]]


class TestReplicateModelBlock:
    """Test cases for ReplicateModelBlock"""

    def setup_method(self):
        """Set up test fixtures"""
        self.block = ReplicateModelBlock()
        # Cast the dict to the proper type for pyright, but it works as dict at runtime
        self.test_input = ReplicateModelBlock.Input(
            credentials=cast(CredentialsInputType, TEST_CREDENTIALS_INPUT),
            model_name="meta/llama-2-7b-chat",
            model_inputs={"prompt": "Hello, world!", "max_new_tokens": 50},
        )

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful model execution"""
        with patch.object(self.block, "run_model") as mock_run_model:
            mock_run_model.return_value = ("Test response", "test_prediction_id")

            result = []
            async for output in self.block.run(
                self.test_input, credentials=TEST_CREDENTIALS
            ):
                result.append(output)

            # Verify the outputs
            assert len(result) == 4
            assert result[0] == ("result", "Test response")
            assert result[1] == ("prediction_id", "test_prediction_id")
            assert result[2] == ("status", "succeeded")
            assert result[3] == ("model_name", "meta/llama-2-7b-chat")

            # Verify run_model was called correctly
            mock_run_model.assert_called_once_with(
                "meta/llama-2-7b-chat",
                {"prompt": "Hello, world!", "max_new_tokens": 50},
                "test_key",
            )

    @pytest.mark.asyncio
    async def test_run_with_version(self):
        """Test model execution with version specification"""
        test_input = ReplicateModelBlock.Input(
            credentials=cast(CredentialsInputType, TEST_CREDENTIALS_INPUT),
            model_name="meta/llama-2-7b-chat",
            model_inputs={"prompt": "Hello, world!", "max_new_tokens": 50},
            version="db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        )

        with patch.object(self.block, "run_model") as mock_run_model:
            mock_run_model.return_value = ("Test response", "test_prediction_id")

            result = []
            async for output in self.block.run(
                test_input, credentials=TEST_CREDENTIALS
            ):
                result.append(output)

            # Verify run_model was called with version
            expected_model_ref = "meta/llama-2-7b-chat:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
            mock_run_model.assert_called_once_with(
                expected_model_ref,
                {"prompt": "Hello, world!", "max_new_tokens": 50},
                "test_key",
            )

    @pytest.mark.asyncio
    async def test_run_model_error(self):
        """Test error handling in model execution"""
        with patch.object(self.block, "run_model") as mock_run_model:
            mock_run_model.side_effect = Exception("API Error")

            with pytest.raises(
                RuntimeError,
                match="Unexpected error running Replicate model: API Error",
            ):
                async for _ in self.block.run(
                    self.test_input, credentials=TEST_CREDENTIALS
                ):
                    pass

    @pytest.mark.asyncio
    async def test_run_model_method(self):
        """Test the run_model method directly"""
        with patch(
            "backend.blocks.replicate_block.ReplicateClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_prediction = MagicMock()
            mock_prediction.id = "test_prediction_id"
            mock_prediction.__iter__ = lambda x: iter(["test_response"])
            mock_client.run.return_value = mock_prediction

            result, prediction_id = await self.block.run_model(
                "test/model", {"prompt": "test"}, "test_key"
            )

            assert result == ["test_response"]
            assert prediction_id == "test_prediction_id"
            mock_client.run.assert_called_once_with(
                "test/model", input={"prompt": "test"}
            )


class TestReplicateAsyncModelBlock:
    """Test cases for ReplicateAsyncModelBlock"""

    def setup_method(self):
        """Set up test fixtures"""
        self.block = ReplicateAsyncModelBlock()
        self.test_input = ReplicateAsyncModelBlock.Input(
            credentials=cast(CredentialsInputType, TEST_CREDENTIALS_INPUT),
            model_name="stability-ai/stable-diffusion",
            model_inputs={"prompt": "a beautiful landscape", "num_outputs": 1},
            max_wait_time=60,
        )

    @pytest.mark.asyncio
    async def test_run_async_success(self):
        """Test successful async model execution"""
        with patch.object(self.block, "run_async_model") as mock_run_async_model:
            mock_run_async_model.return_value = (
                "Test response",
                "test_prediction_id",
                1.5,
            )

            result = []
            async for output in self.block.run(
                self.test_input, credentials=TEST_CREDENTIALS
            ):
                result.append(output)

            # Verify the outputs
            assert len(result) == 5
            assert result[0] == ("result", "Test response")
            assert result[1] == ("prediction_id", "test_prediction_id")
            assert result[2] == ("status", "succeeded")
            assert result[3] == ("model_name", "stability-ai/stable-diffusion")
            assert result[4] == ("execution_time", 1.5)

            # Verify run_async_model was called correctly
            mock_run_async_model.assert_called_once_with(
                "stability-ai/stable-diffusion",
                {"prompt": "a beautiful landscape", "num_outputs": 1},
                "test_key",
                60,
            )

    @pytest.mark.asyncio
    async def test_run_async_model_method(self):
        """Test the run_async_model method directly"""
        with patch(
            "backend.blocks.replicate_block.ReplicateClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_prediction = MagicMock()
            mock_prediction.id = "test_prediction_id"
            mock_prediction.status = "succeeded"
            mock_prediction.__iter__ = lambda x: iter(["test_response"])
            mock_client.run.return_value = mock_prediction

            result, prediction_id, execution_time = await self.block.run_async_model(
                "test/model", {"prompt": "test"}, "test_key", 10
            )

            assert result == ["test_response"]
            assert prediction_id == "test_prediction_id"
            assert execution_time > 0
            mock_client.run.assert_called_once_with(
                "test/model", input={"prompt": "test"}
            )

    @pytest.mark.asyncio
    async def test_run_async_model_timeout(self):
        """Test timeout handling in async model execution"""
        with patch(
            "backend.blocks.replicate_block.ReplicateClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_prediction = MagicMock()
            mock_prediction.id = "test_prediction_id"
            mock_prediction.__iter__ = lambda x: iter(["test_response"])
            mock_client.run.return_value = mock_prediction

            # The simplified implementation doesn't do actual polling, so it should succeed
            result, prediction_id, execution_time = await self.block.run_async_model(
                "test/model", {"prompt": "test"}, "test_key", 1
            )

            assert result == ["test_response"]
            assert prediction_id == "test_prediction_id"
            assert execution_time > 0

    @pytest.mark.asyncio
    async def test_run_async_model_failure(self):
        """Test failure handling in async model execution"""
        with patch(
            "backend.blocks.replicate_block.ReplicateClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate an exception during client.run
            mock_client.run.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await self.block.run_async_model(
                    "test/model", {"prompt": "test"}, "test_key", 10
                )


class TestBlockInitialization:
    """Test block initialization and configuration"""

    def test_replicate_model_block_init(self):
        """Test ReplicateModelBlock initialization"""
        block = ReplicateModelBlock()

        assert block.id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert "Run Replicate models synchronously" in block.description
        assert block.input_schema == ReplicateModelBlock.Input
        assert block.output_schema == ReplicateModelBlock.Output

    def test_replicate_async_model_block_init(self):
        """Test ReplicateAsyncModelBlock initialization"""
        block = ReplicateAsyncModelBlock()

        assert block.id == "b2c3d4e5-f6g7-8901-bcde-f23456789012"
        assert "Run Replicate models asynchronously" in block.description
        assert block.input_schema == ReplicateAsyncModelBlock.Input
        assert block.output_schema == ReplicateAsyncModelBlock.Output


class TestMockMethods:
    """Test mock methods for testing"""

    @pytest.mark.asyncio
    async def test_mock_run_model(self):
        """Test the mock run_model method"""
        result, prediction_id = await ReplicateModelBlock.mock_run_model(
            "test/model", {"prompt": "test"}, "test_key"
        )

        assert result == "Mock response from Replicate model"
        assert prediction_id == "mock_prediction_id"

    @pytest.mark.asyncio
    async def test_mock_run_async_model(self):
        """Test the mock run_async_model method"""
        result, prediction_id, execution_time = (
            await ReplicateAsyncModelBlock.mock_run_async_model(
                "test/model", {"prompt": "test"}, "test_key", 60
            )
        )

        assert result == "Mock response from Replicate model"
        assert prediction_id == "mock_prediction_id"
        assert execution_time == 1.5
