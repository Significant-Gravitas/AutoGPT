"""
Test the example blocks to ensure they work correctly with the provider pattern.
"""

import pytest

from backend.blocks.examples.example_sdk_block import ExampleSDKBlock
from backend.blocks.examples.simple_example_block import SimpleExampleBlock
from backend.sdk import APIKeyCredentials, SecretStr


class TestExampleBlocks:
    """Test the example blocks."""

    @pytest.mark.asyncio
    async def test_simple_example_block(self):
        """Test the simple example block."""
        block = SimpleExampleBlock()

        # Test execution
        outputs = {}
        async for name, value in block.run(
            SimpleExampleBlock.Input(text="Hello ", count=3),
        ):
            outputs[name] = value

        assert outputs["result"] == "Hello Hello Hello "

    @pytest.mark.asyncio
    async def test_example_sdk_block(self):
        """Test the example SDK block with credentials."""
        # Create test credentials
        test_creds = APIKeyCredentials(
            id="test-creds",
            provider="example-service",
            api_key=SecretStr("test-api-key"),
            title="Test API Key",
        )

        block = ExampleSDKBlock()

        # Test execution
        outputs = {}
        async for name, value in block.run(
            ExampleSDKBlock.Input(
                credentials={  # type: ignore
                    "provider": "example-service",
                    "id": "test-creds",
                    "type": "api_key",
                },
                text="Test input",
                max_length=50,
            ),
            credentials=test_creds,
        ):
            outputs[name] = value

        assert outputs["result"] == "PROCESSED: Test input"
        assert outputs["length"] == 21
        assert outputs["api_key_used"] is True
        assert "error" not in outputs or not outputs.get("error")
