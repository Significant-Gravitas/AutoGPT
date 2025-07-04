"""
Simple Example Block using the new SDK

This demonstrates the new SDK import pattern.
Before SDK: Multiple complex imports from various modules
After SDK: Single import statement
"""

# === OLD WAY (Before SDK) ===
# from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
# from backend.data.model import SchemaField, CredentialsField, CredentialsMetaInput
# from backend.integrations.providers import ProviderName
# from backend.data.cost import BlockCost, BlockCostType
# from typing import List, Optional, Dict
# from pydantic import SecretStr

# === NEW WAY (With SDK) ===
from backend.sdk import Block, BlockCategory, BlockOutput, BlockSchema, SchemaField


class SimpleExampleBlock(Block):
    """
    A simple example block showing the power of the SDK.

    Key benefits:
    1. Single import: from backend.sdk import *
    2. Clean, simple block structure
    """

    class Input(BlockSchema):
        text: str = SchemaField(description="Input text")
        count: int = SchemaField(description="Number of repetitions", default=1)

    class Output(BlockSchema):
        result: str = SchemaField(description="Output result")

    def __init__(self):
        super().__init__(
            id="4a6db1ae-e9d5-4a57-b8b6-9186174a6dd3",
            description="Simple example block using SDK",
            categories={BlockCategory.TEXT},
            input_schema=SimpleExampleBlock.Input,
            output_schema=SimpleExampleBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        result = input_data.text * input_data.count
        yield "result", result
