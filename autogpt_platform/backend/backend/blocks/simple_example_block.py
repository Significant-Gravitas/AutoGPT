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
from backend.sdk import *  # noqa: F403, F405


@provider("simple_service")
@cost_config(BlockCost(cost_amount=1, cost_type=BlockCostType.RUN))
class SimpleExampleBlock(Block):
    """
    A simple example block showing the power of the SDK.

    Key benefits:
    1. Single import: from backend.sdk import *
    2. Auto-registration via decorators
    3. No manual config file updates needed
    """

    class Input(BlockSchema):
        text: String = SchemaField(description="Input text")
        count: Integer = SchemaField(description="Number of repetitions", default=1)

    class Output(BlockSchema):
        result: String = SchemaField(description="Output result")

    def __init__(self):
        super().__init__(
            id="simple-example-block-11111111-2222-3333-4444-555555555555",
            description="Simple example block using SDK",
            categories={BlockCategory.TEXT},
            input_schema=SimpleExampleBlock.Input,
            output_schema=SimpleExampleBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        result = input_data.text * input_data.count
        yield "result", result
