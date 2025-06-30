"""
Example Block demonstrating the cost decorator

This shows how to define custom costs for a block using the @cost decorator.
"""

from backend.sdk import (
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    Boolean,
    CredentialsMetaInput,
    Integer,
    SchemaField,
    String,
    cost,
)

from ._config import example_service


# Example 1: Simple block with fixed cost
@cost(BlockCost(cost_type=BlockCostType.RUN, cost_amount=5))
class FixedCostBlock(Block):
    """Block with a fixed cost per run."""

    class Input(BlockSchema):
        data: String = SchemaField(description="Input data")

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed data")
        cost: Integer = SchemaField(description="Cost in credits")

    def __init__(self):
        super().__init__(
            id="96f31d13-d741-46a1-97d4-f7e1c3beb9b5",
            description="Example block with fixed cost of 5 credits per run",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "result", f"Processed: {input_data.data}"
        yield "cost", 5


# Example 2: Block with tiered costs based on input
@cost(
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=1,
        cost_filter={"tier": "basic"},
    ),
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=5,
        cost_filter={"tier": "standard"},
    ),
    BlockCost(
        cost_type=BlockCostType.RUN,
        cost_amount=10,
        cost_filter={"tier": "premium"},
    ),
)
class TieredCostBlock(Block):
    """Block with different costs based on selected tier."""

    class Input(BlockSchema):
        data: String = SchemaField(description="Input data")
        tier: String = SchemaField(
            description="Service tier (basic, standard, or premium)",
            default="basic",
        )

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed data")
        tier_used: String = SchemaField(description="Service tier used")
        cost: Integer = SchemaField(description="Cost in credits")

    def __init__(self):
        super().__init__(
            id="fb30be87-f8f7-4701-86b0-6e8cc8046750",
            description="Example block with tiered pricing",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Simulate different processing based on tier
        tier_costs = {"basic": 1, "standard": 5, "premium": 10}
        cost = tier_costs.get(input_data.tier, 1)

        yield "result", f"Processed with {input_data.tier} tier: {input_data.data}"
        yield "tier_used", input_data.tier
        yield "cost", cost


# Example 3: Block that uses provider base cost (no @cost decorator)
class ProviderCostBlock(Block):
    """Block that inherits cost from its provider configuration."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = example_service.credentials_field(
            description="Example service credentials",
        )
        data: String = SchemaField(description="Input data")

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed data")
        provider_used: String = SchemaField(description="Provider name")

    def __init__(self):
        super().__init__(
            id="68668919-91c7-4374-aa27-b130c456319b",
            description="Example block using provider base cost (1 credit per run)",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # This block will use the base cost from example_service (1 credit)
        yield "result", f"Processed by provider: {input_data.data}"
        yield "provider_used", "example-service"


# Example 4: Block with data-based cost
@cost(
    BlockCost(
        cost_type=BlockCostType.BYTE,
        cost_amount=1,  # 1 credit per byte
    )
)
class DataBasedCostBlock(Block):
    """Block that charges based on data size."""

    class Input(BlockSchema):
        data: String = SchemaField(description="Input data")
        process_intensive: Boolean = SchemaField(
            description="Use intensive processing",
            default=False,
        )

    class Output(BlockSchema):
        result: String = SchemaField(description="Processed data")
        data_size: Integer = SchemaField(description="Data size in bytes")
        estimated_cost: String = SchemaField(description="Estimated cost")

    def __init__(self):
        super().__init__(
            id="cd928dc6-75f0-4548-9004-7fae8f4cc677",
            description="Example block that charges 1 credit per byte",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data_size = len(input_data.data.encode("utf-8"))
        estimated_cost = data_size * 1.0  # 1 credit per byte

        yield "result", f"Processed {data_size} bytes"
        yield "data_size", data_size
        yield "estimated_cost", f"{estimated_cost:.3f} credits"
