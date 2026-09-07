"""Input schema shared by the blocks that price or buy a configured design."""

from backend.sdk import BlockSchemaInput, CredentialsMetaInput, SchemaField

from ._config import rmfg
from ._types import ManufacturingConfiguration, QuoteItemRequest


def credentials_field() -> CredentialsMetaInput:
    return rmfg.credentials_field(
        description="RMFG API key, created at rmfg.com/account under API keys."
    )


class RMFGBasketInput(BlockSchemaInput):
    """One configured design plus optional extra items, as quotes and carts take them."""

    credentials: CredentialsMetaInput = credentials_field()
    design_id: str = SchemaField(
        description="Design ID from Analyze Design.", placeholder="dsn_..."
    )
    quantity: int = SchemaField(
        description=(
            "Completed units of the design. Repeated parts in an assembly are "
            "multiplied by their instance count automatically."
        ),
        default=1,
        ge=1,
    )
    material_id: str = SchemaField(
        description=(
            "Sheet-metal stock for every sheet part, from List Materials. "
            "Leave empty for tube-only designs or when configuration sets it."
        ),
        default="",
        placeholder="mat_...",
    )
    configuration: ManufacturingConfiguration = SchemaField(
        description=(
            "Full manufacturing configuration: per-part material, tube profile, "
            "finish, powder coat, hole operations, welds and accepted risks. "
            "A non-empty material_id above overrides defaults.material_id."
        ),
        default_factory=ManufacturingConfiguration,
        advanced=True,
    )
    quantity_options: list[int] = SchemaField(
        description="Up to ten other quantities to price for comparison.",
        default_factory=list,
        advanced=True,
    )
    additional_items: list[QuoteItemRequest] = SchemaField(
        description="Further configured designs to price in the same basket.",
        default_factory=list,
        advanced=True,
    )
    client_reference_id: str = SchemaField(
        description="Your own reference for this item, echoed back on the result.",
        default="",
        advanced=True,
    )


def build_items(input_data: RMFGBasketInput) -> list[QuoteItemRequest]:
    """Turn the flat block inputs into the API's ``items[]`` basket."""
    item = QuoteItemRequest(
        design_id=input_data.design_id,
        quantity=input_data.quantity,
        configuration=input_data.configuration.with_material(input_data.material_id),
        client_reference_id=input_data.client_reference_id or None,
        quantity_options=input_data.quantity_options,
    )
    return [item, *input_data.additional_items]
