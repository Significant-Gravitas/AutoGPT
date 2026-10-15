"""Response models for RMFG quotes, carts, orders and review links."""

from backend.sdk import BaseModel, Field

from ._models import DFMReport, ManufacturingReviewWarning, Requirement, ResourceError
from ._types import (
    CartStatus,
    ManufacturabilityStatus,
    ManufacturingConfiguration,
    OrderStatus,
    PaymentStatus,
    Process,
    QuoteStatus,
    ShipTo,
)


class ShippingOption(BaseModel):
    """A carrier service plus production speed. Use ``id`` as ``shipping_option_id``."""

    id: str = ""
    production_speed: str = "standard"
    carrier: str = ""
    service: str = ""
    amount_cents: int = 0
    production_surcharge_cents: int = 0
    planned_ship_date: str = ""
    estimated_delivery_date: str = ""
    delivery_days: int = 0
    delivery_date_guaranteed: bool = False
    requires_freight: bool = False
    rate_source: str = "estimated"


class ProductionEstimate(BaseModel):
    speed: str = "standard"
    production_days: int = 0
    estimated_ship_date: str = ""
    surcharge_cents: int = 0


class FulfillmentEstimate(BaseModel):
    production: list[ProductionEstimate] = Field(default_factory=list)
    shipping_status: str = "destination_required"
    shipping_options: list[ShippingOption] = Field(default_factory=list)
    requires_freight: bool = False
    freight_reasons: list[str] = Field(default_factory=list)


class QuantityOption(BaseModel):
    """Price of the same configuration at another quantity."""

    quantity: int = 0
    unit_amount_cents: int = 0
    amount_cents: int = 0


class LineItemBreakdown(BaseModel):
    type: str = "base"
    unit_amount_cents: int = 0
    amount_cents: int = 0


class QuoteLineItem(BaseModel):
    """Pricing for one unique part; quantity is design quantity × instance_count."""

    part_id: str = ""
    process: Process = Process.SHEET_METAL
    quantity: int = 0
    unit_amount_cents: int = 0
    amount_cents: int = 0
    material_id: str | None = None
    tube_profile_id: str | None = None
    breakdown: list[LineItemBreakdown] = Field(default_factory=list)


class QuotedDesign(BaseModel):
    design_id: str = ""
    quantity: int = 1
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    client_reference_id: str | None = None
    unit_amount_cents: int | None = None
    amount_cents: int | None = None
    assembly_operations_amount_cents: int = 0
    quantity_options: list[QuantityOption] = Field(default_factory=list)
    dfm: DFMReport = Field(default_factory=DFMReport)
    requirements: list[Requirement] = Field(default_factory=list)
    line_items: list[QuoteLineItem] = Field(default_factory=list)


class Quote(BaseModel):
    """An immutable priced evaluation of a basket of designs. Amounts are USD cents."""

    id: str = ""
    status: QuoteStatus = QuoteStatus.PROCESSING
    design_id: str | None = None
    currency: str = "usd"
    amount_subtotal_cents: int | None = None
    amount_shipping_cents: int | None = None
    amount_tax_cents: int | None = None
    amount_total_cents: int | None = None
    items: list[QuotedDesign] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    fulfillment: FulfillmentEstimate | None = None
    created_at: str = ""
    expires_at: str | None = None
    error: ResourceError | None = None

    @property
    def shipping_options(self) -> list[ShippingOption]:
        return self.fulfillment.shipping_options if self.fulfillment else []

    @property
    def all_requirements(self) -> list[Requirement]:
        """Quote-level requirements plus every item's, flattened."""
        return self.requirements + [
            requirement for item in self.items for requirement in item.requirements
        ]


class CartTotals(BaseModel):
    """Every amount checkout charges, in USD cents."""

    amount_subtotal_cents: int = 0
    amount_shipping_cents: int | None = None
    amount_production_surcharge_cents: int | None = None
    amount_tax_cents: int | None = None
    tax_status: str = "destination_required"
    amount_total_cents: int = 0


class CartPayment(BaseModel):
    cart_id: str = ""
    status: PaymentStatus = PaymentStatus.PROCESSING
    payment_intent_id: str | None = None
    amount_total_cents: int = 0
    order_id: str | None = None
    paid_at: str | None = None


class CartItem(BaseModel):
    """One configured design as a cart echoes it back.

    The API documents this as the request item shape, but a response model
    must never reject what the server sends, so every field has a default.
    """

    design_id: str = ""
    quantity: int = 1
    configuration: ManufacturingConfiguration = Field(
        default_factory=ManufacturingConfiguration
    )
    client_reference_id: str | None = None
    quantity_options: list[int] = Field(default_factory=list)


class Cart(BaseModel):
    """A server-side basket that re-quotes on every change.

    ``cart_url`` embeds an unguessable token: anyone holding it can check out.
    """

    id: str = ""
    revision: int = 0
    status: CartStatus = CartStatus.OPEN
    cart_url: str = ""
    items: list[CartItem] = Field(default_factory=list)
    ship_to: ShipTo | None = None
    shipping_option_id: str | None = None
    selected_shipping_option: ShippingOption | None = None
    quote: Quote = Field(default_factory=Quote)
    totals: CartTotals = Field(default_factory=CartTotals)
    manufacturing_warnings: list[ManufacturingReviewWarning] = Field(
        default_factory=list
    )
    order_id: str | None = None
    payment: CartPayment | None = None
    client_reference_id: str | None = None
    created_at: str = ""
    expires_at: str | None = None


class OrderTracking(BaseModel):
    carrier: str | None = None
    service: str | None = None
    number: str | None = None
    url: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None


class OrderLineItem(BaseModel):
    id: str = ""
    type: str = "part"
    part_id: str | None = None
    design_id: str | None = None
    name: str | None = None
    quantity: int = 0
    unit_amount_cents: int = 0
    amount_cents: int = 0
    material_id: str | None = None
    tube_profile_id: str | None = None
    powder_coat_color_id: str | None = None
    finish_id: str | None = None
    hole_operation_count: int = 0


class OrderEvent(BaseModel):
    type: str = ""
    at: str = ""


class Order(BaseModel):
    """A paid manufacturing order and what has happened to it since."""

    id: str = ""
    status: OrderStatus = OrderStatus.RECEIVED
    created_at: str = ""
    cart_id: str | None = None
    quote_id: str | None = None
    amount_subtotal_cents: int | None = None
    amount_discount_cents: int | None = None
    amount_tax_cents: int | None = None
    amount_total_cents: int | None = None
    expedited: bool = False
    production_hold: bool = False
    estimated_ship_date: str | None = None
    delivered_at: str | None = None
    tracking: OrderTracking | None = None
    line_items: list[OrderLineItem] = Field(default_factory=list)
    events: list[OrderEvent] = Field(default_factory=list)
    po_number: str | None = None


class ReviewLink(BaseModel):
    """A website hand-off where a person can adjust and save a configuration."""

    id: str = ""
    status: str = "open"
    design_id: str = ""
    dfm_id: str | None = None
    review_url: str = ""
    configuration: ManufacturingConfiguration = Field(
        default_factory=ManufacturingConfiguration
    )
    configuration_updated_at: str | None = None
    created_at: str = ""
    expires_at: str | None = None
