"""Response models for RMFG quotes, carts, orders and review links."""

from typing import Optional

from backend.sdk import BaseModel, Field

from ._models import DFMReport, ManufacturingReviewWarning, Requirement, ResourceError
from ._types import (
    CartStatus,
    ManufacturabilityStatus,
    ManufacturingConfiguration,
    OrderStatus,
    PaymentStatus,
    Process,
    QuoteItemRequest,
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
    material_id: Optional[str] = None
    tube_profile_id: Optional[str] = None
    breakdown: list[LineItemBreakdown] = Field(default_factory=list)


class QuotedDesign(BaseModel):
    design_id: str = ""
    quantity: int = 1
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    client_reference_id: Optional[str] = None
    unit_amount_cents: Optional[int] = None
    amount_cents: Optional[int] = None
    assembly_operations_amount_cents: int = 0
    quantity_options: list[QuantityOption] = Field(default_factory=list)
    dfm: DFMReport = Field(default_factory=DFMReport)
    requirements: list[Requirement] = Field(default_factory=list)
    line_items: list[QuoteLineItem] = Field(default_factory=list)


class Quote(BaseModel):
    """An immutable priced evaluation of a basket of designs. Amounts are USD cents."""

    id: str = ""
    status: QuoteStatus = QuoteStatus.PROCESSING
    design_id: Optional[str] = None
    currency: str = "usd"
    amount_subtotal_cents: Optional[int] = None
    amount_shipping_cents: Optional[int] = None
    amount_tax_cents: Optional[int] = None
    amount_total_cents: Optional[int] = None
    items: list[QuotedDesign] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    fulfillment: Optional[FulfillmentEstimate] = None
    created_at: str = ""
    expires_at: Optional[str] = None
    error: Optional[ResourceError] = None

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
    amount_shipping_cents: Optional[int] = None
    amount_production_surcharge_cents: Optional[int] = None
    amount_tax_cents: Optional[int] = None
    tax_status: str = "destination_required"
    amount_total_cents: int = 0


class CartPayment(BaseModel):
    cart_id: str = ""
    status: PaymentStatus = PaymentStatus.PROCESSING
    payment_intent_id: Optional[str] = None
    amount_total_cents: int = 0
    order_id: Optional[str] = None
    paid_at: Optional[str] = None


class Cart(BaseModel):
    """A server-side basket that re-quotes on every change.

    ``cart_url`` embeds an unguessable token: anyone holding it can check out.
    """

    id: str = ""
    revision: int = 0
    status: CartStatus = CartStatus.OPEN
    cart_url: str = ""
    items: list[QuoteItemRequest] = Field(default_factory=list)
    ship_to: Optional[ShipTo] = None
    shipping_option_id: Optional[str] = None
    selected_shipping_option: Optional[ShippingOption] = None
    quote: Quote = Field(default_factory=Quote)
    totals: CartTotals = Field(default_factory=CartTotals)
    manufacturing_warnings: list[ManufacturingReviewWarning] = Field(
        default_factory=list
    )
    order_id: Optional[str] = None
    payment: Optional[CartPayment] = None
    client_reference_id: Optional[str] = None
    created_at: str = ""
    expires_at: Optional[str] = None


class OrderTracking(BaseModel):
    carrier: Optional[str] = None
    service: Optional[str] = None
    number: Optional[str] = None
    url: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None


class OrderLineItem(BaseModel):
    id: str = ""
    type: str = "part"
    part_id: Optional[str] = None
    design_id: Optional[str] = None
    name: Optional[str] = None
    quantity: int = 0
    unit_amount_cents: int = 0
    amount_cents: int = 0
    material_id: Optional[str] = None
    tube_profile_id: Optional[str] = None
    powder_coat_color_id: Optional[str] = None
    finish_id: Optional[str] = None
    hole_operation_count: int = 0


class OrderEvent(BaseModel):
    type: str = ""
    at: str = ""


class Order(BaseModel):
    """A paid manufacturing order and what has happened to it since."""

    id: str = ""
    status: OrderStatus = OrderStatus.RECEIVED
    created_at: str = ""
    cart_id: Optional[str] = None
    quote_id: Optional[str] = None
    amount_subtotal_cents: Optional[int] = None
    amount_discount_cents: Optional[int] = None
    amount_tax_cents: Optional[int] = None
    amount_total_cents: Optional[int] = None
    expedited: bool = False
    production_hold: bool = False
    estimated_ship_date: Optional[str] = None
    delivered_at: Optional[str] = None
    tracking: Optional[OrderTracking] = None
    line_items: list[OrderLineItem] = Field(default_factory=list)
    events: list[OrderEvent] = Field(default_factory=list)
    po_number: Optional[str] = None


class ReviewLink(BaseModel):
    """A website hand-off where a person can adjust and save a configuration."""

    id: str = ""
    status: str = "open"
    design_id: str = ""
    dfm_id: Optional[str] = None
    review_url: str = ""
    configuration: ManufacturingConfiguration = Field(
        default_factory=ManufacturingConfiguration
    )
    configuration_updated_at: Optional[str] = None
    created_at: str = ""
    expires_at: Optional[str] = None
