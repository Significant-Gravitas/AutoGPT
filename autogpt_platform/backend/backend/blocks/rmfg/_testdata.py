"""Shared block test fixtures.

Kept in its own module so block files never import test data from each other.
"""

from ._models import (
    Design,
    DFMIssue,
    DFMReport,
    Dimensions,
    Finish,
    HardwareOption,
    Hole,
    Material,
    Part,
    PartDFM,
    PowderCoatColor,
    Requirement,
    TubeProfile,
)
from ._models_commerce import (
    Cart,
    CartPayment,
    CartTotals,
    FulfillmentEstimate,
    Order,
    OrderTracking,
    Quote,
    QuotedDesign,
    QuoteLineItem,
    ReviewLink,
    ShippingOption,
)
from ._types import (
    CartStatus,
    DefaultSelection,
    DesignStatus,
    ManufacturabilityStatus,
    ManufacturingConfiguration,
    OrderStatus,
    PartConfiguration,
    PaymentStatus,
    Process,
    QuoteStatus,
    ShipTo,
)

# "ISO-10303-21;" — the first line of any STEP file, as a data URI.
TEST_STEP_DATA_URI = "data:application/step;base64,SVNPLTEwMzAzLTIxOw=="

TEST_MATERIAL = Material(
    id="mat_5052_0125",
    material="5052 Aluminum",
    type="aluminum",
    thickness_in=0.125,
    thickness_mm=3.175,
    display_thickness='0.125"',
    bendable=True,
    weight_g_per_cm2=0.857,
)

TEST_TUBE_PROFILE = TubeProfile(
    id="tube_sq_25x25x2",
    material="Mild Steel",
    type="steel",
    shape="square",
    display_name='1" square, 0.083" wall',
    outer_width_mm=25.4,
    outer_height_mm=25.4,
    wall_thickness_mm=2.1,
)

TEST_FINISH = Finish(
    id="fin_deburr",
    name="Deburr",
    slug="deburr",
    processes=[Process.SHEET_METAL],
    base_price_cents=0,
)

TEST_POWDER_COAT_COLOR = PowderCoatColor(
    id="pc_ral9005",
    name="Jet Black",
    slug="jet-black",
    hex_color="#0A0A0A",
)

TEST_HARDWARE_OPTION = HardwareOption.model_validate(
    {"id": "tap_m4", "size": "M4", "hole_diameter_mm": 3.3, "thread_standard": "metric"}
)

TEST_PART = Part(
    id="prt_bracket",
    name="bracket",
    suggested_process=Process.SHEET_METAL,
    instance_count=2,
    formed_dimensions=Dimensions(length_mm=120, width_mm=60, height_mm=30),
    detected_thickness_mm=3.0,
    bend_count=1,
    hole_count=2,
    holes=[Hole(id=1, diameter_mm=4.5), Hole(id=2, diameter_mm=4.5)],
    model_url="https://api.rmfg.com/v1/designs/dsn_bracket/parts/prt_bracket/model",
    image_url="https://api.rmfg.com/v1/designs/dsn_bracket/parts/prt_bracket/image",
)

TEST_DESIGN = Design(
    id="dsn_bracket",
    status=DesignStatus.READY,
    name="bracket.step",
    formed_dimensions=Dimensions(length_mm=120, width_mm=60, height_mm=30),
    parts=[TEST_PART],
    review_url="https://www.rmfg.com/review/abc",
    image_url="https://api.rmfg.com/v1/designs/dsn_bracket/image",
    created_at="2026-09-01T12:00:00Z",
)

TEST_PENDING_DESIGN = TEST_DESIGN.model_copy(
    update={"status": DesignStatus.PROCESSING, "parts": []}
)

TEST_CONFIGURATION = ManufacturingConfiguration(
    defaults=DefaultSelection(material_id=TEST_MATERIAL.id),
    parts=[PartConfiguration(part_id=TEST_PART.id)],
)

TEST_DFM_ISSUE = DFMIssue(
    code="hole_too_close_to_bend",
    message="Hole 2 is 2.1 mm from bend 1; 6 mm is recommended.",
    severity="warning",
    source="geometry",
    part_id=TEST_PART.id,
    hole_id=2,
    bend_id=1,
)

TEST_DFM_REPORT = DFMReport(
    id="dfm_001",
    design_id=TEST_DESIGN.id,
    status=ManufacturabilityStatus.READY,
    review_url="https://www.rmfg.com/review/abc?dfm=dfm_001",
    configuration=TEST_CONFIGURATION,
    parts=[PartDFM(part_id=TEST_PART.id, issues=[TEST_DFM_ISSUE])],
)

TEST_REQUIREMENT = Requirement(
    code="material_required",
    message="Select a material for part prt_bracket.",
    part_id=TEST_PART.id,
    field="material_id",
)

TEST_SHIPPING_OPTION = ShippingOption(
    id="ship_ups_ground_std",
    production_speed="standard",
    carrier="UPS",
    service="Ground",
    amount_cents=1850,
    planned_ship_date="2026-09-08",
    estimated_delivery_date="2026-09-11",
    delivery_days=3,
    rate_source="live",
)

TEST_QUOTE = Quote(
    id="qte_001",
    status=QuoteStatus.READY,
    design_id=TEST_DESIGN.id,
    amount_subtotal_cents=24800,
    amount_total_cents=24800,
    items=[
        QuotedDesign(
            design_id=TEST_DESIGN.id,
            quantity=10,
            unit_amount_cents=2480,
            amount_cents=24800,
            dfm=TEST_DFM_REPORT,
            line_items=[
                QuoteLineItem(
                    part_id=TEST_PART.id,
                    quantity=20,
                    unit_amount_cents=1240,
                    amount_cents=24800,
                    material_id=TEST_MATERIAL.id,
                )
            ],
        )
    ],
    fulfillment=FulfillmentEstimate(
        shipping_status="ready", shipping_options=[TEST_SHIPPING_OPTION]
    ),
    created_at="2026-09-01T12:05:00Z",
)

TEST_PROCESSING_QUOTE = TEST_QUOTE.model_copy(
    update={"status": QuoteStatus.PROCESSING, "items": []}
)

TEST_SHIP_TO = ShipTo(
    name="Ada Lovelace",
    street1="1 Analytical Way",
    city="Austin",
    state="TX",
    postal_code="78701",
)

TEST_CART = Cart(
    id="crt_001",
    revision=1,
    status=CartStatus.OPEN,
    cart_url="https://www.rmfg.com/cart/secret-token",
    ship_to=TEST_SHIP_TO,
    shipping_option_id=TEST_SHIPPING_OPTION.id,
    selected_shipping_option=TEST_SHIPPING_OPTION,
    quote=TEST_QUOTE,
    totals=CartTotals(
        amount_subtotal_cents=24800,
        amount_shipping_cents=1850,
        amount_tax_cents=2199,
        tax_status="calculated",
        amount_total_cents=28849,
    ),
    created_at="2026-09-01T12:06:00Z",
)

TEST_PAID_CART = TEST_CART.model_copy(
    update={
        "status": CartStatus.CHECKED_OUT,
        "order_id": "ord_001",
        "payment": CartPayment(
            cart_id="crt_001",
            status=PaymentStatus.PAID,
            amount_total_cents=28849,
            order_id="ord_001",
            paid_at="2026-09-01T12:10:00Z",
        ),
    }
)

TEST_ORDER = Order(
    id="ord_001",
    status=OrderStatus.SHIPPED,
    created_at="2026-09-01T12:10:00Z",
    cart_id=TEST_CART.id,
    quote_id=TEST_QUOTE.id,
    amount_total_cents=28849,
    estimated_ship_date="2026-09-08",
    tracking=OrderTracking(
        carrier="UPS",
        service="Ground",
        number="1Z999AA10123456784",
        url="https://www.ups.com/track?tracknum=1Z999AA10123456784",
        shipped_at="2026-09-08T16:00:00Z",
    ),
)

TEST_REVIEW_LINK = ReviewLink(
    id="rvl_001",
    design_id=TEST_DESIGN.id,
    review_url="https://www.rmfg.com/review/abc",
    configuration=TEST_CONFIGURATION,
    created_at="2026-09-01T12:07:00Z",
)
