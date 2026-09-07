"""Enums and request-side models for the RMFG Manufacturing API (v2026-09-01).

Field names mirror the API so a model's ``model_dump`` is the request body.
Response-side models live in ``_models`` and ``_models_commerce``.
"""

from enum import Enum
from typing import Optional

from backend.sdk import BaseModel, Field


class Process(str, Enum):
    SHEET_METAL = "sheet_metal"
    TUBE_LASER = "tube_laser"


class DesignStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ManufacturabilityStatus(str, Enum):
    """Outcome of a DFM evaluation. Only ``ready`` can be checked out."""

    REQUIRES_INPUT = "requires_input"
    READY = "ready"
    BLOCKED = "blocked"


class QuoteStatus(str, Enum):
    PROCESSING = "processing"
    REQUIRES_INPUT = "requires_input"
    READY = "ready"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    FAILED = "failed"


class CartStatus(str, Enum):
    OPEN = "open"
    CHECKED_OUT = "checked_out"
    EXPIRED = "expired"


class PaymentStatus(str, Enum):
    PAID = "paid"
    PROCESSING = "processing"
    FAILED = "failed"
    REFUNDED = "refunded"


class OrderStatus(str, Enum):
    RECEIVED = "received"
    IN_PRODUCTION = "in_production"
    READY_FOR_PICKUP = "ready_for_pickup"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class HardwareKind(str, Enum):
    """Catalog families under ``/v1/hardware/{kind}``."""

    TAPS = "taps"
    STUDS = "studs"
    NUTS = "nuts"
    STANDOFFS = "standoffs"


class HoleSide(str, Enum):
    """Sheet face an operation is applied from; ``top`` is the hole's normal."""

    TOP = "top"
    BOTTOM = "bottom"


class CountersinkAngle(int, Enum):
    DEG_82 = 82
    DEG_90 = 90


class AcceptedRisk(str, Enum):
    """DFM findings a customer can accept responsibility for."""

    HOLE_TOO_CLOSE_TO_BEND = "hole_too_close_to_bend"
    COUNTERSINK_TOO_CLOSE_TO_BEND = "countersink_too_close_to_bend"


class WeldMethod(str, Enum):
    AUTO = "auto"
    MIG = "mig"
    TIG = "tig"


class WeldFinish(str, Enum):
    NONE = "none"
    ROUGH = "rough"
    FINE = "fine"


class JointAccess(str, Enum):
    OPEN = "open"
    TIGHT = "tight"
    BLIND = "blind"


class PaymentType(str, Enum):
    CARD_ON_FILE = "card_on_file"
    PAYMENT_METHOD = "payment_method"


class TapOperation(BaseModel):
    hole_id: int = Field(description="Hole ID from the analyzed part's holes[]")
    tap_id: str = Field(description="Catalog ID from List Hardware (taps)")


class StudOperation(BaseModel):
    hole_id: int = Field(description="Hole ID from the analyzed part's holes[]")
    stud_id: str = Field(description="Catalog ID from List Hardware (studs)")
    side: HoleSide = Field(
        default=HoleSide.TOP, description="Sheet face the stud is installed from"
    )


class NutOperation(BaseModel):
    hole_id: int = Field(description="Hole ID from the analyzed part's holes[]")
    nut_id: str = Field(description="Catalog ID from List Hardware (nuts)")
    side: HoleSide = Field(
        default=HoleSide.TOP, description="Sheet face the nut is installed from"
    )


class StandoffOperation(BaseModel):
    hole_id: int = Field(description="Hole ID from the analyzed part's holes[]")
    standoff_id: str = Field(description="Catalog ID from List Hardware (standoffs)")
    side: HoleSide = Field(
        default=HoleSide.TOP, description="Sheet face the standoff is installed from"
    )


class CountersinkOperation(BaseModel):
    hole_id: int = Field(description="Hole ID from the analyzed part's holes[]")
    outer_diameter_mm: float = Field(description="Countersink outer diameter in mm")
    angle: CountersinkAngle = Field(
        default=CountersinkAngle.DEG_82, description="Included angle in degrees"
    )
    side: HoleSide = Field(
        default=HoleSide.TOP, description="Sheet face the countersink is cut into"
    )


class WeldingOperation(BaseModel):
    """A welded seam. One part instance ID is a seam on that instance; two or more join them."""

    type: str = "weld"
    method: WeldMethod = WeldMethod.AUTO
    weld_length_mm: float = Field(description="Total seam length in mm")
    part_instance_ids: list[str] = Field(
        description="Part instance IDs from the design's part_instances[]"
    )
    weld_finish: WeldFinish = WeldFinish.NONE
    joint_access: JointAccess = JointAccess.OPEN


class DefaultSelection(BaseModel):
    """Choices applied to every compatible part unless a part overrides them."""

    material_id: Optional[str] = Field(
        default=None, description="Sheet-metal stock; applies to sheet parts only"
    )
    tube_profile_id: Optional[str] = Field(
        default=None, description="Tube stock; applies to tube-laser parts only"
    )
    powder_coat_color_id: Optional[str] = None
    finish_id: Optional[str] = None


class PartConfiguration(BaseModel):
    """Choices for one unique part; set fields override the defaults for it."""

    part_id: str = Field(description="Part ID from the analyzed design's parts[]")
    material_id: Optional[str] = None
    tube_profile_id: Optional[str] = None
    powder_coat_color_id: Optional[str] = None
    finish_id: Optional[str] = None
    taps: list[TapOperation] = Field(default_factory=list)
    studs: list[StudOperation] = Field(default_factory=list)
    nuts: list[NutOperation] = Field(default_factory=list)
    standoffs: list[StandoffOperation] = Field(default_factory=list)
    countersinks: list[CountersinkOperation] = Field(default_factory=list)


class ManufacturingConfiguration(BaseModel):
    """Manufacturing intent shared by DFM reports, quotes, carts and review links.

    Precedence is part override, then defaults, then unset. Unset fields are
    omitted from the request, so a missing selection surfaces as a
    ``requires_input`` finding rather than an HTTP error.
    """

    defaults: DefaultSelection = Field(default_factory=DefaultSelection)
    parts: list[PartConfiguration] = Field(default_factory=list)
    assembly_operations: list[WeldingOperation] = Field(default_factory=list)
    accepted_risks: list[AcceptedRisk] = Field(default_factory=list)

    def with_material(self, material_id: str) -> "ManufacturingConfiguration":
        """Return a copy whose default sheet material is ``material_id``."""
        if not material_id:
            return self
        defaults = self.defaults.model_copy(update={"material_id": material_id})
        return self.model_copy(update={"defaults": defaults})

    def to_payload(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


class QuoteItemRequest(BaseModel):
    """One configured design in a quote or cart basket."""

    design_id: str = Field(description="Design ID from Analyze Design")
    quantity: int = Field(default=1, ge=1, description="Completed units of the design")
    configuration: ManufacturingConfiguration = Field(
        default_factory=ManufacturingConfiguration
    )
    client_reference_id: Optional[str] = None
    quantity_options: list[int] = Field(
        default_factory=list,
        description="Up to ten other quantities to price alongside quantity",
    )

    def to_payload(self) -> dict:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["configuration"] = self.configuration.to_payload()
        return payload


class ShipTo(BaseModel):
    name: str = Field(description="Recipient name")
    company: Optional[str] = None
    street1: str = Field(description="Street address line 1")
    street2: Optional[str] = None
    city: str
    state: str = Field(description="Two-letter state code, e.g. TX")
    postal_code: str
    country: str = Field(default="US", description="Two-letter country code")
    phone: Optional[str] = None
    email: Optional[str] = None

    def to_payload(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)
