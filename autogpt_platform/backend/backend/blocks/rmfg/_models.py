"""Response models for the RMFG catalog, design analysis and DFM resources.

Every field carries a default so a payload from a newer API version, or one
with fields the platform does not surface, still validates.
"""

from typing import Any, Optional

from pydantic import ConfigDict

from backend.sdk import BaseModel, Field

from ._types import (
    DesignStatus,
    ManufacturabilityStatus,
    ManufacturingConfiguration,
    Process,
)


class Material(BaseModel):
    """A sheet-metal stock option. Use ``id`` as ``material_id``."""

    id: str = ""
    material: str = ""
    type: str = ""
    thickness_in: float = 0
    thickness_mm: float = 0
    display_thickness: Optional[str] = None
    description: Optional[str] = None
    bendable: bool = False
    weight_g_per_cm2: float = 0


class TubeProfile(BaseModel):
    """A tube-laser stock profile. Use ``id`` as ``tube_profile_id``."""

    id: str = ""
    material: str = ""
    type: str = ""
    shape: str = ""
    display_name: str = ""
    outer_width_mm: Optional[float] = None
    outer_height_mm: Optional[float] = None
    outer_diameter_mm: Optional[float] = None
    wall_thickness_mm: Optional[float] = None
    default_stock_length_mm: Optional[float] = None


class Finish(BaseModel):
    """A mechanical finish. Use ``id`` as ``finish_id``."""

    id: str = ""
    name: str = ""
    slug: str = ""
    description: Optional[str] = None
    category: Optional[str] = None
    processes: list[Process] = Field(default_factory=list)
    base_price_cents: int = 0
    price_per_cm2_cents: float = 0


class PowderCoatColor(BaseModel):
    """A powder-coat color. Use ``id`` as ``powder_coat_color_id``."""

    id: str = ""
    name: str = ""
    slug: str = ""
    hex_color: str = ""
    price_multiplier: float = 1
    available: bool = True


class HardwareOption(BaseModel):
    """A tap, stud, nut or standoff catalog entry.

    The four families share ``id``/``size`` but differ elsewhere, so the
    remaining catalog fields are passed through as-is.
    """

    model_config = ConfigDict(extra="allow")

    id: str = ""
    size: Optional[str] = None
    hole_diameter_mm: Optional[float] = None


class Dimensions(BaseModel):
    length_mm: float = 0
    width_mm: float = 0
    height_mm: float = 0


class Hole(BaseModel):
    """A detected hole or cutout; IDs are stable for the life of the design."""

    id: int = 0
    shape: str = "round"
    diameter_mm: float = 0
    face_id: Optional[int] = None
    nearest_bend_id: Optional[int] = None
    distance_to_nearest_edge_mm: Optional[float] = None
    distance_to_nearest_bend_mm: Optional[float] = None


class Bend(BaseModel):
    id: int = 0
    angle_degrees: float = 0
    radius_mm: float = 0
    length_mm: float = 0


class Part(BaseModel):
    """One unique part of a design; ``instance_count`` says how often it occurs."""

    id: str = ""
    name: str = ""
    suggested_process: Process = Process.SHEET_METAL
    instance_count: int = 1
    formed_dimensions: Dimensions = Field(default_factory=Dimensions)
    flat_pattern_dimensions: Optional[Dimensions] = None
    detected_thickness_mm: Optional[float] = None
    bend_count: int = 0
    bends: list[Bend] = Field(default_factory=list)
    hole_count: int = 0
    holes: list[Hole] = Field(default_factory=list)
    surface_area_cm2: Optional[float] = None
    cut_length_mm: Optional[float] = None
    model_url: str = ""
    image_url: Optional[str] = None


class PartInstance(BaseModel):
    id: str = ""
    part_id: str = ""
    instance_index: int = 0


class ResourceError(BaseModel):
    code: str = ""
    message: str = ""


class Design(BaseModel):
    """An analyzed STEP upload. Parts and instances are filled once ``ready``."""

    id: str = ""
    status: DesignStatus = DesignStatus.QUEUED
    name: str = ""
    formed_dimensions: Optional[Dimensions] = None
    parts: list[Part] = Field(default_factory=list)
    part_instances: list[PartInstance] = Field(default_factory=list)
    model_url: Optional[str] = None
    image_url: Optional[str] = None
    review_url: Optional[str] = None
    review_link_id: Optional[str] = None
    created_at: Optional[str] = None
    error: Optional[ResourceError] = None


class Requirement(BaseModel):
    """A missing selection or decision that keeps a resource from being ready."""

    code: str = ""
    message: str = ""
    part_id: Optional[str] = None
    part_ids: list[str] = Field(default_factory=list)
    field: Optional[str] = None
    allowed_values_url: Optional[str] = None


class DFMIssue(BaseModel):
    code: str = ""
    message: str = ""
    severity: str = "blocking"
    accepted: bool = False
    source: str = ""
    part_id: Optional[str] = None
    hole_id: Optional[int] = None
    bend_id: Optional[int] = None
    operation: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityOption(BaseModel):
    id: str = ""
    name: str = ""
    available: bool = True
    unavailable_reason: Optional[str] = None


class OperationCapability(BaseModel):
    available: bool = False
    unavailable_reason: Optional[str] = None
    options: list[CapabilityOption] = Field(default_factory=list)


class HoleOperationCapability(BaseModel):
    available: bool = False
    unavailable_reason: Optional[str] = None
    option_ids: list[str] = Field(default_factory=list)


class HoleCapabilities(BaseModel):
    hole_id: int = 0
    taps: HoleOperationCapability = Field(default_factory=HoleOperationCapability)
    studs: HoleOperationCapability = Field(default_factory=HoleOperationCapability)
    nuts: HoleOperationCapability = Field(default_factory=HoleOperationCapability)
    standoffs: HoleOperationCapability = Field(default_factory=HoleOperationCapability)
    countersinks: HoleOperationCapability = Field(
        default_factory=HoleOperationCapability
    )


class PartCapabilities(BaseModel):
    """What each configuration field may be set to for one part."""

    taps: OperationCapability = Field(default_factory=OperationCapability)
    studs: OperationCapability = Field(default_factory=OperationCapability)
    nuts: OperationCapability = Field(default_factory=OperationCapability)
    standoffs: OperationCapability = Field(default_factory=OperationCapability)
    countersinks: OperationCapability = Field(default_factory=OperationCapability)
    powder_coat_color_id: OperationCapability = Field(
        default_factory=OperationCapability
    )
    finish_id: OperationCapability = Field(default_factory=OperationCapability)
    holes: list[HoleCapabilities] = Field(default_factory=list)


class ManufacturingReviewWarning(BaseModel):
    """An advisory for manual preparation; never a checkout requirement."""

    design_id: str = ""
    code: str = ""
    message: str = ""


class ProductionFilesStatus(BaseModel):
    status: str = "skipped"
    reason: Optional[str] = None
    error: Optional[ResourceError] = None
    review_warning: Optional[ManufacturingReviewWarning] = None
    corrections_applied: bool = False


class PartDFM(BaseModel):
    part_id: str = ""
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    issues: list[DFMIssue] = Field(default_factory=list)
    capabilities: Optional[PartCapabilities] = None
    image_url: Optional[str] = None
    production_warnings: list[str] = Field(default_factory=list)


class DFMReport(BaseModel):
    """An immutable manufacturability evaluation of one configuration."""

    id: str = ""
    design_id: str = ""
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    review_url: Optional[str] = None
    review_link_id: Optional[str] = None
    configuration: ManufacturingConfiguration = Field(
        default_factory=ManufacturingConfiguration
    )
    parts: list[PartDFM] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    assembly_issues: list[DFMIssue] = Field(default_factory=list)
    production_files: ProductionFilesStatus = Field(
        default_factory=ProductionFilesStatus
    )

    @property
    def issues(self) -> list[DFMIssue]:
        """Every finding across parts and the assembly, flattened."""
        return [
            issue for part in self.parts for issue in part.issues
        ] + self.assembly_issues
