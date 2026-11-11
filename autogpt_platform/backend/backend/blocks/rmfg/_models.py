"""Response models for the RMFG catalog, design analysis and DFM resources.

Every field carries a default so a payload from a newer API version, or one
with fields the platform does not surface, still validates.
"""

from typing import Any

from pydantic import ConfigDict, field_validator

from backend.sdk import BaseModel, Field

from ._types import (
    RMFG_API_URL,
    DesignStatus,
    ManufacturabilityStatus,
    ManufacturingConfiguration,
    Process,
)


def absolute_api_url(value: str | None) -> str | None:
    """RMFG returns links to its own resources as paths such as ``/v1/...``.

    Downstream blocks and people need a full URL, so prefix the API origin.
    Website links and other absolute URLs pass through untouched.
    """
    if value and value.startswith("/"):
        return f"{RMFG_API_URL}{value}"
    return value


class Material(BaseModel):
    """A sheet-metal stock option. Use ``id`` as ``material_id``."""

    id: str = ""
    material: str = ""
    type: str = ""
    thickness_in: float = 0
    thickness_mm: float = 0
    display_thickness: str | None = None
    description: str | None = None
    bendable: bool = False
    weight_g_per_cm2: float = 0


class TubeProfile(BaseModel):
    """A tube-laser stock profile. Use ``id`` as ``tube_profile_id``."""

    id: str = ""
    material: str = ""
    type: str = ""
    shape: str = ""
    display_name: str = ""
    outer_width_mm: float | None = None
    outer_height_mm: float | None = None
    outer_diameter_mm: float | None = None
    wall_thickness_mm: float | None = None
    default_stock_length_mm: float | None = None


class Finish(BaseModel):
    """A mechanical finish. Use ``id`` as ``finish_id``."""

    id: str = ""
    name: str = ""
    slug: str = ""
    description: str | None = None
    category: str | None = None
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
    size: str | None = None
    hole_diameter_mm: float | None = None


class Dimensions(BaseModel):
    length_mm: float = 0
    width_mm: float = 0
    height_mm: float = 0


class Hole(BaseModel):
    """A detected hole or cutout; IDs are stable for the life of the design."""

    id: int = 0
    shape: str = "round"
    diameter_mm: float = 0
    face_id: int | None = None
    nearest_bend_id: int | None = None
    distance_to_nearest_edge_mm: float | None = None
    distance_to_nearest_bend_mm: float | None = None


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
    flat_pattern_dimensions: Dimensions | None = None
    detected_thickness_mm: float | None = None
    bend_count: int = 0
    bends: list[Bend] = Field(default_factory=list)
    hole_count: int = 0
    holes: list[Hole] = Field(default_factory=list)
    surface_area_cm2: float | None = None
    cut_length_mm: float | None = None
    model_url: str = ""
    flat_pattern_model_url: str | None = None
    image_url: str | None = None

    @field_validator("model_url", "flat_pattern_model_url", "image_url")
    @classmethod
    def _absolute(cls, value: str | None) -> str | None:
        return absolute_api_url(value)


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
    formed_dimensions: Dimensions | None = None
    parts: list[Part] = Field(default_factory=list)
    part_instances: list[PartInstance] = Field(default_factory=list)
    model_url: str | None = None
    image_url: str | None = None
    review_url: str | None = None
    review_link_id: str | None = None
    created_at: str | None = None
    error: ResourceError | None = None

    @field_validator("model_url", "image_url")
    @classmethod
    def _absolute(cls, value: str | None) -> str | None:
        return absolute_api_url(value)


class Requirement(BaseModel):
    """A missing selection or decision that keeps a resource from being ready."""

    code: str = ""
    message: str = ""
    part_id: str | None = None
    part_ids: list[str] = Field(default_factory=list)
    field: str | None = None
    allowed_values_url: str | None = None

    @field_validator("allowed_values_url")
    @classmethod
    def _absolute(cls, value: str | None) -> str | None:
        return absolute_api_url(value)


class DFMIssue(BaseModel):
    code: str = ""
    message: str = ""
    severity: str = "blocking"
    accepted: bool = False
    source: str = ""
    part_id: str | None = None
    hole_id: int | None = None
    bend_id: int | None = None
    operation: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityOption(BaseModel):
    id: str = ""
    name: str = ""
    available: bool = True
    unavailable_reason: str | None = None


class OperationCapability(BaseModel):
    available: bool = False
    unavailable_reason: str | None = None
    options: list[CapabilityOption] = Field(default_factory=list)


class HoleOperationCapability(BaseModel):
    available: bool = False
    unavailable_reason: str | None = None
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
    reason: str | None = None
    error: ResourceError | None = None
    review_warning: ManufacturingReviewWarning | None = None
    corrections_applied: bool = False


class PartDFM(BaseModel):
    part_id: str = ""
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    issues: list[DFMIssue] = Field(default_factory=list)
    capabilities: PartCapabilities | None = None
    image_url: str | None = None
    production_warnings: list[str] = Field(default_factory=list)

    @field_validator("image_url")
    @classmethod
    def _absolute(cls, value: str | None) -> str | None:
        return absolute_api_url(value)


class DFMReport(BaseModel):
    """An immutable manufacturability evaluation of one configuration."""

    id: str = ""
    design_id: str = ""
    status: ManufacturabilityStatus = ManufacturabilityStatus.READY
    review_url: str | None = None
    review_link_id: str | None = None
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
