"""Provider cost telemetry for the desktop sandbox blocks.

Every desktop block emits a ``cost_meter`` output estimating the underlying
provider cost of the run from E2B's published per-second rates, so real
infrastructure spend is observable per block execution.
"""

from pydantic import BaseModel

PROVIDER = "e2b"

# Published E2B usage rates (https://e2b.dev/pricing).
USD_PER_VCPU_SECOND = 0.000014
USD_PER_GIB_RAM_SECOND = 0.0000045

# The public "desktop" template's default resources; adjust if a custom
# template with different sizing is used.
DESKTOP_VCPU = 2
DESKTOP_RAM_GIB = 4

RATE_BASIS = (
    f"e2b: ${USD_PER_VCPU_SECOND}/vCPU/s x {DESKTOP_VCPU} vCPU "
    f"+ ${USD_PER_GIB_RAM_SECOND}/GiB-RAM/s x {DESKTOP_RAM_GIB} GiB"
)


class CostMeter(BaseModel):
    provider: str = PROVIDER
    sandbox_id: str
    wall_time_s: float
    resources: dict[str, float] = {"vcpu": DESKTOP_VCPU, "ram_gib": DESKTOP_RAM_GIB}
    estimated_cost_usd: float
    rate_usd_per_hour_running: float
    rate_basis: str = RATE_BASIS


def usd_per_second() -> float:
    return DESKTOP_VCPU * USD_PER_VCPU_SECOND + DESKTOP_RAM_GIB * USD_PER_GIB_RAM_SECOND


def build_cost_meter(sandbox_id: str, wall_time_s: float) -> CostMeter:
    rate = usd_per_second()
    return CostMeter(
        sandbox_id=sandbox_id,
        wall_time_s=round(wall_time_s, 3),
        estimated_cost_usd=round(rate * wall_time_s, 6),
        rate_usd_per_hour_running=round(rate * 3600, 4),
    )
