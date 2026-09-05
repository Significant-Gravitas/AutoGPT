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

# Fallback sizing when E2B cannot tell us what a box actually got. Measured
# on our account on 2026-09-05 the public "desktop" template resolves to
# 8 vCPU / 8 GiB, so the meter prefers the live ``get_info()`` numbers and
# only falls back to these.
DESKTOP_VCPU = 8
DESKTOP_RAM_GIB = 8.0

Resources = tuple[int, float]  # (vCPU, RAM GiB)


def rate_basis(vcpu: float, ram_gib: float) -> str:
    return (
        f"e2b: ${USD_PER_VCPU_SECOND}/vCPU/s x {vcpu:g} vCPU "
        f"+ ${USD_PER_GIB_RAM_SECOND}/GiB-RAM/s x {ram_gib:g} GiB"
    )


RATE_BASIS = rate_basis(DESKTOP_VCPU, DESKTOP_RAM_GIB)


class CostMeter(BaseModel):
    provider: str = PROVIDER
    sandbox_id: str
    wall_time_s: float
    resources: dict[str, float] = {"vcpu": DESKTOP_VCPU, "ram_gib": DESKTOP_RAM_GIB}
    estimated_cost_usd: float
    rate_usd_per_hour_running: float
    rate_basis: str = RATE_BASIS


def usd_per_second(
    vcpu: float = DESKTOP_VCPU, ram_gib: float = DESKTOP_RAM_GIB
) -> float:
    return vcpu * USD_PER_VCPU_SECOND + ram_gib * USD_PER_GIB_RAM_SECOND


def build_cost_meter(
    sandbox_id: str, wall_time_s: float, resources: Resources | None = None
) -> CostMeter:
    """Meter *wall_time_s* on the box's real size when E2B reported it."""
    vcpu, ram_gib = resources or (DESKTOP_VCPU, DESKTOP_RAM_GIB)
    rate = usd_per_second(vcpu, ram_gib)
    return CostMeter(
        sandbox_id=sandbox_id,
        wall_time_s=round(wall_time_s, 3),
        resources={"vcpu": vcpu, "ram_gib": ram_gib},
        estimated_cost_usd=round(rate * wall_time_s, 6),
        rate_usd_per_hour_running=round(rate * 3600, 4),
        rate_basis=rate_basis(vcpu, ram_gib),
    )
