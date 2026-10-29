"""How much of the machine the stack takes.

Every backend service imports the whole backend, about 600 MB of it, so the
number of interpreters decides the footprint. Historical measurements of
eight services held 5.7 GB idle separately and under half of that in three
hosts. The current nine-service, five-host layout has not been remeasured.
A profile says how the services are grouped into processes and how many
agent runs and AutoPilot turns may be in flight at once. It is chosen from
the machine and can be set in config/settings.env:

    AUTOGPT_DESKTOP_PROFILE=compact | balanced | isolated

`isolated` is one process per service, as the appliance runs them, with the
backend's own pool sizes. It is the layout to fall back on, and to compare
against, when a merged process misbehaves.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("autogpt_desktop")

PROFILE_SETTING = "AUTOGPT_DESKTOP_PROFILE"
COMPACT = "compact"
BALANCED = "balanced"
ISOLATED = "isolated"
PROFILES = (COMPACT, BALANCED, ISOLATED)

# 8 GB or less is compact. A machine sold with 8 GB reports a little under
# or, with some firmware, a little over.
COMPACT_RAM_BYTES = int(8.5 * 1024**3)

# The scheduler keeps two connection pools of this size, with no overflow
# (backend/executor/scheduler.py). Upstream's default, set by every profile
# so that the connection budget below does not move when the default does.
SCHEDULER_POOLS = 2
SCHEDULER_DB_POOL_SIZE = 3

# Passed to the server on every start, so installs made before these existed
# get them too (postgresql.conf is written once, when the cluster is made).
# The connection budget at its largest (isolated): three Prisma engines of 5,
# the skills-catalog publisher's 5, the frontend's 10, the scheduler's two
# pools of 3, the runtime's own 2, and PostgreSQL's 3 reserved slots: 41.
POSTGRES_LIMITS = {
    "max_connections": "50",
    "max_worker_processes": "4",
    "max_parallel_workers": "0",
    "autovacuum_max_workers": "1",
}


@dataclass(frozen=True)
class Profile:
    name: str
    # Why this one: the machine it was sized for, or who asked for it.
    reason: str
    # Defaults for the backend's pools; a value in settings.env wins.
    backend_env: dict[str, str]

    @property
    def merged(self) -> bool:
        """Whether services share processes (apps.layout)."""
        return self.name != ISOLATED

    def describe(self) -> str:
        return f"Using the {self.name} profile ({self.reason})"


def choose(
    user: dict[str, str], ram_bytes: int | None = None, cores: int | None = None
) -> Profile:
    """The profile settings.env asks for, otherwise one that fits the machine."""
    ram_bytes = ram_bytes if ram_bytes is not None else _total_ram()
    cores = cores if cores is not None else (os.cpu_count() or 2)
    machine = f"{ram_bytes / 1024**3:.0f} GB of memory, {cores} cores"
    asked = user.get(PROFILE_SETTING, "").strip().lower()
    if asked in PROFILES:
        return _profile(asked, f"set in settings.env; {machine}", cores)
    if asked:
        logger.warning(
            f"{PROFILE_SETTING}={asked} is not one of {', '.join(PROFILES)}; "
            "choosing from the machine instead"
        )
    name = COMPACT if ram_bytes <= COMPACT_RAM_BYTES else BALANCED
    return _profile(name, machine, cores)


def _profile(name: str, reason: str, cores: int) -> Profile:
    pinned = {"SCHEDULER_DB_POOL_SIZE": str(SCHEDULER_DB_POOL_SIZE)}
    return Profile(
        name=name, reason=reason, backend_env={**_pools(name, cores), **pinned}
    )


def _pools(name: str, cores: int) -> dict[str, str]:
    """Threads, not processes: they cost little idle and bound the peak."""
    if name == COMPACT:
        return {"NUM_GRAPH_WORKERS": "4", "NUM_COPILOT_WORKERS": "2"}
    if name == BALANCED:
        return {
            "NUM_GRAPH_WORKERS": str(min(10, max(4, cores))),
            "NUM_COPILOT_WORKERS": str(min(5, max(2, cores // 2))),
        }
    return {}


def _total_ram() -> int:
    import psutil

    return psutil.virtual_memory().total
