"""The sandbox image CoPilot boxes run on, built on the E2B team on demand.

E2B fixes a sandbox's vCPU / RAM at template build time.  Its public
``desktop`` image is 8 vCPU / 8 GiB (about $0.53 an hour running); we
re-snapshot it at 1 vCPU / 2 GiB under our own alias, which is cheaper to
run than E2B's ``base`` (2 vCPU / 512 MiB) with four times the RAM, and it
already carries XFCE, Chrome, Firefox and VS Code, so a box can later turn a
screen on without changing image.  Nothing graphical starts at boot: a shell
box on this image idles at about 90 MiB.

Template aliases live per E2B team, so the first sandbox on a new team (or
key) has to build it.  ``ensure_template`` checks the alias and builds it
from ``desktop`` when missing (12-25 s, once per team), serialised through
Redis so parallel first turns don't each start a build.  Templates we don't
manage are left alone.
"""

import asyncio
import logging

from e2b import AsyncTemplate, Template
from e2b.template.types import BuildInfo
from pydantic import BaseModel, ConfigDict

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)


class TemplateSpec(BaseModel):
    """A template we build ourselves: *source* re-snapshotted at a size."""

    model_config = ConfigDict(frozen=True)

    alias: str
    source: str
    cpu_count: int
    memory_mb: int

    @property
    def tags(self) -> list[str]:
        """Image-side metadata.  ``default`` is what a bare alias resolves to."""
        # E2B tags take Docker-tag characters only, so no colons.
        return [
            "default",
            f"{self.cpu_count}x{self.memory_mb // 1024}",
            f"from-{self.source}",
            "built-by-platform",
        ]


DESKTOP_IMAGE = TemplateSpec(
    alias="agpt-desktop-1x2", source="desktop", cpu_count=1, memory_mb=2048
)
MANAGED_TEMPLATES: dict[str, TemplateSpec] = {DESKTOP_IMAGE.alias: DESKTOP_IMAGE}

# A build takes 12-25 s; the lock outlives a slow one so a crashed builder
# doesn't block the team for long either.
_BUILD_LOCK_TTL_SECONDS = 300
_BUILD_WAIT_SECONDS = 180
_BUILD_POLL_SECONDS = 2.0

# Aliases this process has already confirmed on the team: one API call per
# alias per process lifetime, not one per sandbox.
_ready: set[str] = set()


async def ensure_template(template: str, api_key: str) -> None:
    """Make sure *template* exists on the team before a sandbox is created from it.

    Only templates in ``MANAGED_TEMPLATES`` are ever built; anything else is
    assumed to be provisioned out of band and returns immediately.
    """
    spec = MANAGED_TEMPLATES.get(template)
    if spec is None or template in _ready:
        return
    if await AsyncTemplate.alias_exists(spec.alias, api_key=api_key):
        _ready.add(template)
        return

    redis = await get_redis_async()
    lock_key = f"e2b:template:{spec.alias}:build"
    if await redis.set(lock_key, "1", nx=True, ex=_BUILD_LOCK_TTL_SECONDS):
        try:
            await build_template(spec, api_key)
        finally:
            await redis.delete(lock_key)
    else:
        await _wait_until_exists(spec, api_key)
    _ready.add(template)


async def build_template(spec: TemplateSpec, api_key: str) -> BuildInfo:
    """Build *spec* on the team the key belongs to (blocks until ready)."""
    logger.info(
        "[E2B] Building template %s (%d vCPU / %d MiB) from %s",
        spec.alias,
        spec.cpu_count,
        spec.memory_mb,
        spec.source,
    )
    info = await AsyncTemplate.build(
        Template().from_template(spec.source),
        spec.alias,
        tags=spec.tags,
        cpu_count=spec.cpu_count,
        memory_mb=spec.memory_mb,
        api_key=api_key,
    )
    logger.info("[E2B] Built template %s (%s)", spec.alias, info.template_id)
    return info


def forget_ready_templates() -> None:
    """Drop the process-level cache (tests, or after a template is deleted)."""
    _ready.clear()


async def _wait_until_exists(spec: TemplateSpec, api_key: str) -> None:
    """Another process holds the build lock: wait for its build to land."""
    for _ in range(int(_BUILD_WAIT_SECONDS / _BUILD_POLL_SECONDS)):
        if await AsyncTemplate.alias_exists(spec.alias, api_key=api_key):
            return
        await asyncio.sleep(_BUILD_POLL_SECONDS)
    raise TimeoutError(
        f"E2B template {spec.alias} did not appear within {_BUILD_WAIT_SECONDS}s"
    )
