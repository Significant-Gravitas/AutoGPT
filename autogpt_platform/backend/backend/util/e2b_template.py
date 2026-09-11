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

"Exists" is not "ready": E2B registers an alias the moment a build is
requested, before the build has run, and a failed build leaves the alias in
place with nothing usable behind it.  Readiness therefore means the alias
resolves to a template with at least one ``ready`` build.
"""

import asyncio
import enum
import hashlib
import hmac
import logging
import uuid

from e2b import AsyncTemplate, Template
from e2b.api.client.api.templates import (
    get_templates_aliases_alias,
    get_templates_template_id,
)
from e2b.api.client.models import (
    TemplateAliasResponse,
    TemplateBuildStatus,
    TemplateWithBuilds,
)
from e2b.api.client_async import get_api_client
from e2b.connection_config import ConnectionConfig
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

# A build takes 12-25 s.  The build is cut off before the lock can expire, so
# the lock is only ever released by its owner (or by the TTL after a crash).
# Followers wait as long as the lock can live, so they never give up on a
# build that is still allowed to finish.
_BUILD_LOCK_TTL_SECONDS = 300
_BUILD_TIMEOUT_SECONDS = 240
_BUILD_WAIT_SECONDS = _BUILD_LOCK_TTL_SECONDS
_BUILD_POLL_SECONDS = 2.0

# Release or extend only if we still own the lock: a build that outlived the
# TTL must not touch the lock a later builder took.
_UNLOCK_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)
_EXTEND_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("expire", KEYS[1], ARGV[2]) else return 0 end'
)

# Templates this process has already confirmed ready, keyed by alias and
# team: one round of API calls per alias per team per process lifetime.
_ready: set[str] = set()
# Checks in flight in this process, so concurrent first turns share one.
_inflight: dict[str, "asyncio.Future[None]"] = {}


class TemplateState(enum.Enum):
    READY = "ready"  # a build finished; sandboxes can be created
    BUILDING = "building"  # a build is queued or running
    MISSING = "missing"  # no alias, or an alias with no usable build


async def ensure_template(template: str, api_key: str) -> None:
    """Make sure *template* is ready on the team before a sandbox is created from it.

    Only templates in ``MANAGED_TEMPLATES`` are ever built; anything else is
    assumed to be provisioned out of band and returns immediately.
    """
    spec = MANAGED_TEMPLATES.get(template)
    if spec is None:
        return
    cache_key = _scoped_key(spec, api_key)
    if cache_key in _ready:
        return
    check = _inflight.get(cache_key)
    if check is None:
        check = _inflight[cache_key] = asyncio.ensure_future(
            _check_or_provision(spec, api_key, cache_key)
        )
        try:
            await asyncio.shield(check)
        finally:
            _inflight.pop(cache_key, None)
    else:
        await asyncio.shield(check)


async def _check_or_provision(spec: TemplateSpec, api_key: str, cache_key: str) -> None:
    try:
        if await get_template_state(spec, api_key) is not TemplateState.READY:
            await _provision(spec, api_key)
    except Exception:
        logger.error(
            "[E2B] Template %s is not available on this team and could not be "
            "built. Building it needs an E2B plan that allows template builds "
            "with a 20 GiB build disk. Set CHAT_E2B_SANDBOX_TEMPLATE=base to "
            "use E2B's stock image instead.",
            spec.alias,
        )
        raise
    _ready.add(cache_key)


async def _provision(spec: TemplateSpec, api_key: str) -> None:
    """Build *spec* on the team, or wait for whichever process is building it."""
    redis = await get_redis_async()
    lock_key = f"e2b:template:{_scoped_key(spec, api_key)}:build"
    token = uuid.uuid4().hex
    if not await redis.set(lock_key, token, nx=True, ex=_BUILD_LOCK_TTL_SECONDS):
        if await _wait_until_ready(spec, api_key, lock_key) is not TemplateState.READY:
            raise RuntimeError(
                f"E2B template {spec.alias} was not built by the process "
                "holding the build lock"
            )
        return
    try:
        # Re-check under the lock: the previous holder may have just finished,
        # or a build started elsewhere (the CLI, a crashed holder) may be
        # running.  Only a genuinely missing template gets a new build.
        state = await get_template_state(spec, api_key)
        if state is TemplateState.BUILDING:
            state = await _wait_until_ready(spec, api_key, builder_lock=None)
        if state is TemplateState.MISSING:
            # Watching that build may have used up most of the lock.  Take a
            # fresh TTL so the build cannot outlive it; if the lock is no
            # longer ours, someone else is provisioning, so start over as a
            # follower (or the next holder).
            if not await redis.eval(
                _EXTEND_SCRIPT, 1, lock_key, token, _BUILD_LOCK_TTL_SECONDS
            ):
                return await _provision(spec, api_key)
            await asyncio.wait_for(
                build_template(spec, api_key), timeout=_BUILD_TIMEOUT_SECONDS
            )
    finally:
        await redis.eval(_UNLOCK_SCRIPT, 1, lock_key, token)


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


async def get_template_state(spec: TemplateSpec, api_key: str) -> TemplateState:
    """What the team has behind *spec*'s alias right now."""
    async with get_api_client(ConnectionConfig(api_key=api_key)) as client:
        alias = await get_templates_aliases_alias.asyncio_detailed(
            alias=spec.alias, client=client
        )
        if alias.status_code == 404:
            return TemplateState.MISSING
        if alias.status_code == 403:
            # The alias exists on another team: we can neither inspect its
            # builds nor build our own under that name, so use it as-is.
            return TemplateState.READY
        if not isinstance(alias.parsed, TemplateAliasResponse):
            raise RuntimeError(
                f"E2B alias lookup for {spec.alias} failed: HTTP {alias.status_code}"
            )
        template = await get_templates_template_id.asyncio_detailed(
            template_id=alias.parsed.template_id, client=client
        )
    if not isinstance(template.parsed, TemplateWithBuilds):
        raise RuntimeError(
            f"E2B template lookup for {spec.alias} failed: HTTP {template.status_code}"
        )
    statuses = {build.status for build in template.parsed.builds}
    if TemplateBuildStatus.READY in statuses:
        return TemplateState.READY
    if statuses & {TemplateBuildStatus.BUILDING, TemplateBuildStatus.WAITING}:
        return TemplateState.BUILDING
    return TemplateState.MISSING


def forget_ready_templates() -> None:
    """Drop the process-level cache (tests, or after a template is deleted)."""
    _ready.clear()
    _inflight.clear()


def forget_template(template: str, api_key: str) -> None:
    """Re-check *template* on the next ``ensure_template``.

    Called when creating a sandbox from it failed: the template may have
    been deleted out of band since this process confirmed it.
    """
    spec = MANAGED_TEMPLATES.get(template)
    if spec is not None:
        _ready.discard(_scoped_key(spec, api_key))


def _scoped_key(spec: TemplateSpec, api_key: str) -> str:
    """Alias plus a fingerprint of the team's key: aliases live per team.

    The fingerprint only namespaces cache and lock keys.  It is a keyed MAC
    of a fixed label, so the key is never hashed as data and the value does
    not lead back to it.
    """
    team = hmac.new(api_key.encode(), b"e2b-template", hashlib.sha256).hexdigest()
    return f"{spec.alias}@{team[:16]}"


async def _wait_until_ready(
    spec: TemplateSpec, api_key: str, builder_lock: str | None
) -> TemplateState:
    """Wait for a build this process did not start.

    *builder_lock* is the lock the building process holds, or ``None`` when
    this process holds the lock itself and is watching a build started
    elsewhere.  Returns READY, or MISSING once the watched build has failed:
    for a lock holder that is the moment the state drops out of BUILDING, for
    a follower it is MISSING with the builder's lock gone.  Waiting any longer
    would only run out the clock.
    """
    redis = await get_redis_async()
    for _ in range(int(_BUILD_WAIT_SECONDS / _BUILD_POLL_SECONDS)):
        state = await get_template_state(spec, api_key)
        if state is TemplateState.READY:
            return state
        if state is TemplateState.MISSING and (
            builder_lock is None or not await redis.exists(builder_lock)
        ):
            return state
        await asyncio.sleep(_BUILD_POLL_SECONDS)
    raise TimeoutError(
        f"E2B template {spec.alias} was not ready within {_BUILD_WAIT_SECONDS}s"
    )
