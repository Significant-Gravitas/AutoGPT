"""The box scripts against the real Redis (a cluster in CI): the unit tests
fake their semantics, so only here do the Lua, the key counts and the
cluster slots actually run."""

import uuid

import pytest

from backend.copilot.integration_creds import grant_to_box, granted_to_box
from backend.data.redis_client import get_redis_async

from .e2b_sandbox import (
    SandboxOwner,
    _acquire_turn,
    _active_turns_key,
    _pin_ceilings_key,
    _release_turn,
    _shared_ceiling,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_pins_narrow_each_other_and_a_release_drops_only_its_own():
    owner = SandboxOwner(kind="expert", id=f"ceiling-{uuid.uuid4().hex}")
    redis = await get_redis_async()
    try:
        assert await _shared_ceiling(owner, ("slack",), None, "s1") == ("slack",)
        # A failed script would fall back to this pin's own (None).
        assert await _shared_ceiling(owner, None, None, "s2") == ("slack",)
        await _acquire_turn(owner)
        # The release script touches the counter and the ceilings: one slot.
        assert await _release_turn(owner, "s1") is True
        assert await _shared_ceiling(owner, None, None, "s3") is None
    finally:
        await redis.delete(_active_turns_key(owner))
        await redis.delete(_pin_ceilings_key(owner))


async def test_a_grant_is_recorded_with_its_expiry():
    sandbox_id = f"sb-{uuid.uuid4().hex}"
    redis = await get_redis_async()
    key = f"e2b:egress:grant:{sandbox_id}:github"
    try:
        await grant_to_box(sandbox_id, {"github": "cred-a"})
        assert await granted_to_box(sandbox_id, "github") == {"cred-a"}
        assert await redis.ttl(key) > 0
    finally:
        await redis.delete(key)
