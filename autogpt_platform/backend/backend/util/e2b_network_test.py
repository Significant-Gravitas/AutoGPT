"""The egress chokepoint: every SDK create and connect goes through it, and
with a proxy configured every box is pinned under a credential of its own."""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend
from backend.util import e2b_network
from backend.util.e2b_network import (
    EgressOwner,
    connect_sandbox,
    create_sandbox,
    credential_record,
    password_digest,
)

_M = "backend.util.e2b_network"
_OWNER = EgressOwner(kind="expert", id="exp-1", user_id="user-1")
_PROXY = "proxy.agpt.internal:1080"


def _redis(values: dict[str, str] | None = None) -> MagicMock:
    store: dict[str, str] = dict(values or {})
    r = MagicMock()

    async def _get(key):
        return store.get(key, "").encode() or None

    async def _set(key, value, ex=None):
        store[key] = value
        return True

    async def _delete(*keys):
        for key in keys:
            store.pop(key, None)

    r.get = AsyncMock(side_effect=_get)
    r.set = AsyncMock(side_effect=_set)
    r.delete = AsyncMock(side_effect=_delete)
    r.store = store
    return r


def _box(sandbox_id: str = "sb-1") -> MagicMock:
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    sb.update_network = AsyncMock()
    return sb


def _sdk(box: MagicMock) -> MagicMock:
    cls = MagicMock()
    cls.create = AsyncMock(return_value=box)
    cls.connect = AsyncMock(return_value=box)
    return cls


def _configured(address: str | None):
    settings = MagicMock()
    settings.config.e2b_egress_proxy_address = address or ""
    return patch(f"{_M}.Settings", return_value=settings)


class TestOff:
    @pytest.mark.asyncio
    async def test_create_and_connect_are_passed_through_untouched(self):
        box, cls, redis = _box(), None, _redis()
        cls = _sdk(box)
        with _configured(None), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            assert await create_sandbox(cls, _OWNER, template="t", api_key="k") is box
            assert (
                await connect_sandbox(cls, "sb-1", _OWNER, api_key="k", timeout=9)
                is box
            )
        cls.create.assert_awaited_once_with(template="t", api_key="k")
        cls.connect.assert_awaited_once_with("sb-1", api_key="k", timeout=9)
        box.update_network.assert_not_awaited()
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_a_caller_cannot_bring_its_own_network(self):
        with _configured(None), pytest.raises(ValueError, match="decided here"):
            await create_sandbox(_sdk(_box()), _OWNER, network={"allow_out": []})


class TestPinned:
    @pytest.mark.asyncio
    async def test_create_pins_the_box_under_a_credential_the_proxy_can_resolve(self):
        box, redis = _box("sb-1"), _redis()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(cls, _OWNER, template="t", api_key="k")
            proxy = cls.create.await_args.kwargs["network"]["egress_proxy"]
            record = await credential_record(proxy["username"])

        assert proxy["address"] == _PROXY
        assert record == {
            "owner": "expert:exp-1",
            "user_id": "user-1",
            "sandbox_id": "sb-1",
            "password_sha256": password_digest(proxy["password"]),
        }
        # The password itself is in no record: the proxy compares digests.
        assert proxy["password"] not in json.dumps(redis.store)
        assert redis.store["e2b:egress:box:sb-1"] == proxy["username"]

    @pytest.mark.asyncio
    async def test_the_record_exists_before_the_box_does(self):
        """The box's first connection must already resolve at the proxy."""
        redis = _redis()
        cls = _sdk(_box("sb-1"))
        seen: list[int] = []

        async def _create(**kwargs):
            seen.append(
                len([k for k in redis.store if k.startswith("e2b:egress:cred:")])
            )
            return _box("sb-1")

        cls.create = AsyncMock(side_effect=_create)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(cls, _OWNER, template="t")
        assert seen == [1]

    @pytest.mark.asyncio
    async def test_reconnect_repins_under_a_fresh_credential_and_forgets_the_old(self):
        box, redis = _box("sb-1"), _redis()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(cls, _OWNER, template="t")
            first = cls.create.await_args.kwargs["network"]["egress_proxy"]["username"]
            await connect_sandbox(cls, "sb-1", _OWNER, api_key="k", timeout=420)
            second = box.update_network.await_args.args[0]["egress_proxy"]
            assert await credential_record(first) is None
            record = await credential_record(second["username"])

        assert second["username"] != first and second["address"] == _PROXY
        assert record and record["sandbox_id"] == "sb-1"
        cls.connect.assert_awaited_once_with("sb-1", api_key="k", timeout=420)

    @pytest.mark.asyncio
    async def test_a_box_from_before_the_proxy_is_pinned_on_its_first_reconnect(self):
        box, redis = _box("sb-old"), _redis()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await connect_sandbox(cls, "sb-old", _OWNER)
        update = box.update_network.await_args.args[0]
        assert set(update) == {"egress_proxy"}
        assert update["egress_proxy"]["address"] == _PROXY

    @pytest.mark.asyncio
    async def test_a_failed_update_leaves_the_previous_credential_in_place(self):
        box, redis = _box("sb-1"), _redis()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(cls, _OWNER, template="t")
            first = cls.create.await_args.kwargs["network"]["egress_proxy"]["username"]
            box.update_network = AsyncMock(side_effect=RuntimeError("502"))
            with pytest.raises(RuntimeError):
                await connect_sandbox(cls, "sb-1", _OWNER)
            assert await credential_record(first) is not None
        assert redis.store["e2b:egress:box:sb-1"] == first

    @pytest.mark.asyncio
    async def test_a_connect_that_only_pauses_or_kills_does_not_repin(self):
        box, redis = _box("sb-1"), _redis()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await connect_sandbox(cls, "sb-1", _OWNER, apply_network=False)
        box.update_network.assert_not_awaited()
        assert redis.store == {}


def test_every_sdk_create_and_connect_goes_through_the_chokepoint():
    """A new direct call would create a box whose egress nobody pinned."""
    package = Path(backend.__file__).parent
    roots = [package, package.parent / "scripts"]
    chokepoint = Path(e2b_network.__file__).resolve()
    # A call, not a mention: comments and backticked prose are skipped.
    pattern = re.compile(r"(?<![`\w])\w*Sandbox\.(create|connect)\(")
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name.endswith("_test.py") or path.resolve() == chokepoint:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(package.parent)}:{number}")
    assert offenders == [], (
        "Direct E2B create/connect calls outside backend/util/e2b_network.py: "
        + ", ".join(offenders)
    )
