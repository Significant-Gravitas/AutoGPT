"""The egress chokepoint: every SDK create and connect goes through it, and
with a proxy configured every box is pinned under a credential of its own."""

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend
from backend.util import e2b_network
from backend.util.e2b_network import (
    EgressOwner,
    ProxyCredential,
    connect_sandbox,
    create_sandbox,
    credential_record,
    forget_sandbox,
    kill_sandbox,
    secret_digest,
)

_M = "backend.util.e2b_network"
_OWNER = EgressOwner(kind="expert", id="exp-1", user_id="user-1")
_PROXY = "proxy.agpt.internal:1080"


def _redis(values: dict[str, str] | None = None) -> MagicMock:
    store: dict[str, str] = dict(values or {})
    r = MagicMock()

    async def _get(key):
        return store.get(key, "").encode() or None

    async def _set(key, value, ex=None, nx=False):
        if nx and key in store:
            return None
        store[key] = value
        return True

    async def _eval(script, numkeys, key, value):
        # The rotation lock's compare-and-delete release.
        if store.get(key) == value:
            del store[key]
            return 1
        return 0

    async def _delete(*keys):
        for key in keys:
            store.pop(key, None)

    r.get = AsyncMock(side_effect=_get)
    r.set = AsyncMock(side_effect=_set)
    r.delete = AsyncMock(side_effect=_delete)
    r.eval = AsyncMock(side_effect=_eval)
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
    async def test_a_credential_from_before_the_address_was_removed_is_still_revoked(
        self,
    ):
        redis = _redis(
            {"e2b:egress:box:sb-1": "box-a1", "e2b:egress:cred:box-a1": "{}"}
        )
        with _configured(None), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await forget_sandbox("sb-1")
        assert redis.store == {}

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
        minted = ProxyCredential(username="box-a1", secret="s3cr3t-256-bits")
        with (
            _configured(_PROXY),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}._mint", return_value=minted),
        ):
            await create_sandbox(cls, _OWNER, template="t", api_key="k")
            proxy = cls.create.await_args.kwargs["network"]["egress_proxy"]
            record = await credential_record("box-a1")

        # What E2B's host presents to the proxy...
        assert proxy == {
            "address": _PROXY,
            "username": "box-a1",
            "password": "s3cr3t-256-bits",
        }
        # ...and what the proxy finds for it: the owner and a digest, never
        # the secret itself.
        assert record == {
            "owner": "expert:exp-1",
            "user_id": "user-1",
            "sandbox_id": "sb-1",
            "secret_sha256": secret_digest("s3cr3t-256-bits"),
        }
        assert "s3cr3t-256-bits" not in json.dumps(redis.store)
        assert redis.store["e2b:egress:box:sb-1"] == "box-a1"

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
    async def test_a_failed_create_takes_its_credential_with_it(self):
        redis = _redis()
        cls = _sdk(_box())
        cls.create = AsyncMock(side_effect=RuntimeError("502"))
        minted = ProxyCredential(username="box-a1", secret="s")
        with (
            _configured(_PROXY),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}._mint", return_value=minted),
        ):
            with pytest.raises(RuntimeError):
                await create_sandbox(cls, _OWNER, template="t")
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_a_box_whose_bookkeeping_fails_is_killed_not_leaked(self):
        """After create the box is on the meter; if its record cannot be
        written the caller never gets the handle, so nobody else could."""
        box, redis = _box("sb-1"), _redis()
        box.kill = AsyncMock()
        cls = _sdk(box)
        minted = ProxyCredential(username="box-a1", secret="s")
        with (
            _configured(_PROXY),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}._mint", return_value=minted),
            patch(f"{_M}._bind", AsyncMock(side_effect=ConnectionError("redis"))),
        ):
            with pytest.raises(ConnectionError):
                await create_sandbox(cls, _OWNER, template="t")
        box.kill.assert_awaited_once()
        assert "e2b:egress:cred:box-a1" not in redis.store

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
    async def test_a_failed_update_forgets_the_credential_the_box_never_got(self):
        box, redis = _box("sb-1"), _redis()
        box.pause = AsyncMock()
        cls = _sdk(box)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(cls, _OWNER, template="t")
            before = dict(redis.store)
            box.update_network = AsyncMock(side_effect=RuntimeError("502"))
            with pytest.raises(RuntimeError):
                await connect_sandbox(cls, "sb-1", _OWNER)
        # Same records as before the attempt, lock released; and a box that
        # was pinned stays awake on its previous credential.
        assert redis.store == before
        box.pause.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_box_that_was_never_pinned_is_put_back_to_sleep(self):
        """``connect`` resumed it with direct egress and the handle is lost."""
        box, redis = _box("sb-old"), _redis()
        box.pause = AsyncMock()
        box.update_network = AsyncMock(side_effect=RuntimeError("502"))
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            with pytest.raises(RuntimeError):
                await connect_sandbox(_sdk(box), "sb-old", _OWNER)
        box.pause.assert_awaited_once()
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_concurrent_reconnects_leave_the_box_on_a_credential_that_resolves(
        self,
    ):
        """Several turns share an expert's box.  Whatever order the two
        updates land in, the credential the box ends up presenting must be the
        one that is still on record."""
        box, redis = _box("sb-1"), _redis()
        cls = _sdk(box)
        applied: list[str] = []
        first_is_in = asyncio.Event()

        async def _update(network):
            # The update lands at E2B at once; the first caller only hears
            # back after the second has come and gone.
            applied.append(network["egress_proxy"]["username"])
            if not first_is_in.is_set():
                first_is_in.set()
                await asyncio.sleep(0.3)

        box.update_network = AsyncMock(side_effect=_update)
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            first = asyncio.create_task(connect_sandbox(cls, "sb-1", _OWNER))
            await first_is_in.wait()
            await connect_sandbox(cls, "sb-1", _OWNER)
            await first
            presented = applied[-1]
            assert await credential_record(presented) is not None
        assert redis.store["e2b:egress:box:sb-1"] == presented
        credentials = [k for k in redis.store if k.startswith("e2b:egress:cred:")]
        assert credentials == ["e2b:egress:cred:" + presented]

    @pytest.mark.asyncio
    async def test_a_rotation_that_never_gets_its_turn_gives_up(self):
        box, redis = _box("sb-1"), _redis({"e2b:egress:lock:sb-1": "someone-else"})
        with (
            _configured(_PROXY),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}._ROTATION_LOCK_WAIT", 0.3),
        ):
            with pytest.raises(TimeoutError):
                await connect_sandbox(_sdk(box), "sb-1", _OWNER)
        box.update_network.assert_not_awaited()
        assert redis.store == {"e2b:egress:lock:sb-1": "someone-else"}

    @pytest.mark.asyncio
    async def test_a_paused_or_killed_box_has_its_credential_revoked(self):
        box, redis = _box("sb-1"), _redis()
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(_sdk(box), _OWNER, template="t")
            assert redis.store
            await forget_sandbox("sb-1")
            assert redis.store == {}
            await forget_sandbox("sb-never-pinned")

    @pytest.mark.asyncio
    async def test_killing_through_the_handle_revokes_too(self):
        box, redis = _box("sb-1"), _redis()
        box.kill = AsyncMock()
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(_sdk(box), _OWNER, template="t")
            await kill_sandbox(box)
        box.kill.assert_awaited_once()
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_a_kill_that_fails_keeps_the_credential_of_a_box_still_running(self):
        box, redis = _box("sb-1"), _redis()
        box.kill = AsyncMock(side_effect=RuntimeError("e2b error"))
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(return_value=redis)
        ):
            await create_sandbox(_sdk(box), _OWNER, template="t")
            with pytest.raises(RuntimeError):
                await kill_sandbox(box)
        assert redis.store["e2b:egress:box:sb-1"]

    @pytest.mark.asyncio
    async def test_a_revocation_that_fails_does_not_fail_the_kill_it_follows(self):
        with _configured(_PROXY), patch(
            f"{_M}.get_redis_async", AsyncMock(side_effect=ConnectionError("redis"))
        ):
            await forget_sandbox("sb-1")

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


# ``.create(`` / ``.connect(`` calls that are not a sandbox's, in files that
# import the SDK.  Anything else in such a file has to be listed here to pass.
_NOT_A_SANDBOX = ("AsyncVolume.create(",)


def test_every_sdk_create_and_connect_goes_through_the_chokepoint():
    """A new direct call would create a box whose egress nobody pinned.

    Any ``.create(`` or ``.connect(`` in a file that imports the SDK counts,
    not only ``AsyncSandbox.create(``: ``connect`` is also an instance method
    that resumes a box, and the class can be imported under another name.
    """
    package = Path(backend.__file__).parent
    roots = [package, package.parent / "scripts"]
    chokepoint = Path(e2b_network.__file__).resolve()
    imports_sdk = re.compile(r"^\s*(from|import)\s+e2b", re.MULTILINE)
    call = re.compile(r"\.(create|connect)\(")
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name.endswith("_test.py") or path.resolve() == chokepoint:
                continue
            text = path.read_text()
            if not imports_sdk.search(text):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                # Backticked prose in a docstring is a mention, not a call.
                code = re.sub(r"``[^`]*``|`[^`]*`", "", code)
                if call.search(code) and not any(ok in code for ok in _NOT_A_SANDBOX):
                    offenders.append(f"{path.relative_to(package.parent)}:{number}")
    assert offenders == [], (
        "Direct E2B create/connect calls outside backend/util/e2b_network.py: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize(
    "line",
    [
        "sb = await AsyncSandbox.create(template=t)",
        "sb = await Box.connect(sandbox_id)",  # imported under another name
        "await sandbox.connect(timeout=60)",  # the instance method resumes a box
        "sb = await AsyncSandbox.create(",  # the call continues on the next line
    ],
)
def test_the_guard_sees_every_shape_of_a_direct_call(line, tmp_path, monkeypatch):
    package = tmp_path / "backend"
    package.mkdir()
    (package / "sneaky.py").write_text(f"from e2b import AsyncSandbox as Box\n{line}\n")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(backend, "__file__", str(package / "__init__.py"))
    with pytest.raises(AssertionError, match="sneaky.py:2"):
        test_every_sdk_create_and_connect_goes_through_the_chokepoint()
