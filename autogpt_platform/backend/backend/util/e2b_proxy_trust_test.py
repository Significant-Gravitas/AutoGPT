"""Installing the swap proxy's CA in a box: what is sent, and when it is
skipped.  The script itself was run against a real intercepting proxy (curl,
git, requests, urllib and Node over TLS to github.com); here it is the
contract around it."""

import base64
import json
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util import e2b_proxy_trust
from backend.util.e2b_proxy_trust import (
    BUNDLE_FILE,
    CA_FILE,
    ca_certs,
    fingerprint,
    trust_envs,
    trust_proxy_ca,
)

_M = "backend.util.e2b_proxy_trust"
_DER = bytes(range(256)) * 3
_BODY = base64.b64encode(_DER).decode()
_PEM = "\n".join(
    [
        "-----BEGIN CERTIFICATE-----",
        *textwrap.wrap(_BODY, 64),
        "-----END CERTIFICATE-----",
    ]
)


def _setting(value: str):
    settings = MagicMock()
    settings.config.e2b_egress_proxy_ca_cert = value
    return patch(f"{_M}.Settings", return_value=settings)


def _redis(store: dict[str, str]) -> MagicMock:
    r = MagicMock()

    async def _get(key):
        return store[key].encode() if key in store else None

    async def _set(key, value, ex=None):
        store[key] = value

    r.get = AsyncMock(side_effect=_get)
    r.set = AsyncMock(side_effect=_set)
    return r


def _box(stdout: str = "system-store: ok\n") -> MagicMock:
    box = MagicMock()
    box.sandbox_id = "sb-1"
    box.commands.run = AsyncMock(return_value=MagicMock(stdout=stdout))
    return box


class TestTheSetting:
    def test_unset_is_no_certificate(self):
        with _setting(""):
            assert ca_certs() == []

    def test_a_certificate_is_put_in_the_standard_layout(self):
        """The box matches it against its system bundle line by line."""
        one_line = f"-----BEGIN CERTIFICATE-----\n{_BODY}\n-----END CERTIFICATE-----"
        with _setting(one_line):
            assert ca_certs() == [_PEM]

    def test_escaped_newlines_from_a_one_line_env_var_are_read(self):
        with _setting(_PEM.replace("\n", "\\n")):
            assert ca_certs() == [_PEM]

    def test_several_certificates_are_kept_in_order(self):
        other = _PEM.replace(_BODY[:8], "A" * 8)
        with _setting(f"{other}\n\n{_PEM}\n"):
            certs = ca_certs()
        assert len(certs) == 2 and certs[1] == _PEM

    def test_a_block_that_is_not_base64_is_refused(self):
        with _setting(_PEM.replace(_BODY[:4], "!!!!")), pytest.raises(ValueError):
            ca_certs()

    def test_the_fingerprint_follows_the_set(self):
        assert fingerprint([_PEM]) != fingerprint([_PEM, _PEM])


class TestTheInstall:
    @pytest.mark.asyncio
    async def test_a_setting_that_does_not_parse_fails_no_box(self):
        box = _box()
        with _setting(_PEM.replace(_BODY[:4], "!!!!")):
            assert await trust_proxy_ca(box) is False
        box.commands.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_without_a_certificate_nothing_runs(self):
        box = _box()
        with _setting(""):
            assert await trust_proxy_ca(box) is False
        box.commands.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_it_runs_as_root_with_the_certificate_and_is_recorded(self):
        box, store = _box(), {}
        with (
            _setting(_PEM),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=_redis(store))),
        ):
            assert await trust_proxy_ca(box) is True
        call = box.commands.run.await_args
        assert call.args[0] == e2b_proxy_trust._INSTALL_SCRIPT
        assert call.kwargs["user"] == "root"
        envs = call.kwargs["envs"]
        assert envs["AGPT_CA_PEM"] == _PEM
        assert envs["AGPT_CA_SHA"] == fingerprint([_PEM])
        # Chrome's policy takes the certificate as base64 DER.
        assert json.loads(envs["AGPT_CHROME_POLICY"]) == {"CACertificates": [_BODY]}
        assert store == {"e2b:egress:ca:sb-1": fingerprint([_PEM])}

    @pytest.mark.asyncio
    async def test_a_box_that_trusts_this_set_is_not_touched(self):
        box = _box()
        store = {"e2b:egress:ca:sb-1": fingerprint([_PEM])}
        with (
            _setting(_PEM),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=_redis(store))),
        ):
            assert await trust_proxy_ca(box) is True
        box.commands.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_box_that_trusts_an_older_set_installs_again(self):
        box = _box()
        store = {"e2b:egress:ca:sb-1": "an-older-set"}
        with (
            _setting(_PEM),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=_redis(store))),
        ):
            assert await trust_proxy_ca(box) is True
        box.commands.run.assert_awaited_once()
        assert store["e2b:egress:ca:sb-1"] == fingerprint([_PEM])

    @pytest.mark.asyncio
    async def test_a_failed_install_is_not_raised_or_recorded(self):
        box, store = _box(), {}
        box.commands.run = AsyncMock(side_effect=RuntimeError("exit 1"))
        with (
            _setting(_PEM),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=_redis(store))),
        ):
            assert await trust_proxy_ca(box) is False
        assert store == {}

    @pytest.mark.asyncio
    async def test_without_redis_it_still_installs(self):
        box = _box()
        with (
            _setting(_PEM),
            patch(f"{_M}.get_redis_async", AsyncMock(side_effect=ConnectionError)),
        ):
            assert await trust_proxy_ca(box) is True
        box.commands.run.assert_awaited_once()


def test_the_environment_points_only_at_files_an_install_wrote():
    assert trust_envs(False) == {}
    assert trust_envs(True) == {
        "REQUESTS_CA_BUNDLE": BUNDLE_FILE,
        "SSL_CERT_FILE": BUNDLE_FILE,
        "NODE_EXTRA_CA_CERTS": CA_FILE,
    }
