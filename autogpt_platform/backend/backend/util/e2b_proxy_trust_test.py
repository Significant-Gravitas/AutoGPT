"""Installing the swap proxy's CA in a box: what is sent, through which door,
and when it is skipped or retried.  The script itself was run against real
intercepting proxies, in Docker and in an E2B desktop box; here it is the
contract around it."""

import contextlib
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from backend.util.e2b_proxy_trust import (
    BUNDLE_FILE,
    CA_FILE,
    PROFILE,
    PROFILE_FILE,
    ca_certs,
    fingerprint,
    trust_proxy_ca,
)
from backend.util.e2b_proxy_trust_script import INSTALL_SCRIPT
from backend.util.sandbox_login import LoginChainChanged

_M = "backend.util.e2b_proxy_trust"
_WITH_BUNDLE = {
    "NODE_EXTRA_CA_CERTS": CA_FILE,
    "REQUESTS_CA_BUNDLE": BUNDLE_FILE,
    "SSL_CERT_FILE": BUNDLE_FILE,
}


def _cert(name: str, ca: bool = True) -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
    )
    if ca:
        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
    cert = builder.sign(key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.PEM).decode().strip()


_PEM = _cert("proxy CA")
_NEXT = _cert("next proxy CA")


def _der_b64(pem: str) -> str:
    return "".join(line for line in pem.splitlines() if "-----" not in line)


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


def _box() -> MagicMock:
    box = MagicMock()
    box.sandbox_id = "sb-1"
    return box


def _install(
    store: dict[str, str],
    stdout: str = "bundle: ok\ninstall: complete\n",
    baseline: bool = False,
):
    """The doors the install can go through, and the baseline it may touch."""
    result = MagicMock(stdout=stdout)
    mocks = {
        "redis": AsyncMock(return_value=_redis(store)),
        "has_baseline": AsyncMock(return_value=baseline),
        "run_internal": AsyncMock(return_value=result),
        "run_sandbox_command": AsyncMock(return_value=result),
        "accept_platform_file": AsyncMock(),
        "read_capped": AsyncMock(return_value=None),
    }
    patches = [
        patch(f"{_M}.get_redis_async", mocks["redis"]),
        *(
            patch(f"{_M}.{name}", mock)
            for name, mock in mocks.items()
            if name != "redis"
        ),
    ]
    return patches, mocks


@contextlib.contextmanager
def _applied(patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


class TestTheSetting:
    def test_unset_is_no_certificate(self):
        with _setting(""):
            assert ca_certs() == []

    def test_a_certificate_is_put_in_the_standard_layout(self):
        """The box splits and matches them line by line."""
        one_line = (
            f"-----BEGIN CERTIFICATE-----\n{_der_b64(_PEM)}\n-----END CERTIFICATE-----"
        )
        with _setting(one_line):
            assert ca_certs() == [_PEM]

    def test_escaped_newlines_from_a_one_line_env_var_are_read(self):
        with _setting(_PEM.replace("\n", "\\n")):
            assert ca_certs() == [_PEM]

    def test_several_certificates_are_kept_in_order(self):
        with _setting(f"{_NEXT}\n\n{_PEM}\n"):
            assert ca_certs() == [_NEXT, _PEM]

    @pytest.mark.parametrize(
        "value",
        [
            # A mistyped delimiter would otherwise drop that certificate.
            f"{_PEM}\n-----BEGIN CERTIFICAT-----\nAAAA\n-----END CERTIFICATE-----",
            f"{_PEM}\nnot a certificate",
            _PEM.replace(_der_b64(_PEM)[:4], "!!!!"),
            # Base64, but not a certificate.
            "-----BEGIN CERTIFICATE-----\nAAECAwQF\n-----END CERTIFICATE-----",
            # A certificate, but not a CA's.
            _cert("a leaf", ca=False),
        ],
    )
    def test_anything_but_ca_certificates_is_refused(self, value):
        with _setting(value), pytest.raises(ValueError):
            ca_certs()

    def test_the_fingerprint_follows_the_set(self):
        assert fingerprint([_PEM]) != fingerprint([_PEM, _NEXT])


class TestTheInstall:
    @pytest.mark.asyncio
    async def test_a_setting_that_does_not_parse_fails_no_box(self):
        patches, mocks = _install({})
        with _setting(_cert("a leaf", ca=False)), _applied(patches):
            assert await trust_proxy_ca(_box()) is None
        mocks["run_sandbox_command"].assert_not_awaited()
        mocks["run_internal"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_without_a_certificate_nothing_runs(self):
        patches, mocks = _install({})
        with _setting(""), _applied(patches):
            assert await trust_proxy_ca(_box()) is None
        mocks["run_sandbox_command"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_new_box_runs_it_capped_as_root_with_no_home(self):
        """It has run nothing yet, chat's or not, baseline or not."""
        store: dict[str, str] = {}
        patches, mocks = _install(store, baseline=True)
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box(), fresh=True, chat=True) == _WITH_BUNDLE
        mocks["run_internal"].assert_not_awaited()
        call = mocks["run_sandbox_command"].await_args
        assert call.args[1] == INSTALL_SCRIPT
        assert call.kwargs["user"] == "root"
        envs = call.kwargs["envs"]
        assert envs["HOME"] == "/nonexistent"
        assert envs["AGPT_CA_PEM"] == _PEM
        assert envs["AGPT_CA_SHA"] == fingerprint([_PEM])
        # The script adds the final newline back: the file is PROFILE exactly.
        assert envs["AGPT_PROFILE"] + "\n" == PROFILE
        # Chrome's policy takes the certificate as base64 DER.
        assert json.loads(envs["AGPT_CHROME_POLICY"]) == {
            "CACertificates": [_der_b64(_PEM)]
        }
        assert store == {"e2b:egress:ca:sb-1": f"{fingerprint([_PEM])} bundle"}

    @pytest.mark.asyncio
    async def test_a_chats_box_reconnects_through_the_login_chain_check(self):
        """The profile it writes is taken into the baseline so it does not
        read as an edit."""
        box, store = _box(), {}
        patches, mocks = _install(store, baseline=True)
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(box, chat=True) == _WITH_BUNDLE
        mocks["run_sandbox_command"].assert_not_awaited()
        assert mocks["run_internal"].await_args.kwargs["user"] == "root"
        mocks["accept_platform_file"].assert_awaited_once_with(
            box, PROFILE_FILE, PROFILE.encode()
        )

    @pytest.mark.asyncio
    async def test_a_chats_box_without_a_baseline_is_refused_not_run_unchecked(self):
        """With no baseline every login file reads as changed, so the check
        refuses: nothing runs unchecked, and nothing is recorded."""
        store: dict[str, str] = {}
        patches, mocks = _install(store, baseline=False)
        mocks["run_internal"].side_effect = LoginChainChanged("no baseline")
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box(), chat=True) is None
        mocks["run_sandbox_command"].assert_not_awaited()
        assert store == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "baseline, guarded", [(False, False), (True, True), (ConnectionError, True)]
    )
    async def test_a_blocks_box_is_checked_when_it_has_a_baseline_or_it_is_unknown(
        self, baseline, guarded
    ):
        patches, mocks = _install({})
        if baseline is ConnectionError:
            mocks["has_baseline"].side_effect = ConnectionError
        else:
            mocks["has_baseline"].return_value = baseline
        with _setting(_PEM), _applied(patches):
            await trust_proxy_ca(_box())
        assert mocks["run_internal"].await_count == int(guarded)
        assert mocks["run_sandbox_command"].await_count == int(not guarded)

    @pytest.mark.asyncio
    async def test_an_edited_login_chain_installs_nothing(self):
        store: dict[str, str] = {}
        patches, mocks = _install(store)
        mocks["run_internal"].side_effect = RuntimeError("login files changed")
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box(), chat=True) is None
        mocks["accept_platform_file"].assert_not_awaited()
        assert store == {}

    @pytest.mark.asyncio
    async def test_a_profile_not_taken_into_the_baseline_is_not_recorded(self):
        store: dict[str, str] = {}
        patches, mocks = _install(store)
        mocks["accept_platform_file"].side_effect = ConnectionError
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box(), chat=True) == _WITH_BUNDLE
        assert store == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "on_disk, repaired", [(PROFILE.encode(), True), (b"curl evil\n", False)]
    )
    async def test_a_left_behind_profile_is_taken_in_only_if_it_is_ours(
        self, on_disk, repaired
    ):
        """Otherwise the check would refuse this install, and every command
        after it, until the baseline expires."""
        box = _box()
        patches, mocks = _install({})
        mocks["read_capped"].return_value = on_disk
        mocks["run_internal"].side_effect = RuntimeError("stop after the repair")
        with _setting(_PEM), _applied(patches):
            await trust_proxy_ca(box, chat=True)
        mocks["read_capped"].assert_awaited_once_with(box, PROFILE_FILE)
        assert mocks["accept_platform_file"].await_count == int(repaired)

    @pytest.mark.asyncio
    async def test_a_box_that_trusts_this_set_is_not_touched(self):
        store = {"e2b:egress:ca:sb-1": f"{fingerprint([_PEM])} bundle"}
        patches, mocks = _install(store)
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box()) == _WITH_BUNDLE
        mocks["run_sandbox_command"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_box_with_no_public_roots_gets_no_bundle_path(self):
        """A bundle of the CA alone would fail every other site."""
        store: dict[str, str] = {}
        patches, _ = _install(store, stdout="install: complete\n")
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box()) == {"NODE_EXTRA_CA_CERTS": CA_FILE}
        assert store["e2b:egress:ca:sb-1"].endswith(" nobundle")
        patches, mocks = _install(store)
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box()) == {"NODE_EXTRA_CA_CERTS": CA_FILE}
        mocks["run_sandbox_command"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_box_that_trusts_an_older_set_installs_again(self):
        store = {"e2b:egress:ca:sb-1": "an-older-set bundle"}
        patches, mocks = _install(store)
        with _setting(_PEM), _applied(patches):
            await trust_proxy_ca(_box())
        mocks["run_sandbox_command"].assert_awaited_once()
        assert store["e2b:egress:ca:sb-1"] == f"{fingerprint([_PEM])} bundle"

    @pytest.mark.asyncio
    async def test_a_partial_install_is_not_recorded_so_it_is_tried_again(self):
        store: dict[str, str] = {}
        patches, _ = _install(
            store, stdout="bundle: ok\ninstall: incomplete: firefox\n"
        )
        with _setting(_PEM), _applied(patches):
            # The clients it did reach still trust the CA.
            assert await trust_proxy_ca(_box()) == _WITH_BUNDLE
        assert store == {}

    @pytest.mark.asyncio
    async def test_a_failed_install_is_not_raised_or_recorded(self):
        store: dict[str, str] = {}
        patches, mocks = _install(store)
        mocks["run_sandbox_command"].side_effect = RuntimeError("exit 1")
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box()) is None
        assert store == {}

    @pytest.mark.asyncio
    async def test_without_redis_it_still_installs(self):
        patches, mocks = _install({})
        mocks["redis"].side_effect = ConnectionError
        with _setting(_PEM), _applied(patches):
            assert await trust_proxy_ca(_box()) == _WITH_BUNDLE
        mocks["run_sandbox_command"].assert_awaited_once()


def test_the_profile_points_python_at_the_bundle_only_when_it_is_there():
    assert f"if [ -s {BUNDLE_FILE} ]" in PROFILE
