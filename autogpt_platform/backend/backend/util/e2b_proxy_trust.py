"""Make a box trust the swap proxy's signing CA.

Behind the proxy, a box's TLS to a bound host (GitHub, for one) ends at the
proxy, which presents a certificate it signed with its own CA.  A box that
does not trust that CA fails every such connection, ``gh`` and ``git push``
included.  The CA is one per deployment (``E2B_EGRESS_PROXY_CA_CERT``, the
certificate only: the key stays with the proxy), and boxes run on templates
we do not build (a block's ``template_id``), so it is not baked into an image:
``e2b_network`` installs it on every box it pins, on create and again on a
reconnect whose box does not have this CA yet.

What the install covers, as root inside the box:

- the system store (``update-ca-certificates`` or ``update-ca-trust``), which
  ``gh``, ``git``, ``curl`` and Python's ``ssl`` read;
- ``bundle.pem``, the public roots plus the CA, for ``requests`` and ``httpx``
  (which read certifi's bundle, not the system's), and ``ca.pem`` for Node
  (which reads its own): a login shell gets ``REQUESTS_CA_BUNDLE``,
  ``SSL_CERT_FILE`` and ``NODE_EXTRA_CA_CERTS`` from ``/etc/profile.d``, and
  ``trust_proxy_ca`` returns the same for anything else, such as a code
  interpreter's kernel;
- Chrome and Chromium (the ``CACertificates`` policy, and the user's NSS
  database when ``certutil`` is there) and Firefox (its ``Certificates``
  policy), for the desktop.

The setting may hold several certificates, which is how a rotation goes:
boxes are given the new CA alongside the old one before the proxy signs
with the new key.  A box records which set it trusts, so a changed setting
installs again on the box's next reconnect; so does an install that only
partly worked.

The install is a command the platform runs, so it goes through the same
door as the others.  A box being created has run nothing yet, so it runs
there through the output limit with no home directory.  A reconnect of a
chat's box goes through ``sandbox_login.run_internal``, which refuses to run
when ``/etc``'s login files were edited (or have no baseline) and skips the
user's; the profile script the install writes is taken into that baseline.
A block's box reconnects the way it was created, unless it has a baseline
(a desktop block's), which is then checked too.

Known limits:

- A JavaScript code interpreter kernel is already running when the CA is
  installed, and Node reads ``NODE_EXTRA_CA_CERTS`` only at startup, so
  JavaScript in a code-runner block does not trust the CA.  Python does:
  ``requests`` reads ``REQUESTS_CA_BUNDLE`` per request.
- Firefox keeps a certificate it installed through policy in each profile
  after the policy drops it, and removing it needs ``certutil``, which the
  desktop image lacks.  After a rotation a box's Firefox may still trust the
  retired CA; the box egresses only through the proxy, so nothing else is
  placed to present a certificate it signed.
"""

import base64
import hashlib
import json
import logging
import re
from typing import Optional

from cryptography import x509
from e2b import AsyncSandbox

from backend.data.redis_client import get_redis_async
from backend.util.e2b_proxy_trust_script import INSTALL_SCRIPT
from backend.util.sandbox_login import (
    accept_platform_file,
    has_baseline,
    read_capped,
    run_internal,
)
from backend.util.sandbox_output import run_sandbox_command
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

TRUST_DIR = "/usr/local/share/agpt-egress-proxy"
CA_FILE = f"{TRUST_DIR}/ca.pem"
BUNDLE_FILE = f"{TRUST_DIR}/bundle.pem"
PROFILE_FILE = "/etc/profile.d/agpt-egress-proxy-ca.sh"
# Written byte for byte as here, so the login baseline can take it in.
PROFILE = f"""\
# Written by the platform: trust the egress proxy's CA.
if [ -s {BUNDLE_FILE} ]; then
  : "${{REQUESTS_CA_BUNDLE:={BUNDLE_FILE}}}"; export REQUESTS_CA_BUNDLE
  : "${{SSL_CERT_FILE:={BUNDLE_FILE}}}"; export SSL_CERT_FILE
fi
: "${{NODE_EXTRA_CA_CERTS:={CA_FILE}}}"; export NODE_EXTRA_CA_CERTS
"""

# Which CA set a box trusts, and whether it has a bundle, so a reconnect
# skips the install.  The box's files outlive a pause; the key may lapse
# first, which costs one install.
_TRUSTED_KEY_PREFIX = "e2b:egress:ca:"
_TRUSTED_TTL = 30 * 24 * 3600
# Well inside the 30 s a chat gives a create: the install takes about 1.5 s.
_INSTALL_TIMEOUT_SECONDS = 20

_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----[^-]+-----END CERTIFICATE-----")


def ca_certs() -> list[str]:
    """The proxy's CA certificates as PEM blocks in the standard layout, or
    none when unset.  Raises ``ValueError`` for anything else in the setting:
    text outside a certificate block, or a block that is not a CA
    certificate."""
    value = Settings().config.e2b_egress_proxy_ca_cert
    if not isinstance(value, str):
        return []
    # A value set through a one-line env var may carry literal ``\n``s.
    text = value.replace("\\n", "\n")
    if _PEM_RE.sub("", text).strip():
        raise ValueError("text outside a certificate block")
    return [_pem(_ca_der(block)) for block in _PEM_RE.findall(text)]


def fingerprint(certs: list[str]) -> str:
    return hashlib.sha256("\n".join(certs).encode()).hexdigest()


def _ca_der(pem: str) -> bytes:
    body = "".join(line.strip() for line in pem.splitlines() if "-----" not in line)
    der = base64.b64decode(body, validate=True)
    cert = x509.load_der_x509_certificate(der)
    try:
        is_ca = cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    except x509.ExtensionNotFound:
        is_ca = False
    if not is_ca:
        raise ValueError(f"not a CA certificate: {cert.subject.rfc4514_string()}")
    return der


def _pem(der: bytes) -> str:
    body = base64.b64encode(der).decode()
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "\n".join(
        ["-----BEGIN CERTIFICATE-----", *lines, "-----END CERTIFICATE-----"]
    )


def _envs(bundle: bool) -> dict[str, str]:
    """What a process that does not read ``/etc/profile.d`` needs.  No path
    to a bundle that is not there: it would break every TLS connection."""
    envs = {"NODE_EXTRA_CA_CERTS": CA_FILE}
    if bundle:
        envs |= {"REQUESTS_CA_BUNDLE": BUNDLE_FILE, "SSL_CERT_FILE": BUNDLE_FILE}
    return envs


async def trust_proxy_ca(
    sandbox: AsyncSandbox, *, fresh: bool = False, chat: bool = False
) -> Optional[dict[str, str]]:
    """Install the proxy's CA in *sandbox* unless it already trusts this set.

    *fresh* is a box just created, which has run nothing yet; *chat* is a
    chat's box, whose reconnects always go through the login chain check.
    Returns the environment a process outside a login shell needs to trust
    the CA, or ``None`` when the box does not.  Best effort: a box that
    cannot be made to trust it still works everywhere but on bound hosts,
    where TLS fails loudly, so a failure is logged and not raised.
    Cancellation is the caller's to clean up after.
    """
    certs = _configured_certs(sandbox.sandbox_id)
    if certs is None:
        return None
    want = fingerprint(certs)
    recorded = await _recorded(sandbox.sandbox_id)
    if recorded and recorded[0] == want:
        return _envs(recorded[1:] == ["bundle"])
    out = await _install(
        sandbox, certs, want, guarded=await _guarded(sandbox, fresh, chat)
    )
    if out is None:
        return None
    accepted = await _accept_profile(sandbox)
    bundle = "bundle: ok" in out
    if "install: complete" in out and accepted:
        await _record(sandbox.sandbox_id, want, bundle)
    else:
        # Not recorded, so the next reconnect tries again.
        logger.warning(
            "[E2B] The swap proxy's CA was only partly installed in %.12s: %s%s",
            sandbox.sandbox_id,
            " ".join(line for line in out.splitlines() if line.startswith("install:")),
            "" if accepted else " (profile not taken into the login baseline)",
        )
    return _envs(bundle)


def _configured_certs(sandbox_id: str) -> Optional[list[str]]:
    """The setting's certificates, or ``None`` (logged) when there are none
    to install."""
    try:
        certs = ca_certs()
    except ValueError as error:
        logger.error(
            "[E2B] E2B_EGRESS_PROXY_CA_CERT is not a list of CA certificates "
            "(%s): %.12s gets no CA, and TLS to bound hosts will fail",
            error,
            sandbox_id,
        )
        return None
    if not certs:
        logger.warning(
            "[E2B] %.12s egresses through the swap proxy but "
            "E2B_EGRESS_PROXY_CA_CERT is empty: TLS to bound hosts will fail",
            sandbox_id,
        )
        return None
    return certs


async def _recorded(sandbox_id: str) -> list[str]:
    """The box's record, ``[fingerprint, "bundle" | "nobundle"]``, or empty."""
    try:
        redis = await get_redis_async()
        return _text(await redis.get(_TRUSTED_KEY_PREFIX + sandbox_id)).split()
    except Exception:
        logger.warning("[E2B] Could not read the CA record", exc_info=True)
        return []


async def _record(sandbox_id: str, want: str, bundle: bool) -> None:
    try:
        redis = await get_redis_async()
        await redis.set(
            _TRUSTED_KEY_PREFIX + sandbox_id,
            f"{want} {'bundle' if bundle else 'nobundle'}",
            ex=_TRUSTED_TTL,
        )
    except Exception:
        logger.warning("[E2B] Could not record the CA install", exc_info=True)


async def _guarded(sandbox: AsyncSandbox, fresh: bool, chat: bool) -> bool:
    """Whether the install must pass the login chain check.  A new box has run
    nothing; a chat's box always must, and with no baseline the check
    refuses; any other box must when it has a baseline (a desktop block's),
    or when that cannot be told."""
    if fresh:
        return False
    if chat:
        return True
    try:
        return await has_baseline(sandbox)
    except Exception:
        return True


async def _install(
    sandbox: AsyncSandbox, certs: list[str], want: str, *, guarded: bool
) -> Optional[str]:
    """Run the install script through the door *guarded* calls for; its
    output, or ``None`` when it could not run."""
    envs = {
        "AGPT_CA_PEM": "\n".join(certs),
        "AGPT_CA_SHA": want,
        "AGPT_PROFILE": PROFILE,
        "AGPT_CHROME_POLICY": json.dumps(
            {"CACertificates": [base64.b64encode(_ca_der(c)).decode() for c in certs]}
        ),
    }
    try:
        if guarded:
            await _repair_profile(sandbox)
            result = await run_internal(
                sandbox,
                INSTALL_SCRIPT,
                user="root",
                envs=envs,
                timeout=_INSTALL_TIMEOUT_SECONDS,
            )
        else:
            result = await run_sandbox_command(
                sandbox,
                INSTALL_SCRIPT,
                user="root",
                envs={**envs, "HOME": "/nonexistent"},
                timeout=_INSTALL_TIMEOUT_SECONDS,
            )
    except Exception:
        logger.warning(
            "[E2B] Could not install the swap proxy's CA in %.12s",
            sandbox.sandbox_id,
            exc_info=True,
        )
        return None
    return result.stdout or ""


async def _repair_profile(sandbox: AsyncSandbox) -> None:
    """An earlier install that wrote the profile but could not take it into
    the baseline leaves the login chain check refusing every platform
    command.  The profile is taken in here only if it is byte for byte ours."""
    try:
        if await read_capped(sandbox, PROFILE_FILE) == PROFILE.encode():
            await accept_platform_file(sandbox, PROFILE_FILE, PROFILE.encode())
    except Exception:
        logger.warning("[E2B] Could not check the CA's profile", exc_info=True)


async def _accept_profile(sandbox: AsyncSandbox) -> bool:
    try:
        await accept_platform_file(sandbox, PROFILE_FILE, PROFILE.encode())
        return True
    except Exception:
        logger.warning("[E2B] Could not take in the CA's profile", exc_info=True)
        return False


def _text(value: object) -> str:
    if value is None:
        return ""
    return value.decode() if isinstance(value, bytes) else str(value)
