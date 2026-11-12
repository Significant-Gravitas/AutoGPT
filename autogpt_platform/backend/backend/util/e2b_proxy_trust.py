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
door as the others: a chat's box, which has a login baseline, through
``sandbox_login.run_internal`` (it refuses to run when ``/etc``'s login files
were edited, and skips the user's), and the profile script it writes is
taken into that baseline; any other box, through the output limit with no
home directory.  A box being created has run nothing yet.
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
from backend.util.sandbox_login import accept_platform_file, has_baseline, run_internal
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


async def trust_proxy_ca(sandbox: AsyncSandbox) -> Optional[dict[str, str]]:
    """Install the proxy's CA in *sandbox* unless it already trusts this set.

    Returns the environment a process outside a login shell needs to trust
    it, or ``None`` when the box does not.  Best effort: a box that cannot be
    made to trust it still works everywhere but on bound hosts, where TLS
    fails loudly, so a failure is logged and not raised.  Cancellation is the
    caller's to clean up after.
    """
    try:
        certs = ca_certs()
    except ValueError as error:
        logger.error(
            "[E2B] E2B_EGRESS_PROXY_CA_CERT is not a list of CA certificates "
            "(%s): %.12s gets no CA, and TLS to bound hosts will fail",
            error,
            sandbox.sandbox_id,
        )
        return None
    if not certs:
        logger.warning(
            "[E2B] %.12s egresses through the swap proxy but "
            "E2B_EGRESS_PROXY_CA_CERT is empty: TLS to bound hosts will fail",
            sandbox.sandbox_id,
        )
        return None
    want = fingerprint(certs)
    key = _TRUSTED_KEY_PREFIX + sandbox.sandbox_id
    try:
        redis = await get_redis_async()
        recorded = _text(await redis.get(key)).split()
        if recorded[:1] == [want]:
            return _envs(recorded[1:] == ["bundle"])
    except Exception:
        redis = None
        logger.warning("[E2B] Could not read the CA record", exc_info=True)
    try:
        # Unknown counts as guarded: a chat's box is never run unchecked.
        guarded = await has_baseline(sandbox)
    except Exception:
        guarded = True
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
            result = await run_internal(
                sandbox,
                _INSTALL_SCRIPT,
                user="root",
                envs=envs,
                timeout=_INSTALL_TIMEOUT_SECONDS,
            )
        else:
            result = await run_sandbox_command(
                sandbox,
                _INSTALL_SCRIPT,
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
    out = result.stdout or ""
    try:
        await accept_platform_file(sandbox, PROFILE_FILE, PROFILE.encode())
    except Exception:
        logger.warning("[E2B] Could not take in the CA's profile", exc_info=True)
    bundle = "bundle: ok" in out
    if "install: complete" not in out:
        # Not recorded, so the next reconnect tries again.
        logger.warning(
            "[E2B] The swap proxy's CA was only partly installed in %.12s: %s",
            sandbox.sandbox_id,
            " ".join(line for line in out.splitlines() if line.startswith("install:")),
        )
    elif redis is not None:
        try:
            await redis.set(
                key,
                f"{want} {'bundle' if bundle else 'nobundle'}",
                ex=_TRUSTED_TTL,
            )
        except Exception:
            logger.warning("[E2B] Could not record the CA install", exc_info=True)
    return _envs(bundle)


def _text(value: object) -> str:
    if value is None:
        return ""
    return value.decode() if isinstance(value, bytes) else str(value)


# Runs as root in a box of any image: every step but writing the CA itself
# is skipped where its tool or browser is missing, and one that fails is
# reported, which leaves the box to be installed again.  Each certificate
# also goes in a file of its own (``ca-N.crt``): update-ca-certificates,
# certutil and Firefox read only a file's first.
_INSTALL_SCRIPT = r"""
set -u
dir=/usr/local/share/agpt-egress-proxy
if [ "$(cat "$dir/sha256" 2>/dev/null)" = "$AGPT_CA_SHA" ]; then
  [ -s "$dir/bundle.pem" ] && echo "bundle: ok"
  echo "install: complete"
  exit 0
fi
mkdir -p "$dir" && printf '%s\n' "$AGPT_CA_PEM" > "$dir/ca.pem" || exit 1
rm -f "$dir/sha256" "$dir"/ca-*.crt
awk -v out="$dir/ca" '/BEGIN CERTIFICATE/ { f = sprintf("%s-%d.crt", out, n++) }
  f { print > f } /END CERTIFICATE/ { close(f); f = "" }' "$dir/ca.pem" || exit 1
failed=""
fail() { failed="$failed $1"; }

if command -v update-ca-certificates >/dev/null 2>&1; then
  mkdir -p /usr/local/share/ca-certificates
  rm -f /usr/local/share/ca-certificates/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/usr/local/share/ca-certificates/agpt-egress-proxy-${f##*/}"
  done
  update-ca-certificates >/dev/null 2>&1 || fail system-store
fi
if command -v update-ca-trust >/dev/null 2>&1; then
  rm -f /etc/pki/ca-trust/source/anchors/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/etc/pki/ca-trust/source/anchors/agpt-egress-proxy-${f##*/}"
  done
  update-ca-trust extract >/dev/null 2>&1 || fail system-store
fi

# The public roots plus every configured CA, for clients that do not read the
# system store.  With no public roots to start from, no bundle at all: one
# with only the CA would fail every other site.
roots=""
for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt \
         /etc/ssl/cert.pem; do
  [ -s "$f" ] && roots="$f" && break
done
[ -n "$roots" ] || roots=$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null)
if [ -n "$roots" ] && [ -s "$roots" ]; then
  cat "$roots" "$dir/ca.pem" > "$dir/bundle.pem.new" &&
    mv "$dir/bundle.pem.new" "$dir/bundle.pem" && echo "bundle: ok" || fail bundle
else
  rm -f "$dir/bundle.pem"
fi

printf '%s' "$AGPT_PROFILE" > /etc/profile.d/agpt-egress-proxy-ca.sh || fail profile

for d in /etc/opt/chrome/policies/managed /etc/chromium/policies/managed; do
  mkdir -p "$d" && printf '%s\n' "$AGPT_CHROME_POLICY" > "$d/agpt-egress-proxy.json" ||
    fail chrome-policy
done
# The user's NSS database holds what it was given until told otherwise: a
# retired certificate is deleted, not just left out.
if command -v certutil >/dev/null 2>&1 && id user >/dev/null 2>&1; then
  db="$(getent passwd user | cut -d: -f6)/.pki/nssdb"
  su user -s /bin/sh -c "mkdir -p '$db' &&
    { [ -f '$db/cert9.db' ] || certutil -N -d 'sql:$db' --empty-password; } &&
    certutil -L -d 'sql:$db' | awk '/^agpt-egress-proxy-/ { print \$1 }' |
      while read -r n; do certutil -D -d 'sql:$db' -n \"\$n\"; done &&
    for f in '$dir'/ca-*.crt; do
      n=\${f##*/}
      certutil -A -d 'sql:$db' -t C,, -n agpt-egress-proxy-\${n%.crt} -i \"\$f\" || exit 1
    done" >/dev/null 2>&1 || fail nss
fi

# Firefox reads one policies.json: merged into whichever it has, never replaced.
if command -v python3 >/dev/null 2>&1; then
  python3 - "$dir" <<'PY' >/dev/null 2>&1 || fail firefox
import glob, json, os, sys
ours = sys.argv[1] + "/"
targets = ["/etc/firefox/policies/policies.json"] + [
    os.path.join(d, "policies.json")
    for d in ("/usr/lib/firefox/distribution", "/usr/lib/firefox-esr/distribution",
              "/opt/firefox/distribution")
    if os.path.isdir(d)
]
for path in targets:
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {}
    certs = doc.setdefault("policies", {}).setdefault("Certificates", {})
    kept = [p for p in certs.get("Install", []) if not p.startswith(ours)]
    certs["Install"] = kept + sorted(glob.glob(ours + "ca-*.crt"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
PY
fi

if [ -z "$failed" ]; then
  printf '%s\n' "$AGPT_CA_SHA" > "$dir/sha256" && echo "install: complete"
else
  echo "install: incomplete:$failed"
fi
"""
