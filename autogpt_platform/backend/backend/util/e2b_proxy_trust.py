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
  ``trust_envs`` gives the same to anything else, such as a code
  interpreter's kernel;
- Chrome and Chromium (the ``CACertificates`` policy, and the user's NSS
  database when ``certutil`` is there) and Firefox (its ``Certificates``
  policy), for the desktop.

The setting may hold several certificates, which is how a rotation goes:
boxes are given the new CA alongside the old one before the proxy signs
with the new key.  A box records which set it trusts, so a changed setting
installs again on the box's next reconnect.
"""

import base64
import hashlib
import json
import logging
import re
from typing import Optional

from e2b import AsyncSandbox

from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

TRUST_DIR = "/usr/local/share/agpt-egress-proxy"
CA_FILE = f"{TRUST_DIR}/ca.pem"
BUNDLE_FILE = f"{TRUST_DIR}/bundle.pem"

# Which CA set a box trusts, so a reconnect skips the install.  The box's
# files outlive a pause; the key may lapse first, which costs one install.
_TRUSTED_KEY_PREFIX = "e2b:egress:ca:"
_TRUSTED_TTL = 30 * 24 * 3600
_INSTALL_TIMEOUT_SECONDS = 60

_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----[^-]+-----END CERTIFICATE-----")


def ca_certs() -> list[str]:
    """The proxy's CA certificates as PEM blocks in the standard layout (the
    box matches them against its system bundle line by line), or none when
    unset.  Raises ``ValueError`` for a block that is not base64."""
    value = Settings().config.e2b_egress_proxy_ca_cert
    if not isinstance(value, str):
        return []
    # A value set through a one-line env var may carry literal ``\n``s.
    return [_pem(_der(block)) for block in _PEM_RE.findall(value.replace("\\n", "\n"))]


def fingerprint(certs: list[str]) -> str:
    return hashlib.sha256("\n".join(certs).encode()).hexdigest()


def _der(pem: str) -> bytes:
    body = "".join(line.strip() for line in pem.splitlines() if "-----" not in line)
    return base64.b64decode(body, validate=True)


def _pem(der: bytes) -> str:
    body = base64.b64encode(der).decode()
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "\n".join(
        ["-----BEGIN CERTIFICATE-----", *lines, "-----END CERTIFICATE-----"]
    )


async def trust_proxy_ca(sandbox: AsyncSandbox) -> bool:
    """Install the proxy's CA in *sandbox* unless it already trusts this set.

    True when the box trusts it.  Best effort: a box that cannot be made to
    trust it still works everywhere but on bound hosts, where TLS fails
    loudly, so a failure is logged and not raised.
    """
    try:
        certs = ca_certs()
    except ValueError:
        logger.error(
            "[E2B] E2B_EGRESS_PROXY_CA_CERT holds a certificate that is not "
            "valid base64: %.12s gets no CA, and TLS to bound hosts will fail",
            sandbox.sandbox_id,
        )
        return False
    if not certs:
        logger.warning(
            "[E2B] %.12s egresses through the swap proxy but "
            "E2B_EGRESS_PROXY_CA_CERT is empty: TLS to bound hosts will fail",
            sandbox.sandbox_id,
        )
        return False
    want = fingerprint(certs)
    key = _TRUSTED_KEY_PREFIX + sandbox.sandbox_id
    try:
        redis = await get_redis_async()
        recorded = await redis.get(key)
        if recorded is not None and _text(recorded) == want:
            return True
    except Exception:
        redis = None
        logger.warning("[E2B] Could not read the CA record", exc_info=True)
    try:
        result = await sandbox.commands.run(
            _INSTALL_SCRIPT,
            user="root",
            envs={
                "AGPT_CA_PEM": "\n".join(certs),
                "AGPT_CA_SHA": want,
                "AGPT_CHROME_POLICY": json.dumps(
                    {
                        "CACertificates": [
                            base64.b64encode(_der(pem)).decode() for pem in certs
                        ]
                    }
                ),
            },
            timeout=_INSTALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            "[E2B] Could not install the swap proxy's CA in %.12s",
            sandbox.sandbox_id,
            exc_info=True,
        )
        return False
    if "system-store: missing" in (result.stdout or ""):
        logger.warning(
            "[E2B] %.12s has no system certificate store the CA could go in; "
            "only the bundle and browser policies trust it",
            sandbox.sandbox_id,
        )
    if redis is not None:
        try:
            await redis.set(key, want, ex=_TRUSTED_TTL)
        except Exception:
            logger.warning("[E2B] Could not record the CA install", exc_info=True)
    return True


def trust_envs(trusted: bool) -> dict[str, str]:
    """What a process that does not read ``/etc/profile.d`` needs to trust
    the CA, for a box ``trust_proxy_ca`` said yes for.  Empty otherwise: a
    path to a file that is not there would break every TLS connection."""
    if not trusted:
        return {}
    return {
        "REQUESTS_CA_BUNDLE": BUNDLE_FILE,
        "SSL_CERT_FILE": BUNDLE_FILE,
        "NODE_EXTRA_CA_CERTS": CA_FILE,
    }


def _text(value: object) -> Optional[str]:
    return value.decode() if isinstance(value, bytes) else str(value)


# Runs as root in a box of any image: every step but writing the CA itself
# is skipped where its tool or browser is missing.  Each certificate also goes
# in a file of its own (``ca-N.crt``): update-ca-certificates, certutil and
# Firefox read only a file's first.
_INSTALL_SCRIPT = r"""
set -u
dir=/usr/local/share/agpt-egress-proxy
[ "$(cat "$dir/sha256" 2>/dev/null)" = "$AGPT_CA_SHA" ] && exit 0
mkdir -p "$dir" && printf '%s\n' "$AGPT_CA_PEM" > "$dir/ca.pem" || exit 1
rm -f "$dir"/ca-*.crt
awk -v out="$dir/ca" '/BEGIN CERTIFICATE/ { f = sprintf("%s-%d.crt", out, n++) }
  f { print > f } /END CERTIFICATE/ { close(f); f = "" }' "$dir/ca.pem"

if command -v update-ca-certificates >/dev/null 2>&1; then
  mkdir -p /usr/local/share/ca-certificates
  rm -f /usr/local/share/ca-certificates/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/usr/local/share/ca-certificates/agpt-egress-proxy-${f##*/}"
  done
  update-ca-certificates >/dev/null 2>&1
fi
if command -v update-ca-trust >/dev/null 2>&1; then
  rm -f /etc/pki/ca-trust/source/anchors/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/etc/pki/ca-trust/source/anchors/agpt-egress-proxy-${f##*/}"
  done
  update-ca-trust extract >/dev/null 2>&1
fi

# The public roots and the CA, for clients that do not read the system store.
system=""
for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt \
         /etc/ssl/cert.pem; do
  [ -s "$f" ] && system="$f" && break
done
first=$(sed -n '2p' "$dir/ca.pem")
if [ -n "$system" ] && grep -qF -- "$first" "$system"; then
  cp "$system" "$dir/bundle.pem.new"
  echo "system-store: ok"
else
  certifi=$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null)
  cat ${system:-${certifi:-/dev/null}} "$dir/ca.pem" > "$dir/bundle.pem.new"
  echo "system-store: missing"
fi
mv "$dir/bundle.pem.new" "$dir/bundle.pem"

cat > /etc/profile.d/agpt-egress-proxy-ca.sh <<EOF
: "\${REQUESTS_CA_BUNDLE:=$dir/bundle.pem}"; export REQUESTS_CA_BUNDLE
: "\${SSL_CERT_FILE:=$dir/bundle.pem}"; export SSL_CERT_FILE
: "\${NODE_EXTRA_CA_CERTS:=$dir/ca.pem}"; export NODE_EXTRA_CA_CERTS
EOF

for d in /etc/opt/chrome/policies/managed /etc/chromium/policies/managed; do
  mkdir -p "$d" && printf '%s\n' "$AGPT_CHROME_POLICY" > "$d/agpt-egress-proxy.json"
done
if command -v certutil >/dev/null 2>&1 && id user >/dev/null 2>&1; then
  db="$(getent passwd user | cut -d: -f6)/.pki/nssdb"
  su user -s /bin/sh -c "mkdir -p '$db' &&
    { [ -f '$db/cert9.db' ] || certutil -N -d 'sql:$db' --empty-password; } &&
    for f in '$dir'/ca-*.crt; do
      n=\${f##*/}; certutil -A -d 'sql:$db' -t C,, -n agpt-egress-proxy-\${n%.crt} -i \"\$f\"
    done" >/dev/null 2>&1
fi

# Firefox reads one policies.json: merged into whichever it has, never replaced.
if command -v python3 >/dev/null 2>&1; then
  python3 - "$dir" <<'PY' >/dev/null 2>&1
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

printf '%s\n' "$AGPT_CA_SHA" > "$dir/sha256"
"""
