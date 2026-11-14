"""The script ``e2b_proxy_trust`` runs as root in a box to install the swap
proxy's CA, apart so it reads as the shell it is.  Its inputs come in the
environment (``AGPT_CA_PEM``, ``AGPT_CA_SHA``, ``AGPT_PROFILE``,
``AGPT_CHROME_POLICY``); it reports ``bundle: ok`` when it wrote a bundle, and
``install: complete`` or ``install: incomplete: <steps>``."""

# Runs as root in a box of any image: every step but writing the CA itself
# is skipped where its tool or browser is missing, and one that fails is
# reported, which leaves the box to be installed again.  Each certificate
# also goes in a file of its own (``ca-N.crt``): update-ca-certificates,
# certutil and Firefox read only a file's first.
INSTALL_SCRIPT = r"""
set -u
dir=/usr/local/share/agpt-egress-proxy
# One install at a time per box: two reconnects of one box (an expert's
# turns) would otherwise interleave their writes to the same files.
mkdir -p "$dir" || exit 1
if command -v flock >/dev/null 2>&1; then
  exec 9>"$dir.lock" && flock -w 15 9 || exit 1
fi
if [ "$(cat "$dir/sha256" 2>/dev/null)" = "$AGPT_CA_SHA" ]; then
  [ -s "$dir/bundle.pem" ] && echo "bundle: ok"
  echo "install: complete"
  exit 0
fi
printf '%s\n' "$AGPT_CA_PEM" > "$dir/ca.pem" || exit 1
rm -f "$dir/sha256" "$dir"/ca-*.crt
awk -v out="$dir/ca" '/BEGIN CERTIFICATE/ { f = sprintf("%s-%d.crt", out, n++) }
  f { print > f } /END CERTIFICATE/ { close(f); f = "" }' "$dir/ca.pem" || exit 1
failed=""
fail() { failed="$failed $1"; }

if command -v update-ca-certificates >/dev/null 2>&1; then
  mkdir -p /usr/local/share/ca-certificates
  rm -f /usr/local/share/ca-certificates/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/usr/local/share/ca-certificates/agpt-egress-proxy-${f##*/}" ||
      fail system-store
  done
  update-ca-certificates >/dev/null 2>&1 || fail system-store
fi
if command -v update-ca-trust >/dev/null 2>&1; then
  rm -f /etc/pki/ca-trust/source/anchors/agpt-egress-proxy-*.crt
  for f in "$dir"/ca-*.crt; do
    cp "$f" "/etc/pki/ca-trust/source/anchors/agpt-egress-proxy-${f##*/}" ||
      fail system-store
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
  cat "$roots" "$dir/ca.pem" > "$dir/bundle.pem.$$" &&
    mv "$dir/bundle.pem.$$" "$dir/bundle.pem" && echo "bundle: ok" || fail bundle
else
  rm -f "$dir/bundle.pem"
fi

# The final newline is written here, so the file is byte for byte the profile
# the baseline takes in however the variable arrived.
printf '%s\n' "$AGPT_PROFILE" > /etc/profile.d/agpt-egress-proxy-ca.sh || fail profile

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
    listed=\$(certutil -L -d 'sql:$db') &&
    for n in \$(printf '%s\\n' \"\$listed\" | awk '/^agpt-egress-proxy-/ { print \$1 }'); do
      certutil -D -d 'sql:$db' -n \"\$n\" || exit 1
    done &&
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
    # Replaced whole or not at all: a half-written file would not parse on
    # the next try, and Firefox would lose every policy in it.
    with open(path + ".agpt-new", "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(path + ".agpt-new", path)
PY
fi

if [ -z "$failed" ]; then
  printf '%s\n' "$AGPT_CA_SHA" > "$dir/sha256" && echo "install: complete"
else
  echo "install: incomplete:$failed"
fi
"""
