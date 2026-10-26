#!/usr/bin/env bash
# Rehearses the Developer ID signing of the macOS app on a Mac that has the
# certificate, before the release workflow does it for real.
#
#   bash build/sign_check_macos.sh <SHA-1 of the identity> [path/to/AutoGPT.app]
#
# It signs a COPY of a packaged app (default: dist/mac-arm64/AutoGPT.app, as
# built by `npx electron-builder --config electron-builder.config.js`) with
# the code the workflow uses (build/mac_sign.js), then reports:
#
#   1. codesign --verify --deep --strict
#   2. every Mach-O file that is unsigned, ad-hoc signed, signed by another
#      team (only the Claude Code CLI may be), or lacks the hardened runtime
#      or a timestamp, and every program that lacks an entitlement it needs
#   3. that the Claude Code CLI is byte for byte what was bundled
#   4. what Gatekeeper says (spctl)
#
# It does not notarize, uploads nothing, and does not import, export or
# change anything in a keychain: it only signs with an identity that is
# already there. The identity is named by its SHA-1 because several
# certificates can share one name:
#
#   security find-identity -v -p codesigning
#
# Run it in Terminal on the Mac itself. Over SSH the login keychain is
# locked and codesign fails with "errSecInternalComponent". The first use of
# the key may ask for permission; that dialog is macOS's, not this script's.
# Signing needs the network (Apple's timestamp server).

set -euo pipefail

usage() {
  sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

identity="${1:-}"
[[ "${identity}" =~ ^[0-9A-Fa-f]{40}$ ]] || usage
desktop="$(cd "$(dirname "$0")/.." && pwd)"
app="${2:-${desktop}/dist/mac-arm64/AutoGPT.app}"

fail() {
  echo "error: $1" >&2
  exit 1
}

[ "$(uname -s)" = "Darwin" ] || fail "this runs on macOS only"
[ -d "${app}/Contents/Resources/runtime" ] || fail "no packaged app with a runtime at ${app}"
command -v node >/dev/null || fail "node is not on PATH"
[ -d "${desktop}/node_modules/@electron/osx-sign" ] || fail "run 'npm ci' in ${desktop} first"

# The identity must be a Developer ID Application certificate in a keychain
# on the search list. Nothing is unlocked or imported here.
found="$(security find-identity -v -p codesigning | grep -i "${identity}" || true)"
[ -n "${found}" ] || fail "no code-signing identity with SHA-1 ${identity} (security find-identity -v -p codesigning)"
case "${found}" in
  *"Developer ID Application:"*) echo "Identity: ${found#*\"}" | sed 's/"$//' ;;
  *) fail "that identity is not a Developer ID Application certificate: ${found}" ;;
esac

work="$(mktemp -d "${TMPDIR:-/tmp}/autogpt-sign-check.XXXXXX")"
copy="${work}/AutoGPT.app"
cleanup() {
  if [ "${KEEP:-0}" = "1" ]; then
    echo "Kept the signed copy: ${copy}"
  else
    rm -rf "${work}"
  fi
}
trap cleanup EXIT

# A clone costs no disk space on APFS until a file is changed; only the
# files that get signed are rewritten.
echo "Copying ${app}"
cp -Rc "${app}" "${copy}" 2>/dev/null || ditto "${app}" "${copy}"

claude="Contents/Resources/runtime/site/claude_agent_sdk/_bundled/claude"
[ -f "${app}/${claude}" ] || fail "the Claude Code CLI is not at ${claude}"

echo
echo "== Signing (the timestamp server is asked once per file; this takes minutes)"
started="$(date +%s)"
node "${desktop}/build/mac_sign.js" sign "${copy}" --identity "${identity}"
echo "Signed in $(($(date +%s) - started)) s"

status=0

echo
echo "== 1. codesign --verify --deep --strict"
if codesign --verify --deep --strict --verbose=2 "${copy}"; then
  echo "ok"
else
  status=1
fi

echo
echo "== 2. Every Mach-O file: Developer ID, hardened runtime, timestamp, entitlements"
node "${desktop}/build/mac_sign.js" check "${copy}" || status=1

echo
echo "== 3. The Claude Code CLI is unchanged"
if cmp "${app}/${claude}" "${copy}/${claude}"; then
  codesign --display --verbose=2 "${copy}/${claude}" 2>&1 | grep -E '^(Authority|TeamIdentifier|Timestamp|CodeDirectory)' | head -6
else
  echo "CHANGED: signing must leave it as Anthropic shipped it"
  status=1
fi

echo
echo "== 4. Gatekeeper"
# Before notarization the expected answer is 'rejected' with
# 'source=Unnotarized Developer ID': the signature is good and only the
# ticket is missing. Any other source means the signature itself is wrong.
verdict="$(spctl --assess --type execute --verbose=4 "${copy}" 2>&1 || true)"
echo "${verdict}"
case "${verdict}" in
  *"Notarized Developer ID"* | *"Unnotarized Developer ID"*) ;;
  *)
    echo "UNEXPECTED: Gatekeeper does not see a Developer ID signature"
    status=1
    ;;
esac

echo
echo "== What each program may do"
for program in \
  "Contents/MacOS/AutoGPT" \
  "Contents/Resources/runtime/python/bin/python3" \
  "Contents/Resources/runtime/node/node" \
  "Contents/Resources/runtime/postgres/bin/postgres"; do
  [ -e "${copy}/${program}" ] || continue
  echo "${program}:"
  codesign --display --entitlements - --xml "${copy}/${program}" 2>/dev/null |
    grep -o '<key>[^<]*</key>' | sed 's/<[^>]*>//g; s/^/  /' || true
done

echo
if [ "${status}" = "0" ]; then
  echo "PASSED. Not notarized: that happens in the release workflow."
else
  echo "FAILED: see above."
fi
exit "${status}"
