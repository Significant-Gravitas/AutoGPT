#!/usr/bin/env bash
# Refresh packaging/homebrew/*.rb and packaging/scoop/*.json with the real
# url+sha256+version after a tagged release is published to PyPI.
#
# Usage:
#   scripts/update_packaging.sh v0.0.1
#
# Workflow:
#   1. Tag + push v0.0.1 → release.yml builds, publishes to PyPI, attaches
#      dists to the GitHub release.
#   2. Run this script locally with the tag.
#   3. Copy the rendered files into the homebrew tap / scoop bucket repo
#      and open a PR there.
#
# The repo's own packaging/ stubs are also updated in place so they stay
# usable for ad-hoc testing.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <vX.Y.Z>" >&2
  exit 1
fi

TAG="$1"
VERSION="${TAG#v}"
PKG="autogpt-local-executor"
PKG_UNDERSCORE="${PKG//-/_}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOMEBREW_FORMULA="$REPO_ROOT/packaging/homebrew/${PKG}.rb"
SCOOP_MANIFEST="$REPO_ROOT/packaging/scoop/${PKG}.json"

# Fetch sdist + wheel from PyPI and compute sha256s.
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Resolve the PyPI download URLs via the JSON API (avoids hardcoding the
# Files hosted layout, which has changed shape historically).
PYPI_JSON=$(curl -fsSL "https://pypi.org/pypi/${PKG}/${VERSION}/json")

SDIST_URL=$(echo "$PYPI_JSON" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for f in data["urls"]:
    if f["packagetype"] == "sdist":
        print(f["url"])
        break
')
WHEEL_URL=$(echo "$PYPI_JSON" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for f in data["urls"]:
    if f["packagetype"] == "bdist_wheel":
        print(f["url"])
        break
')

if [ -z "$SDIST_URL" ] || [ -z "$WHEEL_URL" ]; then
  echo "::error::could not resolve sdist/wheel URLs from PyPI for ${PKG} ${VERSION}" >&2
  exit 1
fi

echo "Fetching sdist: $SDIST_URL"
curl -fsSL "$SDIST_URL" -o "$TMPDIR/sdist.tar.gz"
SDIST_SHA=$(sha256sum "$TMPDIR/sdist.tar.gz" | awk '{print $1}')

echo "Fetching wheel: $WHEEL_URL"
curl -fsSL "$WHEEL_URL" -o "$TMPDIR/wheel.whl"
WHEEL_SHA=$(sha256sum "$TMPDIR/wheel.whl" | awk '{print $1}')

echo "sdist sha256 = $SDIST_SHA"
echo "wheel sha256 = $WHEEL_SHA"

# --- Update Homebrew formula ---
# Homebrew prefers sdist for source builds.
python3 - "$HOMEBREW_FORMULA" "$VERSION" "$SDIST_URL" "$SDIST_SHA" <<'PY'
import re, sys
path, version, url, sha = sys.argv[1:]
src = open(path).read()
src = re.sub(r'url "PLACEHOLDER-[^"]*"', f'url "{url}"', src)
src = re.sub(r'sha256 "PLACEHOLDER-[^"]*"', f'sha256 "{sha}"', src)
src = re.sub(r'version "[^"]*"', f'version "{version}"', src)
# Also patch any non-placeholder values from a prior run.
src = re.sub(r'url "https://files\.pythonhosted\.org/[^"]*"', f'url "{url}"', src)
open(path, "w").write(src)
PY

# --- Update Scoop manifest ---
python3 - "$SCOOP_MANIFEST" "$VERSION" "$WHEEL_URL" "$WHEEL_SHA" <<'PY'
import json, sys
path, version, url, sha = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
data["version"] = version
data["url"] = url
data["hash"] = sha
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

echo ""
echo "Updated:"
echo "  $HOMEBREW_FORMULA"
echo "  $SCOOP_MANIFEST"
echo ""
echo "Next:"
echo "  1. Review the diffs."
echo "  2. Copy them into the tap / bucket repos:"
echo "     - Significant-Gravitas/homebrew-tap → Formula/${PKG}.rb"
echo "     - Significant-Gravitas/scoop-bucket → bucket/${PKG}.json"
echo "  3. Open PRs against those repos."
